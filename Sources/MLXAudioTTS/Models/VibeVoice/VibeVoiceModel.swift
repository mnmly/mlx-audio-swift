import Foundation
import HuggingFace
import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

public enum VibeVoiceError: Swift.Error, LocalizedError {
    case invalidRepo(String)
    case missingWeights(URL)
    case missingVoicesDirectory(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidRepo(let repo):
            return "Invalid HuggingFace repo: \(repo)"
        case .missingWeights(let url):
            return "No model.safetensors found in \(url.path)"
        case .missingVoicesDirectory(let url):
            return """
                No voices/ directory in \(url.path). VibeVoice-Realtime ships speakers as \
                prefilled prompts; convert upstream's demo/voices/streaming_model/*.pt and \
                place them in voices/.
                """
        }
    }
}

/// Binary EOS head over the TTS backbone's last hidden state.
final class VibeVoiceEOSClassifier: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int) {
        _fc1.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        _fc2.wrappedValue = Linear(hiddenSize, 1, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(relu(fc1(x)))
    }
}

/// VibeVoice-Realtime-0.5B: streaming, single-speaker text to speech.
///
/// The backbone is a Qwen2 0.5B split in two: the lower layers encode text only, the upper
/// layers see both text and speech. Generation interleaves five text tokens with six 7.5 Hz
/// acoustic latents, each latent produced by a 20-step DPM-Solver++ diffusion sampler with
/// classifier-free guidance and decoded to audio immediately, so speech starts long before
/// the text is exhausted.
public final class VibeVoiceModel: Module, SpeechGenerationModel, @unchecked Sendable {
    /// Text tokens consumed per outer iteration.
    static let textWindowSize = 5
    /// Acoustic latents produced per outer iteration.
    static let speechWindowSize = 6

    @ModuleInfo(key: "language_model") var languageModel: VibeVoiceQwen2Model
    @ModuleInfo(key: "tts_language_model") var ttsLanguageModel: VibeVoiceQwen2Model
    @ModuleInfo(key: "tts_input_types") var ttsInputTypes: Embedding
    @ModuleInfo(key: "acoustic_tokenizer") var acousticDecoder: VibeVoiceTokenizerDecoder
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "prediction_head") var predictionHead: VibeVoiceDiffusionHead
    @ModuleInfo(key: "tts_eos_classifier") var ttsEosClassifier: VibeVoiceEOSClassifier

    @ParameterInfo(key: "speech_scaling_factor") var speechScalingFactor: MLXArray
    @ParameterInfo(key: "speech_bias_factor") var speechBiasFactor: MLXArray

    public let config: VibeVoiceConfiguration
    public let tokenizer: Tokenizers.Tokenizer
    public let voicesDirectory: URL

    public var sampleRate: Int { 24000 }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 4096, temperature: 0)
    }

    /// Classifier-free guidance scale for the acoustic diffusion sampler.
    public var cfgScale: Float = 1.5
    /// Diffusion steps per latent; the checkpoint is tuned for 20.
    public var diffusionSteps: Int

    /// Supplies the initial noise for each latent's reverse diffusion, as `[2, latentSize]`.
    ///
    /// Exists so a test can replay the exact noise the PyTorch reference drew and compare
    /// waveforms sample by sample; generation is otherwise stochastic and two correct
    /// implementations will not agree. Left nil, the model samples its own noise.
    var noiseProvider: (() -> MLXArray)?

    public init(
        config: VibeVoiceConfiguration,
        tokenizer: Tokenizers.Tokenizer,
        voicesDirectory: URL
    ) {
        self.config = config
        self.tokenizer = tokenizer
        self.voicesDirectory = voicesDirectory
        self.diffusionSteps = config.diffusionHeadConfig.ddpmNumInferenceSteps

        let decoderConfig = config.decoderConfig
        let hidden = decoderConfig.hiddenSize

        // The lower stack keeps the embedding table and drops the final norm (upstream
        // replaces it with an identity). The upper stack is the mirror image: its embedding
        // table is never read, because every position's embedding is overwritten by the
        // spliced hidden state before the stack runs.
        _languageModel.wrappedValue = VibeVoiceQwen2Model(
            decoderConfig.withLayerCount(config.languageModelLayers),
            includeEmbedding: true,
            includeFinalNorm: false)
        _ttsLanguageModel.wrappedValue = VibeVoiceQwen2Model(
            decoderConfig.withLayerCount(config.ttsBackboneNumHiddenLayers),
            includeEmbedding: false,
            includeFinalNorm: true)

        _ttsInputTypes.wrappedValue = Embedding(embeddingCount: 2, dimensions: hidden)
        _acousticDecoder.wrappedValue = VibeVoiceTokenizerDecoder(
            config: config.acousticTokenizerConfig, dimension: config.acousticVaeDim)
        _acousticConnector.wrappedValue = VibeVoiceSpeechConnector(
            inputDim: config.acousticVaeDim, outputDim: hidden)
        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(config.diffusionHeadConfig)
        _ttsEosClassifier.wrappedValue = VibeVoiceEOSClassifier(hiddenSize: hidden)

        _speechScalingFactor.wrappedValue = MLXArray(Float(1))
        _speechBiasFactor.wrappedValue = MLXArray(Float(0))

        super.init()
    }

    // MARK: - Generation

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        var chunks: [MLXArray] = []
        try runGeneration(
            text: text, voice: voice, generationParameters: generationParameters,
            onAudioChunk: { chunks.append($0) }, onInfo: { _ in })
        guard !chunks.isEmpty else { return MLXArray.zeros([0]) }
        let audio = concatenated(chunks, axis: 0)
        eval(audio)
        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Swift.Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language, generationParameters: generationParameters,
            streamingInterval: 0.32)
    }

    /// Emits audio as it is decoded rather than at the end — the point of this model.
    /// `streamingInterval` is rounded to whole 7.5 Hz latents, with a floor of one.
    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Swift.Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Swift.Error>.makeStream()

        let framesPerChunk = Swift.max(
            1, Int((streamingInterval * Double(framesPerSecond)).rounded()))

        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish()
                return
            }
            do {
                var pending: [MLXArray] = []
                try self.runGeneration(
                    text: text, voice: voice, generationParameters: generationParameters,
                    onAudioChunk: { chunk in
                        pending.append(chunk)
                        if pending.count >= framesPerChunk {
                            continuation.yield(.audio(concatenated(pending, axis: 0)))
                            pending.removeAll(keepingCapacity: true)
                        }
                    },
                    onInfo: { continuation.yield(.info($0)) })
                if !pending.isEmpty {
                    continuation.yield(.audio(concatenated(pending, axis: 0)))
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    private var framesPerSecond: Int {
        sampleRate / config.acousticTokenizerConfig.compressionRatio
    }

    /// The interleaved text/speech loop. Runs synchronously; callers wrap it in a task.
    private func runGeneration(
        text: String,
        voice: String?,
        generationParameters: GenerateParameters,
        onAudioChunk: (MLXArray) -> Void,
        onInfo: (AudioGenerationInfo) -> Void
    ) throws {
        let start = Date()
        let promptURL = try VibeVoiceVoicePrompt.resolve(name: voice, in: voicesDirectory)
        let prompt = try VibeVoiceVoicePrompt.load(from: promptURL)

        // Match whatever precision the weights were loaded at.
        let dtype = ttsInputTypes.weight.dtype

        // Three caches: the text-only stack, the TTS stack, and the TTS stack's
        // classifier-free-guidance twin. The prompt also carries a fourth, `neg_lm`, which
        // upstream loads but never reads — the negative branch is only ever fed speech
        // embeddings, so it never reaches the text-only stack.
        let lmCache = VibeVoiceVoicePrompt.makeSeededCache(prompt.languageModel, dtype: dtype)
        let ttsCache = VibeVoiceVoicePrompt.makeSeededCache(
            prompt.ttsLanguageModel, dtype: dtype)
        let negativeCache = VibeVoiceVoicePrompt.makeSeededCache(
            prompt.negativeTTSLanguageModel, dtype: dtype)

        // Upstream appends a newline before tokenizing.
        let textIDs = tokenizer.encode(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n",
            addSpecialTokens: false)

        var ttsHidden = prompt.ttsLanguageModel.lastHiddenState.asType(dtype)
        var negativeHidden = prompt.negativeTTSLanguageModel.lastHiddenState.asType(dtype)

        // Upstream stops when the TTS stack's sequence would exceed its context window;
        // `maxTokens` narrows that further. Both text and speech occupy positions.
        let maxNewPositions = Swift.min(
            generationParameters.maxTokens ?? defaultGenerationParameters.maxTokens ?? 4096,
            config.decoderConfig.maxPositionEmbeddings - prompt.ttsLanguageModel.length)

        acousticDecoder.resetState()

        var windowIndex = 0
        var generatedFrames = 0
        var consumedText = 0
        var finished = false

        while !finished {
            try Task.checkCancellation()

            // Each iteration consumes up to five text tokens and then emits six latents;
            // once the text runs out the loop keeps emitting until the EOS head fires.
            let windowStart = windowIndex * Self.textWindowSize
            windowIndex += 1

            if windowStart < textIDs.count {
                let windowEnd = Swift.min(windowStart + Self.textWindowSize, textIDs.count)
                let window = Array(textIDs[windowStart ..< windowEnd])
                consumedText += window.count

                let ids = MLXArray(window.map { Int32($0) }, [1, window.count])
                let lmHidden = languageModel(inputsEmbeds: languageModel.embed(ids), cache: lmCache)
                ttsHidden = forwardTTS(
                    hidden: lmHidden, isText: true, cache: ttsCache)
            } else if textIDs.isEmpty {
                break
            }

            for _ in 0 ..< Self.speechWindowSize {
                try Task.checkCancellation()

                if generatedFrames + consumedText >= maxNewPositions {
                    finished = true
                    break
                }

                let latent = sampleSpeechLatent(
                    condition: ttsHidden[0..., -1, 0...],
                    negativeCondition: negativeHidden[0..., -1, 0...])

                // Undo the training-time normalisation before decoding.
                let scaled = latent / speechScalingFactor - speechBiasFactor
                let audio = acousticDecoder.step(
                    scaled.reshaped([1, 1, -1]).transposed(0, 2, 1).asType(dtype))
                if audio.dim(2) > 0 {
                    let samples = audio.reshaped([-1]).asType(.float32)
                    eval(samples)
                    onAudioChunk(samples)
                }
                generatedFrames += 1

                let acousticEmbed = acousticConnector(
                    latent.reshaped([1, 1, -1]).asType(dtype))

                ttsHidden = forwardTTS(hidden: acousticEmbed, isText: false, cache: ttsCache)
                negativeHidden = forwardTTS(
                    hidden: acousticEmbed, isText: false, cache: negativeCache)

                let eosLogit = ttsEosClassifier(ttsHidden[0..., -1, 0...])
                if sigmoid(eosLogit).item(Float.self) > 0.5 {
                    finished = true
                    break
                }

                if generatedFrames % 50 == 0 {
                    Memory.clearCache()
                }
            }
        }

        let elapsed = Date().timeIntervalSince(start)
        let seconds = Double(generatedFrames) / Double(framesPerSecond)
        onInfo(AudioGenerationInfo(
            promptTokenCount: prompt.ttsLanguageModel.length,
            generationTokenCount: generatedFrames,
            prefillTime: 0,
            generateTime: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(generatedFrames) / elapsed : 0,
            peakMemoryUsage: Double(GPU.peakMemory) / 1024 / 1024 / 1024))
    }

    /// One step of the upper stack.
    ///
    /// The incoming hidden state fully replaces the position's token embedding — upstream
    /// embeds a placeholder id and overwrites it, which is why the upper stack's embedding
    /// table is never needed — and a two-entry type embedding marks text versus speech.
    private func forwardTTS(hidden: MLXArray, isText: Bool, cache: [KVCache]) -> MLXArray {
        let marker = MLXArray([Int32(isText ? 1 : 0)], [1, 1])
        let embeds = hidden + ttsInputTypes(marker)
        return ttsLanguageModel(inputsEmbeds: embeds, cache: cache)
    }

    /// Reverse diffusion for a single acoustic latent, with classifier-free guidance.
    ///
    /// The positive and negative conditions are stacked into one batch of two, and the same
    /// noisy latent is denoised under both; the guided estimate is
    /// `uncond + scale * (cond - uncond)`.
    private func sampleSpeechLatent(condition: MLXArray, negativeCondition: MLXArray) -> MLXArray {
        let solver = VibeVoiceDPMSolver(
            numTrainTimesteps: config.diffusionHeadConfig.ddpmNumSteps,
            betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
            predictionType: config.diffusionHeadConfig.predictionType)
        solver.setTimesteps(diffusionSteps)

        let latentSize = config.acousticVaeDim
        let conditions = concatenated(
            [condition.reshaped([1, -1]), negativeCondition.reshaped([1, -1])], axis: 0)

        var speech = (noiseProvider?() ?? MLXRandom.normal([2, latentSize])).asType(.float32)

        for t in solver.timesteps {
            let half = speech[0 ..< 1, 0...]
            let combined = concatenated([half, half], axis: 0)
            let timesteps = MLXArray([Float(t), Float(t)])

            let eps = predictionHead(
                noisyImages: combined.asType(conditions.dtype),
                timesteps: timesteps,
                condition: conditions).asType(.float32)

            let condEps = eps[0 ..< 1, 0...]
            let uncondEps = eps[1 ..< 2, 0...]
            let guided = uncondEps + cfgScale * (condEps - uncondEps)

            speech = solver.step(
                modelOutput: concatenated([guided, guided], axis: 0), sample: speech)
        }

        return speech[0 ..< 1, 0...]
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> VibeVoiceModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw VibeVoiceError.invalidRepo(modelRepo)
        }
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: [
                "voices/*", "*.json", "tokenizer*", "vocab*", "merges*", "special_tokens*",
            ],
            cache: cache)
        return try await fromModelDirectory(dir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> VibeVoiceModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(
            VibeVoiceConfiguration.self, from: Data(contentsOf: configURL))

        let voicesDirectory = modelDir.appendingPathComponent("voices")
        guard FileManager.default.fileExists(atPath: voicesDirectory.path) else {
            throw VibeVoiceError.missingVoicesDirectory(modelDir)
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        let model = VibeVoiceModel(
            config: config, tokenizer: tokenizer, voicesDirectory: voicesDirectory)

        let weightsURL = modelDir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw VibeVoiceError.missingWeights(modelDir)
        }
        let weights = try MLX.loadArrays(url: weightsURL)
        try model.update(
            parameters: ModuleParameters.unflattened(sanitize(weights: weights)),
            verify: .all)
        model.train(false)
        eval(model)
        return model
    }

    /// Maps the checkpoint onto this module tree.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, value) in weights {
            // The upper stack's embedding table is 136M parameters that are never read:
            // every position it would embed is overwritten by the spliced hidden state.
            if key == "model.tts_language_model.embed_tokens.weight" { continue }

            if key.hasPrefix("model.acoustic_tokenizer.") { continue }
            if key.hasPrefix("model.prediction_head.") { continue }

            if key.hasPrefix("model.") {
                out[String(key.dropFirst("model.".count))] = value
            } else {
                out[key] = value
            }
        }

        // The tokenizer decoder and the diffusion head need their own index remapping.
        for (key, value) in VibeVoiceAcousticTokenizer.sanitize(
            weights: weights, prefix: "model.acoustic_tokenizer.decoder.") {
            out["acoustic_tokenizer.\(key)"] = value
        }
        for (key, value) in VibeVoiceDiffusionHead.sanitize(
            weights: weights, prefix: "model.prediction_head.") {
            out["prediction_head.\(key)"] = value
        }

        return out
    }
}
