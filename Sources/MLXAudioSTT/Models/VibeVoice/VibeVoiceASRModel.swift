import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

public enum VibeVoiceASRError: Swift.Error, LocalizedError {
    case invalidRepo(String)
    case missingConfig(URL)
    case missingWeights(URL)
    case missingTokenizer(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidRepo(let repo):
            return "Invalid HuggingFace repo: \(repo)"
        case .missingConfig(let url):
            return "No config.json in \(url.path)"
        case .missingWeights(let url):
            return "No *.safetensors in \(url.path)"
        case .missingTokenizer(let url):
            return """
                No tokenizer in \(url.path). The original VibeVoice-ASR checkpoints ship \
                without one; copy the files from Qwen/Qwen2.5-7B, or use \
                microsoft/VibeVoice-ASR-HF, which bundles its own.
                """
        }
    }
}

/// VibeVoice ASR: transcription with speaker diarization.
///
/// Audio is encoded twice — an acoustic and a semantic σ-VAE encoder, both at 7.5 Hz — and
/// the two projections are **summed** before being spliced into a ChatML prompt in place of
/// a run of placeholder tokens. A Qwen2.5-7B decoder then emits a JSON array of segments.
public final class VibeVoiceASRModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "acoustic_tokenizer_encoder") var acousticEncoder: VibeVoiceTokenizerEncoder
    @ModuleInfo(key: "semantic_tokenizer_encoder") var semanticEncoder: VibeVoiceTokenizerEncoder
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "semantic_connector") var semanticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "language_model") var languageModel: VibeVoiceQwen2Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: VibeVoiceASRConfiguration
    public let tokenizer: Tokenizers.Tokenizer

    /// Normalise input to -25 dBFS before encoding. The non-streaming checkpoints were
    /// trained with this on; the streaming one ships `normalize_audio: false`.
    public var normalizeAudio: Bool

    /// Draw from the acoustic VAE instead of taking its mean.
    ///
    /// Upstream samples at inference (two `randn` draws per encode), which makes transcripts
    /// non-deterministic for no accuracy gain; its own vLLM plugin exposes `VIBEVOICE_USE_MEAN`
    /// to turn that off. Off by default here for the same reason — turn it on only to
    /// reproduce the reference's exact behaviour.
    public var sampleAcousticLatents = false

    /// Window geometry for `VibeVoiceASRStreamSession`, in 7.5 Hz frames.
    ///
    /// A streaming checkpoint states these in its `preprocessor_config.json` and only works
    /// at the size it was trained on; the defaults match `VibeVoice-ASR-Streaming-7B`
    /// (2.93 s of advance, 0.53 s of lookahead).
    public var streamingChunkFrames = 22
    public var streamingLookaheadFrames = 4

    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(maxTokens: 8192, temperature: 0)
    }

    /// Both tokenizers are trained at 24 kHz, unlike the 16 kHz recognizers around it.
    public var sampleRate: Int { config.samplingRate }

    public init(
        config: VibeVoiceASRConfiguration,
        tokenizer: Tokenizers.Tokenizer,
        normalizeAudio: Bool
    ) {
        self.config = config
        self.tokenizer = tokenizer
        self.normalizeAudio = normalizeAudio

        let hidden = config.textConfig.hiddenSize
        _acousticEncoder.wrappedValue = VibeVoiceTokenizerEncoder(
            config: config.acousticTokenizer, dimension: config.acousticTokenizer.vaeDim)
        _semanticEncoder.wrappedValue = VibeVoiceTokenizerEncoder(
            config: config.semanticTokenizer, dimension: config.semanticTokenizer.vaeDim)
        _acousticConnector.wrappedValue = VibeVoiceSpeechConnector(
            inputDim: config.acousticTokenizer.vaeDim, outputDim: hidden)
        _semanticConnector.wrappedValue = VibeVoiceSpeechConnector(
            inputDim: config.semanticTokenizer.vaeDim, outputDim: hidden)
        _languageModel.wrappedValue = VibeVoiceQwen2Model(config.textConfig)
        _lmHead.wrappedValue = Linear(hidden, config.textConfig.vocabSize, bias: false)

        super.init()
    }

    // MARK: - Audio front end

    /// Audio (mono, 24 kHz) to language-model embeddings, one per 3200 samples.
    public func encodeAudio(_ audio: MLXArray) -> MLXArray {
        // Pad up to a whole number of 3200-sample frames before encoding.
        //
        // The offline convolution path already ceil-pads internally, but the segmented one
        // used for long audio does not, so a final frame covering only a handful of real
        // samples came out wrong — measured against the reference on a 180 s clip whose
        // last frame held 32 samples, that single frame dragged the latents from 68 dB SNR
        // down to 37 dB. Zero-padding here makes every level's length an exact multiple of
        // its remaining hop, so no level needs its own alignment padding and both paths
        // agree.
        let samples = paddedToWholeFrames(normalizedSamples(audio))
        let dtype = lmHead.weight.dtype
        let expectedFrames = samples.size / config.compressionRatio

        var acoustic = acousticLatents(samples)
        let semantic = encodeSegmented(
            samples.asType(audioEncoderDType), encoder: semanticEncoder)

        // Only the acoustic branch is a distribution; upstream takes the semantic one's
        // mean unconditionally.
        if sampleAcousticLatents {
            acoustic = VibeVoiceAcousticTokenizer.sample(
                acoustic, config: config.acousticTokenizer)
        }

        // NCL -> NLC so the connectors see channel-last features, back at the LM's
        // precision since the connectors belong to the language side.
        var acousticFeatures = acousticConnector(swappedAxes(acoustic, 1, 2).asType(dtype))
        var semanticFeatures = semanticConnector(swappedAxes(semantic, 1, 2).asType(dtype))

        // The prompt reserves exactly ceil(samples / 3200) placeholders, so the feature
        // count has to agree or the splice would misalign.
        acousticFeatures = alignFrames(acousticFeatures, to: expectedFrames)
        semanticFeatures = alignFrames(semanticFeatures, to: expectedFrames)

        // The two projections are summed, not concatenated.
        return acousticFeatures + semanticFeatures
    }

    /// Precision the audio encoders run at, read from their own weights.
    var audioEncoderDType: DType { acousticEncoder.stem.conv.conv.weight.dtype }

    /// Acoustic latents (NCL) for already-normalised, frame-aligned samples.
    func acousticLatents(_ samples: MLXArray) -> MLXArray {
        encodeSegmented(samples.asType(audioEncoderDType), encoder: acousticEncoder)
    }

    /// Normalises and frame-aligns raw audio, as `encodeAudio` does before encoding.
    func preparedSamples(_ audio: MLXArray) -> MLXArray {
        paddedToWholeFrames(normalizedSamples(audio))
    }

    /// Longest prompt slice fed to the language model in one forward pass.
    ///
    /// Two reasons. The usual one: a bounded lazy graph keeps prefill memory flat, as
    /// mlx-lm's `prefill_step_size` does. The essential one: mlx-swift 0.31.x (bundling MLX
    /// 0.31.1) JIT-compiles the "NAX" split-K GEMM that M5-class GPUs use for a matmul once
    /// M·N ≥ 2048², K ≥ 10240 and K ≥ 3·max(M, N) — this model's `down_proj` on any
    /// prompt of ~1171 tokens or more — and instantiates that kernel with the float32
    /// accumulator's type instead of the input's, so it reads bfloat16/float16 activations
    /// as float32 and returns garbage. Every layer's MLP is corrupted at once, and greedy
    /// decoding degenerates into a repetition loop from the first token; float32 is only
    /// unaffected because there the two types coincide. Fixed upstream in MLX ≥ 0.32.0
    /// (ml-explore/mlx#3810), which no mlx-swift release bundles yet. Keeping every slice's
    /// M·N under 2048² never reaches that kernel; once the dependency moves past the fix,
    /// this can be relaxed to a plain memory bound.
    var prefillStepSize: Int {
        Swift.min(1024, (2048 * 2048 - 1) / config.textConfig.hiddenSize)
    }

    /// Runs the prompt through the language model in bounded slices, returning the hidden
    /// states of the final slice.
    func prefill(_ embeds: MLXArray, cache: [KVCache]) -> MLXArray {
        let total = embeds.dim(1)
        let step = prefillStepSize
        var start = 0
        while start + step < total {
            let slice = embeds[0..., start ..< (start + step), 0...]
            eval(languageModel(inputsEmbeds: slice, cache: cache))
            Memory.clearCache()
            start += step
        }
        return languageModel(inputsEmbeds: embeds[0..., start..., 0...], cache: cache)
    }

    /// Zero-pads to the next whole 3200-sample frame.
    private func paddedToWholeFrames(_ samples: MLXArray) -> MLXArray {
        let ratio = config.compressionRatio
        let remainder = samples.size % ratio
        guard remainder != 0 else { return samples }
        return concatenated(
            [samples, MLXArray.zeros([ratio - remainder], dtype: samples.dtype)], axis: 0)
    }

    /// Applies the -25 dBFS normalisation the non-streaming checkpoints were trained with.
    ///
    /// `tailor_dB_FS` scales so the RMS lands on target, then backs off if that would clip.
    func normalizedSamples(_ audio: MLXArray) -> MLXArray {
        var samples = audio.reshaped([-1]).asType(.float32)
        guard normalizeAudio else { return samples }

        let eps: Float = 1e-6
        let rms = MLX.sqrt(MLX.mean(samples * samples)).item(Float.self)
        samples = samples * (pow(10, -25.0 / 20.0) / (rms + eps))
        let peak = MLX.max(MLX.abs(samples)).item(Float.self)
        if peak > 1 { samples = samples / (peak + eps) }
        return samples
    }

    /// Runs an encoder over the waveform, splitting very long audio.
    ///
    /// Upstream caps a single encoder pass at 60 s (`acoustic_tokenizer_chunk_size`) to keep
    /// the convolution stack inside 32-bit indexing, carrying convolution state across
    /// segments so the result matches an unsplit pass.
    private func encodeSegmented(
        _ samples: MLXArray,
        encoder: VibeVoiceTokenizerEncoder
    ) -> MLXArray {
        let total = samples.size
        let chunk = config.acousticTokenizerChunkSize

        if total <= chunk {
            return encoder(samples.reshaped([1, 1, total]))
        }

        encoder.resetState()
        var pieces: [MLXArray] = []
        var offset = 0
        while offset < total {
            let end = Swift.min(offset + chunk, total)
            let segment = samples[offset ..< end].reshaped([1, 1, end - offset])
            let encoded = encoder.step(segment)
            if encoded.dim(2) > 0 { pieces.append(encoded) }
            offset = end
        }
        encoder.resetState()
        return pieces.isEmpty
            ? MLXArray.zeros([1, encoder.outputChannels, 0])
            : concatenated(pieces, axis: 2)
    }

    /// Trims or edge-pads the frame axis so it matches what the prompt reserved.
    private func alignFrames(_ features: MLXArray, to frames: Int) -> MLXArray {
        let have = features.dim(1)
        if have == frames { return features }
        if have > frames { return features[0..., 0 ..< frames, 0...] }
        let padding = MLXArray.zeros(
            [features.dim(0), frames - have, features.dim(2)], dtype: features.dtype)
        return concatenated([features, padding], axis: 1)
    }

    // MARK: - Generation

    public func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput {
        transcribe(audio: audio, generationParameters: generationParameters, onToken: { _ in })
    }

    /// Shared transcription core. `onToken` sees each decoded piece as it is produced.
    func transcribe(
        audio: MLXArray,
        generationParameters: STTGenerateParameters,
        onToken: (String) -> Void,
        onTokenID: ((Int) -> Void)? = nil
    ) -> STTOutput {
        let start = Date()
        let features = encodeAudio(audio)
        let duration = Double(audio.size) / Double(config.samplingRate)

        let (embeds, promptLength) = buildPromptEmbeddings(features: features, duration: duration)

        var text = ""
        var generated = 0
        let cache = languageModel.makeCache()
        let eos = tokenizer.eosTokenId ?? 151_643
        var pieces: [Int] = []

        var hidden = prefill(embeds, cache: cache)
        var next = nextToken(hidden, parameters: generationParameters, generated: pieces)
        asyncEval(next)
        let prefillEnd = Date()

        // Pipelined decode, as mlx-lm does it: the forward for token n+1 is queued before
        // token n is read back, so the GPU works while the host decodes text and checks for
        // termination. Reading `next` is the only synchronisation point per step.
        while generated < generationParameters.maxTokens {
            let token = next.item(Int.self)
            if token == eos { break }
            pieces.append(token)
            generated += 1
            onToken(tokenizer.decode(tokens: [token]))
            onTokenID?(token)

            if Self.isDegenerate(pieces, parameters: generationParameters) { break }
            if generated == generationParameters.maxTokens { break }

            hidden = languageModel(
                inputsEmbeds: languageModel.embed(next.reshaped([1, 1])), cache: cache)
            next = nextToken(hidden, parameters: generationParameters, generated: pieces)
            asyncEval(next)

            if generated % 256 == 0 { Memory.clearCache() }
        }

        text = tokenizer.decode(tokens: pieces)
        let segments = VibeVoiceASRPrompt.parse(text)
        let elapsed = Date().timeIntervalSince(start)
        let prefill = prefillEnd.timeIntervalSince(start)

        return STTOutput(
            text: segments.isEmpty ? text : VibeVoiceASRPrompt.plainText(segments),
            segments: segments.map { segment in
                var dict: [String: Any] = [:]
                if let text = segment.text { dict["text"] = text }
                if let speaker = segment.speakerID { dict["speaker_id"] = speaker }
                // `start`/`end` is what the rest of this package's tooling reads (and what
                // the SRT/VTT writers need); `start_time`/`end_time` mirror what upstream's
                // own post-processing returns. Both are emitted so neither caller is
                // surprised. Plain-format transcripts carry no timings, so the keys are
                // omitted rather than faked as zero.
                if let start = segment.startTime {
                    dict["start"] = start
                    dict["start_time"] = start
                }
                if let end = segment.endTime {
                    dict["end"] = end
                    dict["end_time"] = end
                }
                return dict
            },
            language: generationParameters.language,
            promptTokens: promptLength,
            generationTokens: generated,
            totalTokens: promptLength + generated,
            promptTps: prefill > 0 ? Double(promptLength) / prefill : 0,
            generationTps: elapsed > prefill ? Double(generated) / (elapsed - prefill) : 0,
            totalTime: elapsed,
            peakMemoryUsage: Double(GPU.peakMemory) / 1024 / 1024 / 1024)
    }

    /// Embeds the ChatML prompt with the audio features in place of the audio span.
    ///
    /// The audio token ids are inserted directly rather than written into the prompt text
    /// and re-tokenized: whether a tokenizer recognises `<|box_start|>` inside a string
    /// depends on how its special tokens are configured, and a silent miss here would
    /// misalign every frame. MLX also has no in-place masked assignment, so the sequence is
    /// assembled by concatenation — which is cheap, since the audio span is contiguous.
    func buildPromptEmbeddings(features: MLXArray, duration: Double) -> (MLXArray, Int) {
        let frames = features.dim(1)
        let (head, tail) = VibeVoiceASRPrompt.parts(audioDuration: duration)

        let headIDs = tokenizer.encode(text: head, addSpecialTokens: false)
        let tailIDs = tokenizer.encode(text: tail, addSpecialTokens: false)

        let prefix = headIDs + [config.audioBosTokenID]
        let suffix = [config.audioEosTokenID] + tailIDs

        let embeds = concatenated(
            [
                languageModel.embed(MLXArray(prefix.map { Int32($0) }, [1, prefix.count])),
                features.asType(lmHead.weight.dtype),
                languageModel.embed(MLXArray(suffix.map { Int32($0) }, [1, suffix.count])),
            ], axis: 1)

        return (embeds, prefix.count + frames + suffix.count)
    }

    /// The next token as a lazy `[1]` int32 array, so the caller can queue its evaluation.
    func nextToken(
        _ hidden: MLXArray,
        parameters: STTGenerateParameters,
        generated: [Int]
    ) -> MLXArray {
        var logits = lmHead(hidden[0..., -1, 0...])
        if parameters.temperature > 0 { logits = logits / parameters.temperature }
        logits = Self.applyRepetitionPenalty(
            logits, generated: generated,
            penalty: parameters.repetitionPenalty,
            contextSize: parameters.repetitionContextSize)

        if parameters.temperature <= 0 {
            return MLX.argMax(logits, axis: -1).asType(.int32)
        }
        return MLXRandom.categorical(logits).asType(.int32)
    }

    /// Sign-aware repetition penalty, matching the mlx-lm convention already used by
    /// Qwen3-ASR in this package: recently emitted tokens are pushed toward zero, so a
    /// greedy decode can climb out of a loop.
    static func applyRepetitionPenalty(
        _ logits: MLXArray,
        generated: [Int],
        penalty: Float,
        contextSize: Int
    ) -> MLXArray {
        guard penalty != 1.0, !generated.isEmpty else { return logits }
        let recent = MLXArray(Array(generated.suffix(Swift.max(1, contextSize))).map { Int32($0) })
        let selected = logits[0..., recent]
        let scale = MLXArray(penalty)
        let out = logits
        out[0..., recent] = MLX.where(selected .> 0, selected / scale, selected * scale)
        return out
    }

    /// Backstop for greedy callers, which have no penalty to escape a loop with.
    ///
    /// Long or difficult audio can collapse into repeating one phrase forever — a two-hour
    /// lecture produced "I mean, I mean, I mean…" until it hit the token cap. Left alone
    /// that grows the KV cache for nothing and stalls for minutes, so a run of almost no
    /// distinct tokens ends generation. Skipped when the caller supplies a penalty, since
    /// that is the real fix and genuinely repetitive speech should not be truncated.
    static func isDegenerate(_ generated: [Int], parameters: STTGenerateParameters) -> Bool {
        guard parameters.repetitionPenalty == 1.0, generated.count >= 24 else { return false }
        return Set(generated.suffix(24)).count <= 3
    }

    // MARK: - Loading

    /// Precision to run the language model at.
    ///
    /// The checkpoint is bfloat16 and transcribes correctly at that precision, long
    /// recordings included, now that prompts are prefilled in slices (see
    /// `prefillStepSize`). float32 remains available for comparisons against the reference
    /// implementation; it doubles the weights to roughly 33 GB.
    public enum Precision: String, Sendable {
        /// The checkpoint's own precision. Fast and compact.
        case bfloat16
        /// The reference implementation's float32 path, at twice the memory.
        case float32
    }

    public static func fromPretrained(
        _ modelRepo: String,
        precision: Precision = .bfloat16,
        cache: HubCache = .default
    ) async throws -> VibeVoiceASRModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw VibeVoiceASRError.invalidRepo(modelRepo)
        }
        let dir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: [
                "*.json", "tokenizer*", "vocab*", "merges*", "special_tokens*", "*.jinja",
            ],
            cache: cache)
        return try await fromModelDirectory(dir, precision: precision)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        precision: Precision = .bfloat16
    ) async throws -> VibeVoiceASRModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw VibeVoiceASRError.missingConfig(modelDir)
        }
        let config = try VibeVoiceASRConfiguration(from: Data(contentsOf: configURL))

        // Only the non-streaming checkpoints normalise; the streaming one says so explicitly.
        var normalize = true
        let preprocessorURL = modelDir.appendingPathComponent("preprocessor_config.json")
        var chunkFrames: Int?
        var lookaheadFrames: Int?
        if let data = try? Data(contentsOf: preprocessorURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            normalize = json["normalize_audio"] as? Bool ?? normalize
            chunkFrames = json["chunk_frames"] as? Int
            lookaheadFrames = json["lookahead_frames"] as? Int
        }

        let tokenizer: Tokenizers.Tokenizer
        do {
            tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            throw VibeVoiceASRError.missingTokenizer(modelDir)
        }

        let model = VibeVoiceASRModel(
            config: config, tokenizer: tokenizer, normalizeAudio: normalize)
        if let chunkFrames { model.streamingChunkFrames = chunkFrames }
        if let lookaheadFrames { model.streamingLookaheadFrames = lookaheadFrames }

        let shards = try FileManager.default
            .contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shards.isEmpty else { throw VibeVoiceASRError.missingWeights(modelDir) }

        var weights: [String: MLXArray] = [:]
        for shard in shards {
            for (key, value) in try MLX.loadArrays(url: shard) { weights[key] = value }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(
                sanitize(weights: weights, layout: config.layout)),
            verify: .all)

        // Keep the audio encoders in float32 even though the checkpoint is bfloat16.
        //
        // They are ~33 convolutions deep, and bf16 accumulates through them badly: measured
        // against the reference on a 3 minute clip, the latents come out at 40.7 dB SNR in
        // bf16 against 68.2 dB in float32. That degradation is enough to send greedy
        // decoding into a repetition loop on long audio. The two encoders are a small part
        // of an 8B model, so this costs about a gigabyte and leaves the language model at
        // its native precision.
        model.acousticEncoder.update(
            parameters: model.acousticEncoder.parameters().mapValues { $0.asType(.float32) })
        model.semanticEncoder.update(
            parameters: model.semanticEncoder.parameters().mapValues { $0.asType(.float32) })

        if precision == .float32 {
            model.update(parameters: model.parameters().mapValues { $0.asType(.float32) })
        }

        model.train(false)
        eval(model)
        return model
    }
}

// MARK: - STTGenerationModel

extension VibeVoiceASRModel: STTGenerationModel {

    /// Streams the transcript as it is decoded.
    ///
    /// This is token streaming over a complete recording. Transcribing audio that is still
    /// arriving is a different mechanism with its own checkpoint — see
    /// `VibeVoiceASRStreamSession`.
    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Swift.Error> {
        let (stream, continuation) = AsyncThrowingStream<STTGeneration, Swift.Error>.makeStream()

        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish()
                return
            }
            let start = Date()
            let output = self.transcribe(
                audio: audio, generationParameters: generationParameters,
                onToken: { continuation.yield(.token($0)) })
            let elapsed = Date().timeIntervalSince(start)
            continuation.yield(.info(STTGenerationInfo(
                promptTokenCount: output.promptTokens,
                generationTokenCount: output.generationTokens,
                prefillTime: 0,
                generateTime: elapsed,
                tokensPerSecond: output.generationTps,
                peakMemoryUsage: output.peakMemoryUsage)))
            continuation.yield(.result(output))
            continuation.finish()
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }
}
