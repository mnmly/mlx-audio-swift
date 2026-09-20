import Foundation
@preconcurrency import MLX
import MLXAudioCodecs
@preconcurrency import MLXLMCommon
import MLXNN

/// Transcribes audio that is still arriving.
///
/// This is not a chunked audio *encoder*: the encoder runs fresh on each overlapping window,
/// and what persists between chunks is the language model's KV cache. Each chunk's features
/// are wrapped in the audio start/end embeddings, decoded until the model emits
/// `<|text_chunk_end|>`, and then that token's embedding is appended to the cache whether or
/// not the model produced it — that forced separator is what keeps the chunks aligned.
///
/// Needs a checkpoint trained for it (`microsoft/VibeVoice-ASR-Streaming-7B`), whose
/// `preprocessor_config.json` carries the chunk and lookahead sizes it was trained at.
public final class VibeVoiceASRStreamSession {
    public struct Chunk: Sendable {
        public let text: String
        /// Start of the audio this chunk covers, in seconds.
        public let startTime: Double
    }

    private let model: VibeVoiceASRModel
    private var cache: [KVCache]
    private let audioStartEmbed: MLXArray
    private let audioEndEmbed: MLXArray
    private let textChunkEndID: Int
    private let eosID: Int

    /// Samples the window advances by each step.
    public let chunkSamples: Int
    /// Extra samples of future context encoded with each window but not advanced over.
    public let lookaheadSamples: Int

    private var buffer: [Float] = []
    private var consumedSamples = 0
    private var finished = false

    public let maxTokensPerChunk: Int
    public var temperature: Float = 0

    public init(
        model: VibeVoiceASRModel,
        contextInfo: String? = nil,
        chunkFrames: Int? = nil,
        lookaheadFrames: Int? = nil,
        maxTokensPerChunk: Int = 256
    ) {
        self.model = model
        self.maxTokensPerChunk = maxTokensPerChunk

        let ratio = model.config.compressionRatio
        chunkSamples = (chunkFrames ?? model.streamingChunkFrames) * ratio
        lookaheadSamples = (lookaheadFrames ?? model.streamingLookaheadFrames) * ratio

        textChunkEndID = model.tokenizer.convertTokenToId("<|text_chunk_end|>") ?? 151_665
        eosID = model.tokenizer.eosTokenId ?? 151_643

        audioStartEmbed = model.languageModel.embed(
            MLXArray([Int32(model.config.audioBosTokenID)], [1, 1]))
        audioEndEmbed = model.languageModel.embed(
            MLXArray([Int32(model.config.audioEosTokenID)], [1, 1]))

        // Prefill the instruction. The streaming prompt is raw text, not ChatML.
        cache = model.languageModel.makeCache()
        let prompt = VibeVoiceASRPrompt.buildStreaming(contextInfo: contextInfo)
        let ids = model.tokenizer.encode(text: prompt, addSpecialTokens: false)
        let embeds = model.languageModel.embed(
            MLXArray(ids.map { Int32($0) }, [1, ids.count]))
        _ = model.languageModel(inputsEmbeds: embeds, cache: cache)
    }

    /// Feeds newly captured audio, returning a chunk for every complete window it unlocks.
    public func append(_ samples: MLXArray) -> [Chunk] {
        buffer.append(contentsOf: samples.reshaped([-1]).asType(.float32).asArray(Float.self))
        var chunks: [Chunk] = []
        // A window needs its lookahead before it can be decoded.
        while buffer.count >= chunkSamples + lookaheadSamples {
            chunks.append(decodeWindow(padding: false))
        }
        return chunks
    }

    /// Flushes whatever is left, zero-padding the final window as upstream does.
    public func finish() -> [Chunk] {
        var chunks: [Chunk] = []
        while !buffer.isEmpty, !finished {
            chunks.append(decodeWindow(padding: true))
        }
        finished = true
        return chunks
    }

    private func decodeWindow(padding: Bool) -> Chunk {
        let target = chunkSamples + lookaheadSamples
        let available = Swift.min(target, buffer.count)

        var window = Array(buffer[0 ..< available])
        if padding, window.count < target {
            window.append(contentsOf: [Float](repeating: 0, count: target - window.count))
        }

        let startTime = Double(consumedSamples) / Double(model.config.samplingRate)
        let advance = Swift.min(chunkSamples, buffer.count)
        buffer.removeFirst(advance)
        consumedSamples += advance

        let features = model.encodeAudio(MLXArray(window))
        let text = decode(features: features)
        return Chunk(text: text, startTime: startTime)
    }

    /// One chunk: push the audio, decode until the separator, then force the separator in.
    private func decode(features: MLXArray) -> String {
        let dtype = model.lmHead.weight.dtype
        let audioEmbeds = concatenated(
            [audioStartEmbed, features.asType(dtype), audioEndEmbed], axis: 1)

        var hidden = model.languageModel(inputsEmbeds: audioEmbeds, cache: cache)
        var tokens: [Int] = []

        for _ in 0 ..< maxTokensPerChunk {
            let logits = model.lmHead(hidden[0..., -1, 0...])
            let token: Int
            if temperature <= 0 {
                token = MLX.argMax(logits, axis: -1).item(Int.self)
            } else {
                token = MLXRandom.categorical(logits / temperature).item(Int.self)
            }

            if token == textChunkEndID || token == eosID { break }
            tokens.append(token)

            let embed = model.languageModel.embed(MLXArray([Int32(token)], [1, 1]))
            hidden = model.languageModel(inputsEmbeds: embed, cache: cache)
        }

        // The separator goes into the cache even when the model stopped for another reason,
        // so the next chunk always starts from the same kind of boundary.
        let separator = model.languageModel.embed(
            MLXArray([Int32(textChunkEndID)], [1, 1]))
        _ = model.languageModel(inputsEmbeds: separator, cache: cache)

        var text = model.tokenizer.decode(tokens: tokens)
        for marker in [
            "<|text_chunk_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>",
        ] {
            text = text.replacingOccurrences(of: marker, with: "")
        }
        return text
    }
}
