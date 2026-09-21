import MLX

public struct STTGenerateParameters: Sendable {
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let verbose: Bool
    public let language: String?
    public let chunkDuration: Float
    public let minChunkDuration: Float
    public let repetitionPenalty: Float
    public let repetitionContextSize: Int
    /// KV-cache quantization bits; `nil` keeps model precision.
    public let kvBits: Int?
    /// Group size for KV-cache quantization.
    public let kvGroupSize: Int
    /// Cache offset that must be exceeded before the KV cache is quantized.
    public let quantizedKVStart: Int
    /// Free-text hint biasing recognition toward names, jargon or topics the audio is known
    /// to contain. Only models whose prompt has somewhere to put it read this; VibeVoice ASR
    /// calls it `context_info` and appends it to the transcription request.
    public let contextInfo: String?

    public init(
        maxTokens: Int = 8192,
        temperature: Float = 0.0,
        topP: Float = 0.95,
        topK: Int = 0,
        verbose: Bool = false,
        language: String? = nil,
        chunkDuration: Float = 1200.0,
        minChunkDuration: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        repetitionContextSize: Int = 32,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        contextInfo: String? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.verbose = verbose
        self.language = language
        self.chunkDuration = chunkDuration
        self.minChunkDuration = minChunkDuration
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.contextInfo = contextInfo
    }
}

public protocol STTGenerationModel: AnyObject {
    var defaultGenerationParameters: STTGenerateParameters { get }

    /// Sample rate, in Hz, that `generate(audio:)` expects its input at.
    var sampleRate: Int { get }

    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error>
}

public extension STTGenerationModel {
    /// Nearly every speech recognizer here is 16 kHz; VibeVoice ASR, at 24 kHz, is the
    /// exception, so the requirement is defaulted rather than forced on every model.
    var sampleRate: Int { 16000 }

    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters? = nil
    ) -> STTOutput {
        generate(audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters)
    }

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters? = nil
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters)
    }
}
