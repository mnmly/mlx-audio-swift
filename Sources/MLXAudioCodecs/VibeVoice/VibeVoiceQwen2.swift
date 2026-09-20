import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

/// Qwen2 decoder configuration as VibeVoice stores it under `decoder_config`.
public struct VibeVoiceQwen2Configuration: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_936
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 8192
    }

    public init(
        hiddenSize: Int, numHiddenLayers: Int, intermediateSize: Int,
        numAttentionHeads: Int, numKeyValueHeads: Int, rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000, vocabSize: Int = 151_936,
        maxPositionEmbeddings: Int = 8192
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    /// A copy with a different layer count, for splitting one checkpoint across two stacks.
    public func withLayerCount(_ count: Int) -> VibeVoiceQwen2Configuration {
        var copy = self
        copy.numHiddenLayers = count
        return copy
    }
}

/// Grouped-query attention with Qwen2's q/k/v biases and no q/k norm.
public final class VibeVoiceQwen2Attention: Module {
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    private let numHeads: Int
    private let numKvHeads: Int
    private let headDim: Int
    private let scale: Float
    private let rope: RoPE

    public init(_ config: VibeVoiceQwen2Configuration) {
        numHeads = config.numAttentionHeads
        numKvHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKvHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return oProj(output)
    }
}

public final class VibeVoiceQwen2MLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(_ config: VibeVoiceQwen2Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public final class VibeVoiceQwen2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: VibeVoiceQwen2Attention
    @ModuleInfo(key: "mlp") public var mlp: VibeVoiceQwen2MLP
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: RMSNorm

    public init(_ config: VibeVoiceQwen2Configuration) {
        _selfAttn.wrappedValue = VibeVoiceQwen2Attention(config)
        _mlp.wrappedValue = VibeVoiceQwen2MLP(config)
        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

/// A Qwen2 decoder stack that accepts embeddings directly and carries an externally owned
/// KV cache.
///
/// `MLXLLM.Qwen2Model` cannot be used here: it takes token ids only, and its `layers` are
/// `fileprivate`, so VibeVoice's split backbone (4 text-only layers, 20 text+speech layers)
/// and its hidden-state splice are both out of reach. Both submodules are optional because
/// the checkpoint's two stacks differ — the lower stack's final norm is replaced by an
/// identity upstream, and the upper stack's embedding table is never read.
public final class VibeVoiceQwen2Model: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding?
    @ModuleInfo(key: "layers") public var layers: [VibeVoiceQwen2DecoderLayer]
    @ModuleInfo(key: "norm") public var norm: RMSNorm?

    public let config: VibeVoiceQwen2Configuration

    public init(
        _ config: VibeVoiceQwen2Configuration,
        includeEmbedding: Bool = true,
        includeFinalNorm: Bool = true
    ) {
        self.config = config
        _embedTokens.wrappedValue = includeEmbedding
            ? Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
            : nil
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            VibeVoiceQwen2DecoderLayer(config)
        }
        _norm.wrappedValue = includeFinalNorm
            ? RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            : nil
    }

    public func embed(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokens else {
            fatalError("This stack was built without an embedding table")
        }
        return embedTokens(inputIds)
    }

    public func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else if let inputIds {
            h = embed(inputIds)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let mask = createAttentionMask(h: h, cache: cache?.first)
        let caches = cache ?? [KVCache?](repeating: nil, count: layers.count)
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches[index])
        }
        return norm?(h) ?? h
    }

    public func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }
}
