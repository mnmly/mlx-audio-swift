import Foundation

/// Configuration for VibeVoice's causal convolutional σ-VAE speech tokenizers.
///
/// The same layout backs both the acoustic tokenizer (`vae_dim` 64, used by the realtime
/// TTS model and by ASR) and the semantic tokenizer (`vae_dim` 128, encoder only, ASR only).
/// Ratios multiply out to the sample-per-token compression: `[8, 5, 5, 4, 2, 2]` is 3200,
/// i.e. 7.5 tokens per second at 24 kHz.
public struct VibeVoiceTokenizerConfiguration: Codable, Sendable {
    public var channels: Int
    public var causal: Bool
    public var vaeDim: Int
    public var fixStd: Float
    public var stdDistType: String

    public var mixerLayer: String
    public var convNorm: String
    public var padMode: String
    public var disableLastNorm: Bool
    public var layernorm: String
    public var layernormEps: Float
    public var layernormElementwiseAffine: Bool
    public var convBias: Bool
    public var layerScaleInitValue: Float

    public var encoderNFilters: Int
    public var encoderRatios: [Int]
    public var encoderDepths: String

    public var decoderNFilters: Int
    public var decoderRatios: [Int]?
    public var decoderDepths: String?

    /// Kernel size for the stem, head, and every `Block1D` mixer. Not present in shipped
    /// configs; upstream defaults it to 7 via `getattr`.
    public var kernelSize: Int
    public var lastKernelSize: Int
    public var ffnExpansion: Int

    enum CodingKeys: String, CodingKey {
        case channels
        case causal
        case vaeDim = "vae_dim"
        case fixStd = "fix_std"
        case stdDistType = "std_dist_type"
        case mixerLayer = "mixer_layer"
        case convNorm = "conv_norm"
        case padMode = "pad_mode"
        case disableLastNorm = "disable_last_norm"
        case layernorm
        case layernormEps = "layernorm_eps"
        case layernormElementwiseAffine = "layernorm_elementwise_affine"
        case convBias = "conv_bias"
        case layerScaleInitValue = "layer_scale_init_value"
        case encoderNFilters = "encoder_n_filters"
        case encoderRatios = "encoder_ratios"
        case encoderDepths = "encoder_depths"
        case decoderNFilters = "decoder_n_filters"
        case decoderRatios = "decoder_ratios"
        case decoderDepths = "decoder_depths"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case ffnExpansion = "ffn_expansion"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        vaeDim = try c.decodeIfPresent(Int.self, forKey: .vaeDim) ?? 64
        fixStd = try c.decodeIfPresent(Float.self, forKey: .fixStd) ?? 0.5
        stdDistType = try c.decodeIfPresent(String.self, forKey: .stdDistType) ?? "fix"
        mixerLayer = try c.decodeIfPresent(String.self, forKey: .mixerLayer) ?? "depthwise_conv"
        convNorm = try c.decodeIfPresent(String.self, forKey: .convNorm) ?? "none"
        padMode = try c.decodeIfPresent(String.self, forKey: .padMode) ?? "constant"
        disableLastNorm = try c.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? true
        layernorm = try c.decodeIfPresent(String.self, forKey: .layernorm) ?? "RMSNorm"
        layernormEps = try c.decodeIfPresent(Float.self, forKey: .layernormEps) ?? 1e-5
        layernormElementwiseAffine =
            try c.decodeIfPresent(Bool.self, forKey: .layernormElementwiseAffine) ?? true
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? true
        layerScaleInitValue = try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
        encoderNFilters = try c.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? 32
        encoderRatios = try c.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? [8, 5, 5, 4, 2, 2]
        encoderDepths = try c.decodeIfPresent(String.self, forKey: .encoderDepths) ?? "3-3-3-3-3-3-8"
        decoderNFilters = try c.decodeIfPresent(Int.self, forKey: .decoderNFilters) ?? 32
        decoderRatios = try c.decodeIfPresent([Int].self, forKey: .decoderRatios)
        decoderDepths = try c.decodeIfPresent(String.self, forKey: .decoderDepths)
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        lastKernelSize = try c.decodeIfPresent(Int.self, forKey: .lastKernelSize) ?? 7
        ffnExpansion = try c.decodeIfPresent(Int.self, forKey: .ffnExpansion) ?? 4
    }

    /// Encoder depths, parsed from the `"3-3-3-3-3-3-8"` form.
    public var parsedEncoderDepths: [Int] {
        encoderDepths.split(separator: "-").compactMap { Int($0) }
    }

    /// Decoder depths. Upstream reverses the encoder depths when `decoder_depths` is null.
    public var parsedDecoderDepths: [Int] {
        if let decoderDepths {
            return decoderDepths.split(separator: "-").compactMap { Int($0) }
        }
        return parsedEncoderDepths.reversed()
    }

    /// Decoder ratios default to the encoder's.
    public var resolvedDecoderRatios: [Int] { decoderRatios ?? encoderRatios }

    /// Samples consumed per latent frame — 3200 for the shipped checkpoints.
    public var compressionRatio: Int { encoderRatios.reduce(1, *) }
}
