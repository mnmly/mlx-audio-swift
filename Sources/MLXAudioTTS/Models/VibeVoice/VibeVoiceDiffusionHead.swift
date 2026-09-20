import Foundation
import MLX
import MLXNN

/// Configuration for the per-frame diffusion head (`diffusion_head_config`).
public struct VibeVoiceDiffusionHeadConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var headLayers: Int
    public var headFFNRatio: Float
    public var rmsNormEps: Float
    public var latentSize: Int
    public var predictionType: String
    public var ddpmNumSteps: Int
    public var ddpmNumInferenceSteps: Int
    public var ddpmBetaSchedule: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case headLayers = "head_layers"
        case headFFNRatio = "head_ffn_ratio"
        case rmsNormEps = "rms_norm_eps"
        case latentSize = "latent_size"
        case predictionType = "prediction_type"
        case ddpmNumSteps = "ddpm_num_steps"
        case ddpmNumInferenceSteps = "ddpm_num_inference_steps"
        case ddpmBetaSchedule = "ddpm_beta_schedule"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        headLayers = try c.decodeIfPresent(Int.self, forKey: .headLayers) ?? 4
        headFFNRatio = try c.decodeIfPresent(Float.self, forKey: .headFFNRatio) ?? 3.0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        latentSize = try c.decodeIfPresent(Int.self, forKey: .latentSize) ?? 64
        predictionType = try c.decodeIfPresent(String.self, forKey: .predictionType)
            ?? "v_prediction"
        ddpmNumSteps = try c.decodeIfPresent(Int.self, forKey: .ddpmNumSteps) ?? 1000
        ddpmNumInferenceSteps = try c.decodeIfPresent(Int.self, forKey: .ddpmNumInferenceSteps)
            ?? 20
        ddpmBetaSchedule = try c.decodeIfPresent(String.self, forKey: .ddpmBetaSchedule)
            ?? "cosine"
    }
}

/// Sinusoidal timestep features followed by a two-layer MLP.
///
/// Upstream concatenates **cosine before sine**, which is the opposite of the diffusers
/// convention; getting this backwards silently produces noise.
final class VibeVoiceTimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [Linear]

    private let frequencyEmbeddingSize: Int

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        // Index 1 of the upstream Sequential is the SiLU, so the checkpoint numbers the two
        // Linears 0 and 2. A three-element array with an unused middle entry would add a
        // phantom parameter, so the gap is handled in `sanitize` instead.
        _mlp.wrappedValue = [
            Linear(frequencyEmbeddingSize, hiddenSize, bias: false),
            Linear(hiddenSize, hiddenSize, bias: false),
        ]
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = frequencyEmbeddingSize / 2
        let exponents = MLXArray(0 ..< half).asType(.float32) / Float(half)
        let freqs = MLX.exp(-log(10000.0) * exponents)
        let args = t.reshaped([-1, 1]).asType(.float32) * freqs.reshaped([1, -1])
        let embedding = concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        return mlp[1](silu(mlp[0](embedding)))
    }
}

/// SwiGLU feed-forward, as used inside the diffusion head (distinct from the tokenizer's
/// plain GELU FFN).
final class VibeVoiceDiffusionFFN: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(embedDim: Int, ffnDim: Int) {
        _gateProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        _upProj.wrappedValue = Linear(embedDim, ffnDim, bias: false)
        _downProj.wrappedValue = Linear(ffnDim, embedDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

@inline(__always)
private func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

/// One adaLN-modulated residual FFN block.
final class VibeVoiceHeadLayer: Module {
    @ModuleInfo(key: "ffn") var ffn: VibeVoiceDiffusionFFN
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Linear]

    private let embedDim: Int

    init(embedDim: Int, ffnDim: Int, condDim: Int, normEps: Float) {
        self.embedDim = embedDim
        _ffn.wrappedValue = VibeVoiceDiffusionFFN(embedDim: embedDim, ffnDim: ffnDim)
        _norm.wrappedValue = RMSNorm(dimensions: embedDim, eps: normEps)
        _adaLNModulation.wrappedValue = [Linear(condDim, 3 * embedDim, bias: false)]
    }

    func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let modulation = adaLNModulation[0](silu(c))
        let parts = split(modulation, parts: 3, axis: -1)
        return x + parts[2] * ffn(modulate(norm(x), shift: parts[0], scale: parts[1]))
    }
}

/// Final adaLN block, projecting back to the latent dimension.
///
/// `norm_final` is built with `elementwise_affine=False` upstream, so it carries no weight
/// and contributes no checkpoint key — the scale comes entirely from the adaLN modulation.
/// It is therefore normalised inline rather than with `MLXNN.RMSNorm`, which always owns a
/// weight parameter.
final class VibeVoiceFinalLayer: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var adaLNModulation: [Linear]

    private let normEps: Float

    init(hiddenSize: Int, outputSize: Int, condSize: Int, normEps: Float) {
        self.normEps = normEps
        _linear.wrappedValue = Linear(hiddenSize, outputSize, bias: false)
        _adaLNModulation.wrappedValue = [Linear(condSize, 2 * hiddenSize, bias: false)]
    }

    func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let normed = x * MLX.rsqrt(mean(x * x, axis: -1, keepDims: true) + normEps)
        let modulation = adaLNModulation[0](silu(c))
        let parts = split(modulation, parts: 2, axis: -1)
        return linear(modulate(normed, shift: parts[0], scale: parts[1]))
    }
}

/// Predicts the diffusion target for a single 7.5 Hz acoustic latent, conditioned on the
/// TTS backbone's last hidden state.
public final class VibeVoiceDiffusionHead: Module {
    @ModuleInfo(key: "noisy_images_proj") var noisyImagesProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "t_embedder") var tEmbedder: VibeVoiceTimestepEmbedder
    @ModuleInfo(key: "layers") var layers: [VibeVoiceHeadLayer]
    @ModuleInfo(key: "final_layer") var finalLayer: VibeVoiceFinalLayer

    public let config: VibeVoiceDiffusionHeadConfiguration

    public init(_ config: VibeVoiceDiffusionHeadConfiguration) {
        self.config = config
        let ffnDim = Int(Float(config.hiddenSize) * config.headFFNRatio)

        _noisyImagesProj.wrappedValue = Linear(config.latentSize, config.hiddenSize, bias: false)
        _condProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _tEmbedder.wrappedValue = VibeVoiceTimestepEmbedder(hiddenSize: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.headLayers).map { _ in
            VibeVoiceHeadLayer(
                embedDim: config.hiddenSize, ffnDim: ffnDim,
                condDim: config.hiddenSize, normEps: config.rmsNormEps)
        }
        _finalLayer.wrappedValue = VibeVoiceFinalLayer(
            hiddenSize: config.hiddenSize, outputSize: config.latentSize,
            condSize: config.hiddenSize, normEps: config.rmsNormEps)
    }

    public func callAsFunction(
        noisyImages: MLXArray,
        timesteps: MLXArray,
        condition: MLXArray
    ) -> MLXArray {
        var x = noisyImagesProj(noisyImages)
        let c = condProj(condition) + tEmbedder(timesteps)
        for layer in layers { x = layer(x, c) }
        return finalLayer(x, c)
    }

    /// Collapses upstream's `nn.Sequential` indices: `adaLN_modulation.1` is the only Linear
    /// in that Sequential (index 0 is the activation), and `t_embedder.mlp` numbers its two
    /// Linears 0 and 2.
    public static func sanitize(weights: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (rawKey, value) in weights where rawKey.hasPrefix(prefix) {
            var key = String(rawKey.dropFirst(prefix.count))
            key = key.replacingOccurrences(
                of: "adaLN_modulation.1.", with: "adaLN_modulation.0.")
            key = key.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.mlp.1.")
            out[key] = value
        }
        return out
    }
}
