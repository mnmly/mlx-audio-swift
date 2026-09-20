import Foundation
import MLX
import MLXNN

/// Projects acoustic (or semantic) VAE latents into the language model's hidden space.
///
/// Shared by the realtime TTS model and by ASR, which uses one instance per tokenizer.
public final class VibeVoiceSpeechConnector: Module {
    @ModuleInfo(key: "fc1") public var fc1: Linear
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    @ModuleInfo(key: "fc2") public var fc2: Linear

    public init(inputDim: Int, outputDim: Int) {
        _fc1.wrappedValue = Linear(inputDim, outputDim, bias: true)
        // Upstream uses LlamaRMSNorm with a hard-coded 1e-6, not the model's rms_norm_eps.
        _norm.wrappedValue = RMSNorm(dimensions: outputDim, eps: 1e-6)
        _fc2.wrappedValue = Linear(outputDim, outputDim, bias: true)
    }

    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        fc2(norm(fc1(features)))
    }
}
