import Foundation
import MLX
import MLXFast
import MLXNN

/// RMSNorm over the channel axis of an NCL tensor.
///
/// Upstream (`ConvRMSNorm`) transposes to channel-last, normalizes, and transposes back; it
/// also force-casts to float32 for the reduction, which the fused kernel already does. The
/// weight is held directly rather than by wrapping `MLXNN.RMSNorm`, so checkpoint keys stay
/// `norm.weight` instead of gaining a nested `norm.norm.weight`.
public final class VibeVoiceConvRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    private let eps: Float

    public init(dimensions: Int, eps: Float) {
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]), key: "weight")
        self.eps = eps
    }

    public func callAsFunction(_ xNCL: MLXArray) -> MLXArray {
        let channelLast = swappedAxes(xNCL, 1, 2)
        let normed = MLXFast.rmsNorm(channelLast, weight: weight, eps: eps)
        return swappedAxes(normed, 1, 2)
    }
}

/// Position-wise feed-forward used inside `Block1D`, applied channel-last.
public final class VibeVoiceFFN: Module {
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear

    public init(embedDim: Int, ffnDim: Int, bias: Bool) {
        self._linear1 = ModuleInfo(
            wrappedValue: Linear(embedDim, ffnDim, bias: bias), key: "linear1")
        self._linear2 = ModuleInfo(
            wrappedValue: Linear(ffnDim, embedDim, bias: bias), key: "linear2")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Upstream uses ACT2FN["gelu"], which is the exact erf form rather than the tanh
        // approximation.
        linear2(gelu(linear1(x)))
    }
}

/// Thin wrapper mirroring upstream's `Convlayer`, which exists only so that checkpoint keys
/// nest as `mixer.conv.conv.conv.weight`.
public final class VibeVoiceConvLayer: Module {
    @ModuleInfo(key: "conv") public var conv: StreamableConv1d

    public init(dim: Int, kernelSize: Int, groups: Int, bias: Bool, causal: Bool, padMode: PadMode) {
        self._conv = ModuleInfo(
            wrappedValue: StreamableConv1d(
                inChannels: dim, outChannels: dim, ksize: kernelSize, stride: 1,
                dilation: 1, groups: groups, bias: bias, causal: causal, padMode: padMode
            ),
            key: "conv")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
    public func step(_ x: MLXArray) -> MLXArray { conv.step(x) }
    public func resetState() { conv.resetState() }
}

/// ConvNeXt-style block: depthwise causal conv mixer then a channel-wise FFN, each with a
/// LayerScale gain and a residual.
public final class VibeVoiceBlock1D: Module {
    @ModuleInfo(key: "norm") public var norm: VibeVoiceConvRMSNorm
    @ModuleInfo(key: "ffn_norm") public var ffnNorm: VibeVoiceConvRMSNorm
    @ModuleInfo(key: "mixer") public var mixer: VibeVoiceConvLayer
    @ModuleInfo(key: "ffn") public var ffn: VibeVoiceFFN

    @ParameterInfo(key: "gamma") public var gamma: MLXArray
    @ParameterInfo(key: "ffn_gamma") public var ffnGamma: MLXArray

    public init(dim: Int, config: VibeVoiceTokenizerConfiguration, padMode: PadMode) {
        self._norm = ModuleInfo(
            wrappedValue: VibeVoiceConvRMSNorm(dimensions: dim, eps: config.layernormEps),
            key: "norm")
        self._ffnNorm = ModuleInfo(
            wrappedValue: VibeVoiceConvRMSNorm(dimensions: dim, eps: config.layernormEps),
            key: "ffn_norm")
        self._mixer = ModuleInfo(
            wrappedValue: VibeVoiceConvLayer(
                dim: dim,
                kernelSize: config.kernelSize,
                groups: config.mixerLayer == "depthwise_conv" ? dim : 1,
                bias: config.convBias,
                causal: config.causal,
                padMode: padMode
            ),
            key: "mixer")
        // Block1D forwards the shared `bias` flag to the FFN, so it follows conv_bias.
        self._ffn = ModuleInfo(
            wrappedValue: VibeVoiceFFN(
                embedDim: dim, ffnDim: config.ffnExpansion * dim, bias: config.convBias),
            key: "ffn")
        self._gamma = ParameterInfo(wrappedValue: MLXArray.ones([dim]), key: "gamma")
        self._ffnGamma = ParameterInfo(wrappedValue: MLXArray.ones([dim]), key: "ffn_gamma")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        applyFFN(x + mixer(norm(x)) * gamma.expandedDimensions(axis: -1))
    }

    public func step(_ x: MLXArray) -> MLXArray {
        // The mixer is stride 1, so a streaming step always emits as many frames as it
        // consumes and the residual lines up without extra bookkeeping.
        applyFFN(x + mixer.step(norm(x)) * gamma.expandedDimensions(axis: -1))
    }

    private func applyFFN(_ x: MLXArray) -> MLXArray {
        let h = swappedAxes(ffn(swappedAxes(ffnNorm(x), 1, 2)), 1, 2)
        return x + h * ffnGamma.expandedDimensions(axis: -1)
    }

    public func resetState() { mixer.resetState() }
}
