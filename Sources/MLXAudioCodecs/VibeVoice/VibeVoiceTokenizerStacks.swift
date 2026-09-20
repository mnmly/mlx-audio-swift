import Foundation
import MLX
import MLXNN

/// Audio (NCL, 1 channel) to latent means (NCL, `vaeDim` channels).
///
/// Mirrors upstream's `TokenizerEncoder`: a stem conv, then for each of the seven stages a
/// downsampling conv followed by a run of `Block1D`s. Ratios are applied in reverse, so the
/// waveform is decimated 2, 2, 4, 5, 5, 8 while channels double 32 → 2048.
public final class VibeVoiceTokenizerEncoder: Module {
    @ModuleInfo(key: "stem") public var stem: StreamableConv1d
    @ModuleInfo(key: "downsample_layers") public var downsampleLayers: [StreamableConv1d]
    @ModuleInfo(key: "stages") public var stages: [[VibeVoiceBlock1D]]
    @ModuleInfo(key: "head") public var head: StreamableConv1d

    private let depths: [Int]

    public init(config: VibeVoiceTokenizerConfiguration, dimension: Int) {
        let padMode = config.mlxPadMode
        let ratios = Array(config.encoderRatios.reversed())
        let depths = config.parsedEncoderDepths
        self.depths = depths

        self._stem = ModuleInfo(wrappedValue: StreamableConv1d(
            inChannels: config.channels, outChannels: config.encoderNFilters,
            ksize: config.kernelSize, stride: 1, dilation: 1, groups: 1,
            bias: config.convBias, causal: config.causal, padMode: padMode
        ), key: "stem")

        self._downsampleLayers = ModuleInfo(wrappedValue: ratios.enumerated().map { index, ratio in
            StreamableConv1d(
                inChannels: config.encoderNFilters * (1 << index),
                outChannels: config.encoderNFilters * (1 << (index + 1)),
                ksize: ratio * 2, stride: ratio, dilation: 1, groups: 1,
                bias: config.convBias, causal: config.causal, padMode: padMode
            )
        }, key: "downsample_layers")

        self._stages = ModuleInfo(wrappedValue: depths.enumerated().map { index, depth in
            let dim = config.encoderNFilters * (1 << index)
            return (0 ..< depth).map { _ in
                VibeVoiceBlock1D(dim: dim, config: config, padMode: padMode)
            }
        }, key: "stages")

        let finalDim = config.encoderNFilters * (1 << (depths.count - 1))
        self._head = ModuleInfo(wrappedValue: StreamableConv1d(
            inChannels: finalDim, outChannels: dimension,
            ksize: config.lastKernelSize, stride: 1, dilation: 1, groups: 1,
            bias: config.convBias, causal: config.causal, padMode: padMode
        ), key: "head")
    }

    /// `disable_last_norm` is true for every shipped checkpoint, so the trailing norm is an
    /// identity and is deliberately not modelled.
    public func callAsFunction(_ xNCL: MLXArray) -> MLXArray {
        var x = xNCL
        for index in depths.indices {
            x = index == 0 ? stem(x) : downsampleLayers[index - 1](x)
            for block in stages[index] { x = block(x) }
        }
        return head(x)
    }

    public func step(_ xNCL: MLXArray) -> MLXArray {
        var x = xNCL
        for index in depths.indices {
            x = index == 0 ? stem.step(x) : downsampleLayers[index - 1].step(x)
            if x.shape[2] == 0 { return x }
            for block in stages[index] { x = block.step(x) }
        }
        return head.step(x)
    }

    public func resetState() {
        stem.resetState()
        for layer in downsampleLayers { layer.resetState() }
        for stage in stages { for block in stage { block.resetState() } }
        head.resetState()
    }
}

/// Latents (NCL, `vaeDim` channels) back to audio (NCL, 1 channel).
///
/// The mirror of the encoder: ratios are applied in their configured order, so the first
/// transposed conv upsamples by 8 while channels halve 2048 → 32.
public final class VibeVoiceTokenizerDecoder: Module {
    @ModuleInfo(key: "stem") public var stem: StreamableConv1d
    @ModuleInfo(key: "upsample_layers") public var upsampleLayers: [StreamableConvTranspose1d]
    @ModuleInfo(key: "stages") public var stages: [[VibeVoiceBlock1D]]
    @ModuleInfo(key: "head") public var head: StreamableConv1d

    private let depths: [Int]

    public init(config: VibeVoiceTokenizerConfiguration, dimension: Int) {
        let padMode = config.mlxPadMode
        let ratios = config.resolvedDecoderRatios
        let depths = config.parsedDecoderDepths
        self.depths = depths

        let topDim = config.decoderNFilters * (1 << (depths.count - 1))
        self._stem = ModuleInfo(wrappedValue: StreamableConv1d(
            inChannels: dimension, outChannels: topDim,
            ksize: config.kernelSize, stride: 1, dilation: 1, groups: 1,
            bias: config.convBias, causal: config.causal, padMode: padMode
        ), key: "stem")

        self._upsampleLayers = ModuleInfo(wrappedValue: ratios.enumerated().map { index, ratio in
            StreamableConvTranspose1d(
                inChannels: config.decoderNFilters * (1 << (depths.count - 1 - index)),
                outChannels: config.decoderNFilters * (1 << (depths.count - 2 - index)),
                ksize: ratio * 2, stride: ratio, groups: 1,
                bias: config.convBias, causal: config.causal
            )
        }, key: "upsample_layers")

        self._stages = ModuleInfo(wrappedValue: depths.enumerated().map { index, depth in
            let dim = config.decoderNFilters * (1 << (depths.count - 1 - index))
            return (0 ..< depth).map { _ in
                VibeVoiceBlock1D(dim: dim, config: config, padMode: padMode)
            }
        }, key: "stages")

        self._head = ModuleInfo(wrappedValue: StreamableConv1d(
            inChannels: config.decoderNFilters, outChannels: config.channels,
            ksize: config.lastKernelSize, stride: 1, dilation: 1, groups: 1,
            bias: config.convBias, causal: config.causal, padMode: padMode
        ), key: "head")
    }

    public func callAsFunction(_ latentsNCL: MLXArray) -> MLXArray {
        var x = latentsNCL
        for index in depths.indices {
            x = index == 0 ? stem(x) : upsampleLayers[index - 1](x)
            for block in stages[index] { x = block(x) }
        }
        return head(x)
    }

    public func step(_ latentsNCL: MLXArray) -> MLXArray {
        var x = latentsNCL
        for index in depths.indices {
            x = index == 0 ? stem.step(x) : upsampleLayers[index - 1].step(x)
            if x.shape[2] == 0 { return x }
            for block in stages[index] { x = block.step(x) }
        }
        return head.step(x)
    }

    public func resetState() {
        stem.resetState()
        for layer in upsampleLayers { layer.resetState() }
        for stage in stages { for block in stage { block.resetState() } }
        head.resetState()
    }
}

extension VibeVoiceTokenizerConfiguration {
    var mlxPadMode: PadMode {
        padMode == "constant" ? .constant : .edge
    }
}
