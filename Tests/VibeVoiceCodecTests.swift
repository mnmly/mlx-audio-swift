import Foundation
@preconcurrency import MLX
import MLXNN
import Testing

@testable import MLXAudioCodecs

/// Reference values captured from VibeVoice's PyTorch `SConv1d` / `SConvTranspose1d`
/// (`vibevoice/modular/modular_vibevoice_tokenizer.py`) with deterministic weights
/// `w[i] = (i % 7 - 3) / 10` and input `x[i] = (i % 11 - 5) / 10`.
///
/// VibeVoice's causal convolutions use `padding_total = (k - 1) * dilation - (stride - 1)`,
/// which is the same quantity `StreamableConv1d` calls `kEff - stride`, so the Mimi
/// primitives can back the VibeVoice tokenizer directly. These fixtures pin that down.
private enum VibeVoiceConvGolden {
    // Conv1d: in 2, out 3, kernel 4, stride 2, causal, constant padding. Input is (1, 2, 10).
    static let conv1dWeight: [Float] = [-0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, -0.300000, -0.200000, -0.100000]
    static let conv1dBias: [Float] = [-0.300000, -0.200000, -0.100000]
    static let conv1dInput: [Float] = [-0.500000, -0.400000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, -0.500000, -0.400000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000]
    static let conv1dExpected: [Float] = [0.050000, -0.120000, -0.290000, -0.350000, -0.410000, -0.290000, 0.050000, -0.210000, -0.250000, -0.290000, -0.280000, 0.290000, -0.060000, -0.080000, -0.100000]

    // ConvTranspose1d: in 3, out 2, kernel 4, stride 2, causal, trim_right_ratio 1.0. Input is (1, 3, 5).
    static let convTr1dWeight: [Float] = [-0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, -0.300000, -0.200000, -0.100000]
    static let convTr1dBias: [Float] = [-0.300000, -0.200000]
    static let convTr1dInput: [Float] = [-0.500000, -0.400000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, -0.500000, -0.400000, -0.300000, -0.200000]
    static let convTr1dExpected: [Float] = [-0.200000, -0.200000, -0.050000, -0.130000, -0.220000, -0.350000, -0.280000, -0.350000, -0.340000, -0.350000, -0.100000, -0.450000, -0.620000, 0.000000, -0.360000, 0.070000, -0.320000, 0.030000, -0.280000, -0.010000]
}

private func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
    MLX.max(MLX.abs(lhs - rhs)).item(Float.self)
}

/// `conv1d` reproduces PyTorch bit-for-bit on both the CPU and GPU backends.
private let convTolerance: Float = 1e-5

/// MLX's Metal `conv_transpose1d` kernel accumulates at roughly bf16 precision: against
/// PyTorch it lands within ~1.6e-3 absolute (~7e-4 relative) on the GPU, versus ~2.4e-7 on
/// the CPU backend. That is a ~-56 dBFS noise floor on decoded audio, which is inaudible,
/// but it is far above float32 round-off — so the transpose fixtures need their own bound.
private let convTransposeTolerance: Float = 5e-3

@Suite("VibeVoice acoustic tokenizer convolutions")
struct VibeVoiceCodecTests {

    /// The causal `StreamableConv1d` must reproduce VibeVoice's offline `SConv1d`.
    @Test func streamableConv1dMatchesVibeVoiceReference() throws {
        let conv = StreamableConv1d(
            inChannels: 2, outChannels: 3, ksize: 4, stride: 2,
            dilation: 1, groups: 1, bias: true, causal: true, padMode: .constant
        )
        // PyTorch Conv1d weights are (out, in, k); MLX expects (out, k, in).
        let weight = MLXArray(VibeVoiceConvGolden.conv1dWeight, [3, 2, 4]).transposed(0, 2, 1)
        try conv.update(parameters: ModuleParameters.unflattened([
            "conv.conv.weight": weight,
            "conv.conv.bias": MLXArray(VibeVoiceConvGolden.conv1dBias),
        ]), verify: .none)

        let x = MLXArray(VibeVoiceConvGolden.conv1dInput, [1, 2, 10])
        let expected = MLXArray(VibeVoiceConvGolden.conv1dExpected, [1, 3, 5])

        let y = conv(x)
        #expect(y.shape == expected.shape)
        #expect(maxAbsDiff(y, expected) < convTolerance)
    }

    /// Streaming the same input in ragged chunks must equal the offline result.
    /// An off-by-one in the left-context carry shifts every frame downstream, so this
    /// guards the whole 7.5 Hz tokenizer.
    @Test func streamableConv1dStreamingMatchesOffline() throws {
        let conv = StreamableConv1d(
            inChannels: 2, outChannels: 3, ksize: 4, stride: 2,
            dilation: 1, groups: 1, bias: true, causal: true, padMode: .constant
        )
        let weight = MLXArray(VibeVoiceConvGolden.conv1dWeight, [3, 2, 4]).transposed(0, 2, 1)
        try conv.update(parameters: ModuleParameters.unflattened([
            "conv.conv.weight": weight,
            "conv.conv.bias": MLXArray(VibeVoiceConvGolden.conv1dBias),
        ]), verify: .none)

        let x = MLXArray(VibeVoiceConvGolden.conv1dInput, [1, 2, 10])
        let expected = MLXArray(VibeVoiceConvGolden.conv1dExpected, [1, 3, 5])

        conv.resetState()
        var chunks: [MLXArray] = []
        var offset = 0
        for length in [4, 4, 2] {
            let piece = x[0..., 0..., offset ..< (offset + length)]
            offset += length
            let out = conv.step(piece)
            if out.shape[2] > 0 { chunks.append(out) }
        }
        let streamed = concatenated(chunks, axis: 2)

        #expect(streamed.shape == expected.shape)
        #expect(maxAbsDiff(streamed, expected) < convTolerance)
    }

    /// The causal `StreamableConvTranspose1d` must reproduce VibeVoice's offline
    /// `SConvTranspose1d` with `trim_right_ratio = 1.0`.
    @Test func streamableConvTranspose1dMatchesVibeVoiceReference() throws {
        let convtr = StreamableConvTranspose1d(
            inChannels: 3, outChannels: 2, ksize: 4, stride: 2,
            groups: 1, bias: true, causal: true
        )
        // PyTorch ConvTranspose1d weights are (in, out, k); MLX expects (out, k, in).
        let weight = MLXArray(VibeVoiceConvGolden.convTr1dWeight, [3, 2, 4]).transposed(1, 2, 0)
        try convtr.update(parameters: ModuleParameters.unflattened([
            "convtr.convtr.weight": weight,
            "convtr.convtr.bias": MLXArray(VibeVoiceConvGolden.convTr1dBias),
        ]), verify: .none)

        let x = MLXArray(VibeVoiceConvGolden.convTr1dInput, [1, 3, 5])
        let expected = MLXArray(VibeVoiceConvGolden.convTr1dExpected, [1, 2, 10])

        let y = convtr(x)
        #expect(y.shape == expected.shape)
        #expect(maxAbsDiff(y, expected) < convTransposeTolerance)
    }

    /// Mimi's transpose streams by overlap-add while VibeVoice recomputes from cached
    /// context; both must land on the same samples, one latent frame at a time.
    @Test func streamableConvTranspose1dStreamingMatchesOffline() throws {
        let convtr = StreamableConvTranspose1d(
            inChannels: 3, outChannels: 2, ksize: 4, stride: 2,
            groups: 1, bias: true, causal: true
        )
        let weight = MLXArray(VibeVoiceConvGolden.convTr1dWeight, [3, 2, 4]).transposed(1, 2, 0)
        try convtr.update(parameters: ModuleParameters.unflattened([
            "convtr.convtr.weight": weight,
            "convtr.convtr.bias": MLXArray(VibeVoiceConvGolden.convTr1dBias),
        ]), verify: .none)

        let x = MLXArray(VibeVoiceConvGolden.convTr1dInput, [1, 3, 5])
        let expected = MLXArray(VibeVoiceConvGolden.convTr1dExpected, [1, 2, 10])

        convtr.resetState()
        var chunks: [MLXArray] = []
        for frame in 0 ..< 5 {
            let out = convtr.step(x[0..., 0..., frame ..< (frame + 1)])
            if out.shape[2] > 0 { chunks.append(out) }
        }
        let streamed = concatenated(chunks, axis: 2)

        #expect(streamed.shape == expected.shape)
        #expect(maxAbsDiff(streamed, expected) < convTransposeTolerance)
    }
}
