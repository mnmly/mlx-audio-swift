import Foundation
@preconcurrency import MLX
import MLXNN
import Testing

@testable import MLXAudioCodecs

/// A miniature acoustic tokenizer: 2 base filters, ratios [2, 2], depths 1-1-1, vae_dim 4.
/// Small enough to build in CI, structurally identical to the shipped 32-filter model.
private let tinyTokenizerConfigJSON = """
{
  "channels": 1, "causal": true, "vae_dim": 4, "fix_std": 0.5,
  "std_dist_type": "gaussian", "mixer_layer": "depthwise_conv", "conv_norm": "none",
  "pad_mode": "constant", "disable_last_norm": true, "layernorm": "RMSNorm",
  "layernorm_eps": 0.00001, "layernorm_elementwise_affine": true, "conv_bias": true,
  "layer_scale_init_value": 0.000001, "encoder_n_filters": 2, "encoder_ratios": [2, 2],
  "encoder_depths": "1-1-1", "decoder_n_filters": 2, "decoder_ratios": null,
  "decoder_depths": null
}
"""

/// Every tensor in the PyTorch `state_dict()` of that tiny model, captured from
/// `VibeVoiceAcousticTokenizerModel` upstream. If the Swift module tree drifts from the
/// checkpoint layout — a renamed submodule, a missing bias, a wrong channel count — loading
/// these with `verify: .all` fails and names the offending key.
private let tinyTokenizerStateDict: [(String, [Int])] = [
        ("encoder.downsample_layers.0.0.conv.conv.weight", [2, 1, 7]),
        ("encoder.downsample_layers.0.0.conv.conv.bias", [2]),
        ("encoder.downsample_layers.1.0.conv.conv.weight", [4, 2, 4]),
        ("encoder.downsample_layers.1.0.conv.conv.bias", [4]),
        ("encoder.downsample_layers.2.0.conv.conv.weight", [8, 4, 4]),
        ("encoder.downsample_layers.2.0.conv.conv.bias", [8]),
        ("encoder.stages.0.0.gamma", [2]),
        ("encoder.stages.0.0.ffn_gamma", [2]),
        ("encoder.stages.0.0.norm.weight", [2]),
        ("encoder.stages.0.0.ffn_norm.weight", [2]),
        ("encoder.stages.0.0.mixer.conv.conv.conv.weight", [2, 1, 7]),
        ("encoder.stages.0.0.mixer.conv.conv.conv.bias", [2]),
        ("encoder.stages.0.0.ffn.linear1.weight", [8, 2]),
        ("encoder.stages.0.0.ffn.linear1.bias", [8]),
        ("encoder.stages.0.0.ffn.linear2.weight", [2, 8]),
        ("encoder.stages.0.0.ffn.linear2.bias", [2]),
        ("encoder.stages.1.0.gamma", [4]),
        ("encoder.stages.1.0.ffn_gamma", [4]),
        ("encoder.stages.1.0.norm.weight", [4]),
        ("encoder.stages.1.0.ffn_norm.weight", [4]),
        ("encoder.stages.1.0.mixer.conv.conv.conv.weight", [4, 1, 7]),
        ("encoder.stages.1.0.mixer.conv.conv.conv.bias", [4]),
        ("encoder.stages.1.0.ffn.linear1.weight", [16, 4]),
        ("encoder.stages.1.0.ffn.linear1.bias", [16]),
        ("encoder.stages.1.0.ffn.linear2.weight", [4, 16]),
        ("encoder.stages.1.0.ffn.linear2.bias", [4]),
        ("encoder.stages.2.0.gamma", [8]),
        ("encoder.stages.2.0.ffn_gamma", [8]),
        ("encoder.stages.2.0.norm.weight", [8]),
        ("encoder.stages.2.0.ffn_norm.weight", [8]),
        ("encoder.stages.2.0.mixer.conv.conv.conv.weight", [8, 1, 7]),
        ("encoder.stages.2.0.mixer.conv.conv.conv.bias", [8]),
        ("encoder.stages.2.0.ffn.linear1.weight", [32, 8]),
        ("encoder.stages.2.0.ffn.linear1.bias", [32]),
        ("encoder.stages.2.0.ffn.linear2.weight", [8, 32]),
        ("encoder.stages.2.0.ffn.linear2.bias", [8]),
        ("encoder.head.conv.conv.weight", [4, 8, 7]),
        ("encoder.head.conv.conv.bias", [4]),
        ("decoder.upsample_layers.0.0.conv.conv.weight", [8, 4, 7]),
        ("decoder.upsample_layers.0.0.conv.conv.bias", [8]),
        ("decoder.upsample_layers.1.0.convtr.convtr.weight", [8, 4, 4]),
        ("decoder.upsample_layers.1.0.convtr.convtr.bias", [4]),
        ("decoder.upsample_layers.2.0.convtr.convtr.weight", [4, 2, 4]),
        ("decoder.upsample_layers.2.0.convtr.convtr.bias", [2]),
        ("decoder.stages.0.0.gamma", [8]),
        ("decoder.stages.0.0.ffn_gamma", [8]),
        ("decoder.stages.0.0.norm.weight", [8]),
        ("decoder.stages.0.0.ffn_norm.weight", [8]),
        ("decoder.stages.0.0.mixer.conv.conv.conv.weight", [8, 1, 7]),
        ("decoder.stages.0.0.mixer.conv.conv.conv.bias", [8]),
        ("decoder.stages.0.0.ffn.linear1.weight", [32, 8]),
        ("decoder.stages.0.0.ffn.linear1.bias", [32]),
        ("decoder.stages.0.0.ffn.linear2.weight", [8, 32]),
        ("decoder.stages.0.0.ffn.linear2.bias", [8]),
        ("decoder.stages.1.0.gamma", [4]),
        ("decoder.stages.1.0.ffn_gamma", [4]),
        ("decoder.stages.1.0.norm.weight", [4]),
        ("decoder.stages.1.0.ffn_norm.weight", [4]),
        ("decoder.stages.1.0.mixer.conv.conv.conv.weight", [4, 1, 7]),
        ("decoder.stages.1.0.mixer.conv.conv.conv.bias", [4]),
        ("decoder.stages.1.0.ffn.linear1.weight", [16, 4]),
        ("decoder.stages.1.0.ffn.linear1.bias", [16]),
        ("decoder.stages.1.0.ffn.linear2.weight", [4, 16]),
        ("decoder.stages.1.0.ffn.linear2.bias", [4]),
        ("decoder.stages.2.0.gamma", [2]),
        ("decoder.stages.2.0.ffn_gamma", [2]),
        ("decoder.stages.2.0.norm.weight", [2]),
        ("decoder.stages.2.0.ffn_norm.weight", [2]),
        ("decoder.stages.2.0.mixer.conv.conv.conv.weight", [2, 1, 7]),
        ("decoder.stages.2.0.mixer.conv.conv.conv.bias", [2]),
        ("decoder.stages.2.0.ffn.linear1.weight", [8, 2]),
        ("decoder.stages.2.0.ffn.linear1.bias", [8]),
        ("decoder.stages.2.0.ffn.linear2.weight", [2, 8]),
        ("decoder.stages.2.0.ffn.linear2.bias", [2]),
        ("decoder.head.conv.conv.weight", [1, 2, 7]),
        ("decoder.head.conv.conv.bias", [1]),
]

/// Full forward-pass reference for the tiny model, captured from PyTorch with every tensor
/// filled by `w[i] = (i % 7 - 3) / 10`. This is what catches a wrong activation, a norm on
/// the wrong axis, or a mis-ordered stage — things a key/shape check cannot see.
private enum VibeVoiceTokenizerGolden {
    static let encoderOutput: [Float] = [-0.338816, -0.238816, -0.138816, -0.038816, -0.403454, -0.303454, -0.203454, -0.103454, -0.371456, -0.271456, -0.171456, -0.071456, -0.293430, -0.193430, -0.093430, 0.006570, -0.403402, -0.303402, -0.203402, -0.103402, -0.316737, -0.216737, -0.116737, -0.016737]
    static let decoderOutput: [Float] = [-0.375081, -0.500741, -0.484985, -0.546695, -0.482444, -0.530988, -0.313302, -0.209312, -0.233130, -0.297611, -0.260901, -0.179434, -0.333379, -0.393758, -0.293905, -0.175608, -0.315959, -0.348786, -0.273302, -0.196615, -0.312881, -0.392415, -0.284052, -0.237977]
}

/// Largest elementwise difference relative to the reference's peak magnitude.
private func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
    let scale = MLX.max(MLX.abs(expected)).item(Float.self)
    let diff = MLX.max(MLX.abs(actual - expected)).item(Float.self)
    return scale > 0 ? diff / scale : diff
}

/// MLX's Metal convolution kernels accumulate at reduced precision: measured against
/// PyTorch, `conv1d` at 8 channels / kernel 7 is off by 6.5e-4 absolute (~1e-3 relative) and
/// `conv_transpose1d` by 1.6e-3, while the same calls on the CPU backend agree to ~1e-7 and
/// `rms_norm` is exact on both. The deviation is therefore in the kernels, not this port; at
/// roughly -60 dBFS on decoded audio it is inaudible, but it rules out a tight bound here.
private let gpuConvRelativeTolerance: Float = 5e-3

@Suite("VibeVoice acoustic tokenizer weights")
struct VibeVoiceTokenizerWeightTests {

    private func makeConfig() throws -> VibeVoiceTokenizerConfiguration {
        try JSONDecoder().decode(
            VibeVoiceTokenizerConfiguration.self,
            from: Data(tinyTokenizerConfigJSON.utf8))
    }

    /// The sanitized checkpoint must satisfy the module tree exactly: no missing keys, no
    /// extra keys, no shape mismatches.
    @Test func checkpointKeysAndShapesMatchModuleTree() throws {
        let config = try makeConfig()
        let tokenizer = VibeVoiceAcousticTokenizer(config: config)

        var checkpoint: [String: MLXArray] = [:]
        for (key, shape) in tinyTokenizerStateDict {
            checkpoint[key] = MLXArray.ones(shape)
        }

        let sanitized = VibeVoiceAcousticTokenizer.sanitize(weights: checkpoint)
        #expect(sanitized.count == tinyTokenizerStateDict.count)

        // `.all` verifies both that every module parameter is supplied and that the shapes
        // line up, so this is the real assertion.
        try tokenizer.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
    }

    /// Frame arithmetic: the tiny model compresses by 4, the shipped one by 3200.
    @Test func compressionRatioMatchesRatioProduct() throws {
        let config = try makeConfig()
        #expect(config.compressionRatio == 4)
        #expect(config.parsedDecoderDepths == [1, 1, 1])

        let shipped = try JSONDecoder().decode(
            VibeVoiceTokenizerConfiguration.self,
            from: Data(#"{"encoder_ratios": [8, 5, 5, 4, 2, 2], "encoder_depths": "3-3-3-3-3-3-8"}"#.utf8))
        #expect(shipped.compressionRatio == 3200)
        #expect(shipped.parsedEncoderDepths == [3, 3, 3, 3, 3, 3, 8])
        #expect(shipped.parsedDecoderDepths == [8, 3, 3, 3, 3, 3, 3])
    }

    /// A round trip must land on the frame count the processor assumes,
    /// `ceil(samples / compressionRatio)`, and decode back to the same length.
    @Test func encodeDecodeShapesRoundTrip() throws {
        let config = try makeConfig()
        let tokenizer = VibeVoiceAcousticTokenizer(config: config)

        var checkpoint: [String: MLXArray] = [:]
        for (key, shape) in tinyTokenizerStateDict {
            checkpoint[key] = MLXArray.ones(shape) * 0.05
        }
        try tokenizer.update(
            parameters: ModuleParameters.unflattened(
                VibeVoiceAcousticTokenizer.sanitize(weights: checkpoint)),
            verify: .all)

        let samples = 4 * 9
        let audio = MLXArray.zeros([1, 1, samples])
        let latents = tokenizer.encode(audio)
        #expect(latents.shape == [1, 4, 9])

        let decoded = tokenizer.decode(latents)
        #expect(decoded.shape == [1, 1, samples])
    }

    /// Decoding one latent frame at a time must equal decoding them in one call; this is the
    /// path the realtime TTS model uses to emit audio while the LM is still running.
    @Test func streamingDecodeMatchesOfflineDecode() throws {
        let config = try makeConfig()
        let tokenizer = VibeVoiceAcousticTokenizer(config: config)

        var checkpoint: [String: MLXArray] = [:]
        for (key, shape) in tinyTokenizerStateDict {
            let count = shape.reduce(1, *)
            let values = (0 ..< count).map { Float(($0 % 7) - 3) / 10 }
            checkpoint[key] = MLXArray(values, shape)
        }
        try tokenizer.update(
            parameters: ModuleParameters.unflattened(
                VibeVoiceAcousticTokenizer.sanitize(weights: checkpoint)),
            verify: .all)

        let frames = 6
        let values = (0 ..< (4 * frames)).map { Float(($0 % 11) - 5) / 10 }
        let latents = MLXArray(values, [1, 4, frames])

        let offline = tokenizer.decode(latents)

        tokenizer.resetState()
        var chunks: [MLXArray] = []
        for frame in 0 ..< frames {
            let out = tokenizer.decodeStep(latents[0..., 0..., frame ..< (frame + 1)])
            if out.shape[2] > 0 { chunks.append(out) }
        }
        let streamed = concatenated(chunks, axis: 2)

        #expect(streamed.shape == offline.shape)
        // Not bit-exact: the streaming path convolves one frame at a time while the
        // offline path takes all six at once, so the Metal kernel tiles (and therefore
        // accumulates) differently. The bound is the same one the PyTorch comparison uses.
        #expect(relativeError(streamed, offline) < gpuConvRelativeTolerance)
    }

    /// Builds the tiny model with the reference weights and checks both directions against
    /// PyTorch end to end.
    @Test func forwardPassMatchesPyTorchReference() throws {
        let tokenizer = try makeLoadedTinyTokenizer()

        let frames = 6, ratio = 4, vaeDim = 4
        let audioValues = (0 ..< (frames * ratio)).map { Float(($0 % 11) - 5) / 10 }
        let audio = MLXArray(audioValues, [1, 1, frames * ratio])

        // `encode` returns NCL; the reference permutes to (B, T, C) on the way out.
        let encoded = swappedAxes(tokenizer.encode(audio), 1, 2)
        let expectedEncoded = MLXArray(VibeVoiceTokenizerGolden.encoderOutput, [1, frames, vaeDim])
        #expect(encoded.shape == expectedEncoded.shape)
        #expect(relativeError(encoded, expectedEncoded) < gpuConvRelativeTolerance)

        let latentValues = (0 ..< (vaeDim * frames)).map { Float(($0 % 11) - 5) / 10 }
        let latents = MLXArray(latentValues, [1, vaeDim, frames])
        let decoded = tokenizer.decode(latents)
        let expectedDecoded = MLXArray(VibeVoiceTokenizerGolden.decoderOutput, [1, 1, frames * ratio])
        #expect(decoded.shape == expectedDecoded.shape)
        #expect(relativeError(decoded, expectedDecoded) < gpuConvRelativeTolerance)
    }

    /// Shared fixture: the tiny model with the deterministic reference weights loaded.
    private func makeLoadedTinyTokenizer() throws -> VibeVoiceAcousticTokenizer {
        let config = try makeConfig()
        let tokenizer = VibeVoiceAcousticTokenizer(config: config)
        var checkpoint: [String: MLXArray] = [:]
        for (key, shape) in tinyTokenizerStateDict {
            let count = shape.reduce(1, *)
            let values = (0 ..< count).map { Float(($0 % 7) - 3) / 10 }
            checkpoint[key] = MLXArray(values, shape)
        }
        try tokenizer.update(
            parameters: ModuleParameters.unflattened(
                VibeVoiceAcousticTokenizer.sanitize(weights: checkpoint)),
            verify: .all)
        return tokenizer
    }
}
