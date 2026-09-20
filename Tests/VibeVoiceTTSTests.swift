import Foundation
@preconcurrency import MLX
import MLXNN
import Testing

@testable import MLXAudioCodecs
@testable import MLXAudioTTS

/// Every tensor in the PyTorch `state_dict()` of a miniature `VibeVoiceDiffusionHead`
/// (hidden 8, 2 layers, ffn ratio 2, latent 4). Note the absence of
/// `final_layer.norm_final.weight`: that norm is built with `elementwise_affine=False`.
private let tinyHeadStateDict: [(String, [Int])] = [
    ("noisy_images_proj.weight", [8, 4]),
    ("cond_proj.weight", [8, 8]),
    ("t_embedder.mlp.0.weight", [8, 256]),
    ("t_embedder.mlp.2.weight", [8, 8]),
    ("layers.0.ffn.gate_proj.weight", [16, 8]),
    ("layers.0.ffn.up_proj.weight", [16, 8]),
    ("layers.0.ffn.down_proj.weight", [8, 16]),
    ("layers.0.norm.weight", [8]),
    ("layers.0.adaLN_modulation.1.weight", [24, 8]),
    ("layers.1.ffn.gate_proj.weight", [16, 8]),
    ("layers.1.ffn.up_proj.weight", [16, 8]),
    ("layers.1.ffn.down_proj.weight", [8, 16]),
    ("layers.1.norm.weight", [8]),
    ("layers.1.adaLN_modulation.1.weight", [24, 8]),
    ("final_layer.linear.weight", [4, 8]),
    ("final_layer.adaLN_modulation.1.weight", [16, 8]),
]

/// Reference values from that model with every tensor filled by `w[i] = (i % 7 - 3) / 10`.
private enum VibeVoiceHeadGolden {
    static let noisy: [Float] = [-0.500000, -0.400000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000]
    static let condition: [Float] = [-0.400000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000, 0.300000, 0.400000, -0.400000, -0.300000, -0.200000, -0.100000, 0.000000, 0.100000, 0.200000]
    static let timestepEmbedding: [Float] = [0.400412, 1.183161, -0.731068, 0.076987, -1.121804, -0.351095, 0.543406, 0.400412, 0.360429, 0.838461, 1.386631, -0.865061, -0.266111, -1.026751, -0.427599, 0.360429]
    static let output: [Float] = [-0.286737, 0.416408, -0.127678, -0.158501, 0.219066, -0.075763, 0.519544, -0.566527]
}

private let tinyHeadConfigJSON = """
{
  "hidden_size": 8, "head_layers": 2, "head_ffn_ratio": 2.0, "rms_norm_eps": 0.00001,
  "latent_size": 4, "prediction_type": "v_prediction", "ddpm_num_steps": 1000,
  "ddpm_num_inference_steps": 20, "ddpm_beta_schedule": "cosine"
}
"""

/// Largest elementwise difference relative to the reference's peak magnitude.
private func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
    let scale = MLX.max(MLX.abs(expected)).item(Float.self)
    let diff = MLX.max(MLX.abs(actual - expected)).item(Float.self)
    return scale > 0 ? diff / scale : diff
}

/// MLX's Metal kernels accumulate at reduced precision relative to PyTorch: measured here,
/// a 256-wide `matmul` lands within ~7.4e-4 relative and `conv1d` within ~1e-3, while the
/// same calls on the CPU backend agree to ~6e-7 and elementwise math (`exp`, `cos`, `sin`)
/// is exact. So these comparisons are bounded by the backend, not by the port.
///
/// Stacking layers amplifies that floor — the head's adaLN blocks scale by `1 + scale`
/// before each residual — so the full forward pass lands near 4e-3 relative where the
/// timestep embedder alone is under 1e-3.
private let gpuRelativeTolerance: Float = 5e-3

@Suite("VibeVoice TTS")
struct VibeVoiceTTSTests {

    private func makeLoadedHead() throws -> VibeVoiceDiffusionHead {
        let config = try JSONDecoder().decode(
            VibeVoiceDiffusionHeadConfiguration.self, from: Data(tinyHeadConfigJSON.utf8))
        let head = VibeVoiceDiffusionHead(config)

        var checkpoint: [String: MLXArray] = [:]
        for (key, shape) in tinyHeadStateDict {
            let count = shape.reduce(1, *)
            checkpoint["model.prediction_head.\(key)"] =
                MLXArray((0 ..< count).map { Float(($0 % 7) - 3) / 10 }, shape)
        }
        try head.update(
            parameters: ModuleParameters.unflattened(
                VibeVoiceDiffusionHead.sanitize(
                    weights: checkpoint, prefix: "model.prediction_head.")),
            verify: .all)
        return head
    }

    /// Checkpoint keys, shapes, and the `nn.Sequential` index collapsing all have to line up.
    @Test func diffusionHeadLoadsCheckpointLayout() throws {
        _ = try makeLoadedHead()
    }

    /// The timestep embedding concatenates cosine **before** sine, opposite to the diffusers
    /// convention — getting it backwards yields noise that still has the right shape.
    @Test func timestepEmbeddingMatchesReference() throws {
        let head = try makeLoadedHead()
        let embedding = head.tEmbedder(MLXArray([Float(999), Float(500)]))
        let expected = MLXArray(VibeVoiceHeadGolden.timestepEmbedding, [2, 8])
        #expect(embedding.shape == expected.shape)
        #expect(relativeError(embedding, expected) < gpuRelativeTolerance)
    }

    /// Full forward pass: SwiGLU FFN, adaLN modulation, and the affine-free final norm.
    @Test func diffusionHeadForwardMatchesReference() throws {
        let head = try makeLoadedHead()
        let out = head(
            noisyImages: MLXArray(VibeVoiceHeadGolden.noisy, [2, 4]),
            timesteps: MLXArray([Float(999), Float(500)]),
            condition: MLXArray(VibeVoiceHeadGolden.condition, [2, 8]))
        let expected = MLXArray(VibeVoiceHeadGolden.output, [2, 4])
        #expect(out.shape == expected.shape)
        #expect(relativeError(out, expected) < gpuRelativeTolerance)
    }

    /// The factory has to route both the checkpoint's `model_type` and the repo name.
    @Test func modelTypeResolution() {
        #expect(
            TTS.resolveModelType(modelRepo: "microsoft/VibeVoice-Realtime-0.5B")
                == "vibevoice_streaming")
        #expect(
            TTS.resolveModelType(
                modelRepo: "anything", modelType: "vibevoice_streaming") == "vibevoice_streaming")
    }

    /// The 24-layer backbone splits 4 text-only + 20 text-and-speech.
    @Test func configurationSplitsBackbone() throws {
        let json = """
        {
          "acoustic_vae_dim": 64,
          "acoustic_tokenizer_config": {"vae_dim": 64, "encoder_ratios": [8,5,5,4,2,2],
            "encoder_depths": "3-3-3-3-3-3-8"},
          "decoder_config": {"hidden_size": 896, "num_hidden_layers": 24,
            "intermediate_size": 4864, "num_attention_heads": 14, "num_key_value_heads": 2,
            "rms_norm_eps": 0.000001, "rope_theta": 1000000.0, "vocab_size": 151936,
            "max_position_embeddings": 8192},
          "diffusion_head_config": {"hidden_size": 896, "head_layers": 4,
            "head_ffn_ratio": 3.0, "latent_size": 64},
          "tts_backbone_num_hidden_layers": 20
        }
        """
        let config = try JSONDecoder().decode(
            VibeVoiceConfiguration.self, from: Data(json.utf8))
        #expect(config.languageModelLayers == 4)
        #expect(config.ttsBackboneNumHiddenLayers == 20)
        #expect(config.decoderConfig.headDim == 64)
        #expect(config.acousticTokenizerConfig.compressionRatio == 3200)
    }

    /// `sanitize` has to strip the `model.` prefix, route the two sub-trees that need index
    /// remapping, and drop the upper stack's unused embedding table.
    @Test func sanitizeRoutesSubtreesAndDropsUnusedEmbedding() {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "model.tts_language_model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "model.tts_language_model.norm.weight": MLXArray.zeros([2]),
            "model.tts_input_types.weight": MLXArray.zeros([2, 2]),
            "model.speech_scaling_factor": MLXArray(Float(1)),
            "tts_eos_classifier.fc1.weight": MLXArray.zeros([2, 2]),
            "model.acoustic_tokenizer.decoder.upsample_layers.0.0.conv.conv.weight":
                MLXArray.zeros([8, 4, 7]),
            "model.acoustic_tokenizer.decoder.upsample_layers.1.0.convtr.convtr.weight":
                MLXArray.zeros([8, 4, 4]),
            "model.prediction_head.layers.0.adaLN_modulation.1.weight": MLXArray.zeros([24, 8]),
            "model.prediction_head.t_embedder.mlp.2.weight": MLXArray.zeros([8, 8]),
        ]

        let out = VibeVoiceModel.sanitize(weights: weights)

        #expect(out["tts_language_model.embed_tokens.weight"] == nil)
        #expect(out["language_model.embed_tokens.weight"] != nil)
        #expect(out["tts_language_model.norm.weight"] != nil)
        #expect(out["tts_eos_classifier.fc1.weight"] != nil)
        #expect(out["speech_scaling_factor"] != nil)

        // The single-element Sequential wrapping each resampling conv collapses away, and
        // index 0 of that list is the stem rather than an upsampling layer.
        #expect(out["acoustic_tokenizer.stem.conv.conv.weight"] != nil)
        #expect(out["acoustic_tokenizer.upsample_layers.0.convtr.convtr.weight"] != nil)

        #expect(out["prediction_head.layers.0.adaLN_modulation.0.weight"] != nil)
        #expect(out["prediction_head.t_embedder.mlp.1.weight"] != nil)
    }
}
