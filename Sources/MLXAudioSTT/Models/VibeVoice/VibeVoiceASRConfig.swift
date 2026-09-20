import Foundation
import MLXAudioCodecs

/// Which checkpoint family a VibeVoice ASR directory holds.
///
/// Microsoft ships the same model under two incompatible layouts, and both matter: the
/// nested one is what `microsoft/VibeVoice-ASR` and the streaming checkpoint use, while the
/// flat one (`microsoft/VibeVoice-ASR-HF`) is the native-Transformers port, which renames
/// every module, drops the unused acoustic decoder, and bundles a tokenizer.
public enum VibeVoiceASRLayout: String, Sendable {
    /// `model.language_model.*`, `model.acoustic_tokenizer.encoder.*`, `model.*_connector.*`.
    case nested
    /// `language_model.model.*`, `acoustic_tokenizer_encoder.*`, `multi_modal_projector.*`.
    case flat
}

/// Configuration for VibeVoice ASR, normalised across both checkpoint schemas.
public struct VibeVoiceASRConfiguration: Sendable {
    public var layout: VibeVoiceASRLayout
    public var acousticTokenizer: VibeVoiceTokenizerConfiguration
    public var semanticTokenizer: VibeVoiceTokenizerConfiguration
    public var textConfig: VibeVoiceQwen2Configuration

    /// Wraps the run of audio placeholders in the prompt.
    public var audioBosTokenID: Int
    public var audioEosTokenID: Int
    /// The placeholder whose embedding each audio frame replaces.
    public var audioTokenID: Int

    /// Samples per encoder segment for long audio. Upstream splits at 60 s to keep the
    /// convolution stack from overflowing its 32-bit indexing.
    public var acousticTokenizerChunkSize: Int

    public var samplingRate: Int = 24000

    /// Samples consumed per audio token (3200, i.e. 7.5 Hz at 24 kHz).
    public var compressionRatio: Int { acousticTokenizer.compressionRatio }

    public init(from data: Data) throws {
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]

        if json["text_config"] != nil {
            layout = .flat
            let textData = try JSONSerialization.data(withJSONObject: json["text_config"] ?? [:])
            textConfig = try JSONDecoder().decode(VibeVoiceQwen2Configuration.self, from: textData)
            acousticTokenizer = try Self.flatTokenizerConfig(
                json["acoustic_tokenizer_encoder_config"] as? [String: Any] ?? [:])
            semanticTokenizer = try Self.flatTokenizerConfig(
                json["semantic_tokenizer_encoder_config"] as? [String: Any] ?? [:])
        } else {
            layout = .nested
            let textData = try JSONSerialization.data(withJSONObject: json["decoder_config"] ?? [:])
            textConfig = try JSONDecoder().decode(VibeVoiceQwen2Configuration.self, from: textData)
            let acousticData = try JSONSerialization.data(
                withJSONObject: json["acoustic_tokenizer_config"] ?? [:])
            let semanticData = try JSONSerialization.data(
                withJSONObject: json["semantic_tokenizer_config"] ?? [:])
            acousticTokenizer = try JSONDecoder().decode(
                VibeVoiceTokenizerConfiguration.self, from: acousticData)
            semanticTokenizer = try JSONDecoder().decode(
                VibeVoiceTokenizerConfiguration.self, from: semanticData)
        }

        // Only the flat schema names these; the nested one relies on the tokenizer's
        // Qwen2.5 vocabulary, where they are fixed.
        audioBosTokenID = json["audio_bos_token_id"] as? Int ?? 151_646
        audioEosTokenID = json["audio_eos_token_id"] as? Int ?? 151_647
        audioTokenID = json["audio_token_id"] as? Int ?? 151_648
        acousticTokenizerChunkSize = json["acoustic_tokenizer_chunk_size"] as? Int ?? 1_440_000
        samplingRate = json["target_sample_rate"] as? Int ?? 24000
    }

    /// Maps the flat schema's encoder config onto the shared tokenizer configuration.
    ///
    /// Two traps: `downsampling_ratios` is already in encoder application order (the nested
    /// schema stores the reverse and the encoder flips it), and `hidden_size` here means the
    /// latent width, not a transformer width.
    private static func flatTokenizerConfig(
        _ json: [String: Any]
    ) throws -> VibeVoiceTokenizerConfiguration {
        let ratios = json["downsampling_ratios"] as? [Int] ?? [2, 2, 4, 5, 5, 8]
        let depths = json["depths"] as? [Int] ?? [3, 3, 3, 3, 3, 3, 8]

        var rebuilt: [String: Any] = [
            "channels": json["channels"] ?? 1,
            "causal": true,
            "vae_dim": json["hidden_size"] ?? 64,
            "encoder_n_filters": json["num_filters"] ?? 32,
            "encoder_ratios": Array(ratios.reversed()),
            "encoder_depths": depths.map(String.init).joined(separator: "-"),
            "decoder_n_filters": json["num_filters"] ?? 32,
            "layernorm_eps": json["rms_norm_eps"] ?? 1e-5,
            "layer_scale_init_value": json["layer_scale_init_value"] ?? 1e-6,
            "kernel_size": json["kernel_size"] ?? 7,
            "last_kernel_size": json["kernel_size"] ?? 7,
            "ffn_expansion": json["ffn_expansion"] ?? 4,
            "layernorm": "RMSNorm",
            "pad_mode": "constant",
            "conv_norm": "none",
            "conv_bias": true,
            "disable_last_norm": true,
            "mixer_layer": "depthwise_conv",
        ]
        // `vae_std` is the already-scaled standard deviation (fix_std / 0.8), so undo that
        // to keep one meaning for `fix_std` across both schemas.
        if let vaeStd = json["vae_std"] as? Double {
            rebuilt["fix_std"] = vaeStd * 0.8
            rebuilt["std_dist_type"] = "gaussian"
        }

        let data = try JSONSerialization.data(withJSONObject: rebuilt)
        return try JSONDecoder().decode(VibeVoiceTokenizerConfiguration.self, from: data)
    }
}
