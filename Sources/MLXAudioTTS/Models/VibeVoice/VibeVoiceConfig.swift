import Foundation
import MLXAudioCodecs

/// Top-level configuration for `model_type: vibevoice_streaming`
/// (microsoft/VibeVoice-Realtime-0.5B).
public struct VibeVoiceConfiguration: Codable, Sendable {
    public var acousticTokenizerConfig: VibeVoiceTokenizerConfiguration
    public var decoderConfig: VibeVoiceQwen2Configuration
    public var diffusionHeadConfig: VibeVoiceDiffusionHeadConfiguration

    /// How many of the decoder's layers form the upper, text+speech stack. The remaining
    /// layers form the lower, text-only stack.
    public var ttsBackboneNumHiddenLayers: Int

    public var acousticVaeDim: Int

    enum CodingKeys: String, CodingKey {
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case decoderConfig = "decoder_config"
        case diffusionHeadConfig = "diffusion_head_config"
        case ttsBackboneNumHiddenLayers = "tts_backbone_num_hidden_layers"
        case acousticVaeDim = "acoustic_vae_dim"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        acousticTokenizerConfig = try c.decode(
            VibeVoiceTokenizerConfiguration.self, forKey: .acousticTokenizerConfig)
        decoderConfig = try c.decode(VibeVoiceQwen2Configuration.self, forKey: .decoderConfig)
        diffusionHeadConfig = try c.decode(
            VibeVoiceDiffusionHeadConfiguration.self, forKey: .diffusionHeadConfig)
        ttsBackboneNumHiddenLayers = try c.decodeIfPresent(
            Int.self, forKey: .ttsBackboneNumHiddenLayers) ?? 20
        acousticVaeDim = try c.decodeIfPresent(Int.self, forKey: .acousticVaeDim)
            ?? acousticTokenizerConfig.vaeDim
    }

    /// Layers in the lower, text-only stack.
    public var languageModelLayers: Int {
        decoderConfig.numHiddenLayers - ttsBackboneNumHiddenLayers
    }
}
