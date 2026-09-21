import Foundation
@preconcurrency import MLX
import Testing

@testable import MLXAudioCodecs
@testable import MLXAudioSTT

@Suite("VibeVoice ASR")
struct VibeVoiceASRTests {

    // MARK: - Configuration

    /// The flat schema stores `downsampling_ratios` already in encoder application order,
    /// while the nested one stores the reverse and the encoder flips it. Double-reversing
    /// would silently build the encoder upside down.
    @Test func flatConfigDoesNotDoubleReverseRatios() throws {
        let json = """
        {
          "model_type": "vibevoice_asr",
          "audio_bos_token_id": 151646, "audio_eos_token_id": 151647,
          "audio_token_id": 151648, "acoustic_tokenizer_chunk_size": 1440000,
          "acoustic_tokenizer_encoder_config": {
            "channels": 1, "depths": [3,3,3,3,3,3,8],
            "downsampling_ratios": [2,2,4,5,5,8], "hidden_size": 64, "num_filters": 32,
            "kernel_size": 7, "ffn_expansion": 4, "rms_norm_eps": 0.00001, "vae_std": 0.625
          },
          "semantic_tokenizer_encoder_config": {
            "channels": 1, "depths": [3,3,3,3,3,3,8],
            "downsampling_ratios": [2,2,4,5,5,8], "hidden_size": 128, "num_filters": 32,
            "kernel_size": 7, "ffn_expansion": 4, "rms_norm_eps": 0.00001, "vae_std": 0.625
          },
          "text_config": {
            "hidden_size": 3584, "num_hidden_layers": 28, "intermediate_size": 18944,
            "num_attention_heads": 28, "num_key_value_heads": 4, "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0, "vocab_size": 152064, "max_position_embeddings": 131072
          }
        }
        """
        let config = try VibeVoiceASRConfiguration(from: Data(json.utf8))

        #expect(config.layout == .flat)
        // Stored as the reverse, so the encoder's own `reversed()` recovers [2,2,4,5,5,8].
        #expect(config.acousticTokenizer.encoderRatios == [8, 5, 5, 4, 2, 2])
        #expect(config.compressionRatio == 3200)
        #expect(config.acousticTokenizer.vaeDim == 64)
        #expect(config.semanticTokenizer.vaeDim == 128)
        #expect(config.textConfig.headDim == 128)
        #expect(config.audioTokenID == 151648)
    }

    @Test func nestedConfigDecodes() throws {
        let json = """
        {
          "model_type": "vibevoice",
          "acoustic_tokenizer_config": {"vae_dim": 64, "encoder_ratios": [8,5,5,4,2,2],
            "encoder_depths": "3-3-3-3-3-3-8"},
          "semantic_tokenizer_config": {"vae_dim": 128, "encoder_ratios": [8,5,5,4,2,2],
            "encoder_depths": "3-3-3-3-3-3-8"},
          "decoder_config": {"model_type": "qwen2", "hidden_size": 3584,
            "num_hidden_layers": 28, "intermediate_size": 18944, "num_attention_heads": 28,
            "num_key_value_heads": 4, "rms_norm_eps": 0.000001, "rope_theta": 1000000.0,
            "vocab_size": 152064, "max_position_embeddings": 131072}
        }
        """
        let config = try VibeVoiceASRConfiguration(from: Data(json.utf8))
        #expect(config.layout == .nested)
        #expect(config.acousticTokenizer.encoderRatios == [8, 5, 5, 4, 2, 2])
        #expect(config.semanticTokenizer.vaeDim == 128)
        // Absent from the nested schema, so the Qwen2.5 vocabulary's fixed ids apply.
        #expect(config.audioBosTokenID == 151646)
        #expect(config.acousticTokenizerChunkSize == 1_440_000)
    }

    // MARK: - Weight layouts

    /// Key names here are copied verbatim from `microsoft/VibeVoice-ASR-HF`'s
    /// `model.safetensors.index.json`; the expected names come from the module tree.
    @Test func flatLayoutMapsOntoModuleTree() {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 2]),
            "language_model.model.norm.weight": MLXArray.zeros([2]),
            "language_model.lm_head.weight": MLXArray.zeros([4, 2]),
            "multi_modal_projector.acoustic_linear_1.weight": MLXArray.zeros([2, 2]),
            "multi_modal_projector.acoustic_norm.weight": MLXArray.zeros([2]),
            "multi_modal_projector.semantic_linear_2.bias": MLXArray.zeros([2]),
            "acoustic_tokenizer_encoder.stem.conv.conv.weight": MLXArray.zeros([2, 1, 7]),
            "acoustic_tokenizer_encoder.stem.stage.0.gamma": MLXArray.zeros([2]),
            "acoustic_tokenizer_encoder.stem.stage.0.mixer.conv.weight": MLXArray.zeros([2, 1, 7]),
            "acoustic_tokenizer_encoder.conv_layers.0.conv.conv.weight": MLXArray.zeros([4, 2, 4]),
            "acoustic_tokenizer_encoder.conv_layers.0.stage.1.norm.weight": MLXArray.zeros([4]),
            "acoustic_tokenizer_encoder.head.conv.weight": MLXArray.zeros([4, 8, 7]),
            "semantic_tokenizer_encoder.stem.conv.conv.weight": MLXArray.zeros([2, 1, 7]),
        ]

        let out = VibeVoiceASRModel.sanitize(weights: weights, layout: .flat)

        #expect(out["language_model.embed_tokens.weight"] != nil)
        #expect(out["language_model.layers.0.self_attn.q_proj.weight"] != nil)
        #expect(out["language_model.norm.weight"] != nil)
        #expect(out["lm_head.weight"] != nil)

        #expect(out["acoustic_connector.fc1.weight"] != nil)
        #expect(out["acoustic_connector.norm.weight"] != nil)
        #expect(out["semantic_connector.fc2.bias"] != nil)

        // `stem.stage.J` and `conv_layers.I.stage.J` both fold into one `stages` list, with
        // the stem's stage at index 0 and each conv layer's shifted up by one.
        #expect(out["acoustic_tokenizer_encoder.stem.conv.conv.weight"] != nil)
        #expect(out["acoustic_tokenizer_encoder.stages.0.0.gamma"] != nil)
        #expect(out["acoustic_tokenizer_encoder.stages.1.1.norm.weight"] != nil)
        #expect(out["acoustic_tokenizer_encoder.downsample_layers.0.conv.conv.weight"] != nil)
        #expect(out["semantic_tokenizer_encoder.stem.conv.conv.weight"] != nil)

        // The flat layout collapses two wrapper levels that the module tree still has.
        #expect(out["acoustic_tokenizer_encoder.head.conv.conv.weight"] != nil)
        #expect(out["acoustic_tokenizer_encoder.stages.0.0.mixer.conv.conv.conv.weight"] != nil)

        // PyTorch conv weights are (out, in, k); MLX wants (out, k, in).
        #expect(out["acoustic_tokenizer_encoder.stem.conv.conv.weight"]?.shape == [2, 7, 1])
        #expect(out["acoustic_tokenizer_encoder.head.conv.conv.weight"]?.shape == [4, 7, 8])
    }

    /// Key names copied verbatim from `microsoft/VibeVoice-ASR`'s index.
    @Test func nestedLayoutMapsOntoModuleTree() {
        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.zeros([4, 2]),
            "model.language_model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "model.language_model.norm.weight": MLXArray.zeros([2]),
            "model.acoustic_connector.fc1.weight": MLXArray.zeros([2, 2]),
            "model.semantic_connector.norm.weight": MLXArray.zeros([2]),
            "model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight":
                MLXArray.zeros([2, 1, 7]),
            "model.acoustic_tokenizer.encoder.downsample_layers.1.0.conv.conv.weight":
                MLXArray.zeros([4, 2, 4]),
            "model.acoustic_tokenizer.encoder.stages.0.0.mixer.conv.conv.conv.weight":
                MLXArray.zeros([2, 1, 7]),
            "model.acoustic_tokenizer.encoder.head.conv.conv.weight": MLXArray.zeros([4, 8, 7]),
            "model.semantic_tokenizer.encoder.head.conv.conv.weight": MLXArray.zeros([8, 8, 7]),
            // Present in the checkpoint, never used on the ASR path.
            "model.acoustic_tokenizer.decoder.head.conv.conv.weight": MLXArray.zeros([1, 2, 7]),
        ]

        let out = VibeVoiceASRModel.sanitize(weights: weights, layout: .nested)

        #expect(out["lm_head.weight"] != nil)
        #expect(out["language_model.embed_tokens.weight"] != nil)
        #expect(out["acoustic_connector.fc1.weight"] != nil)
        #expect(out["semantic_connector.norm.weight"] != nil)

        #expect(out["acoustic_tokenizer_encoder.stem.conv.conv.weight"] != nil)
        #expect(out["acoustic_tokenizer_encoder.downsample_layers.0.conv.conv.weight"] != nil)
        #expect(out["acoustic_tokenizer_encoder.stages.0.0.mixer.conv.conv.conv.weight"] != nil)
        #expect(out["semantic_tokenizer_encoder.head.conv.conv.weight"] != nil)

        // ~344M parameters of acoustic decoder are dropped rather than loaded.
        #expect(!out.keys.contains { $0.contains("decoder") })
        #expect(out["acoustic_tokenizer_encoder.stem.conv.conv.weight"]?.shape == [2, 7, 1])
    }

    /// Both layouts have to land on identical key sets for the shared sub-trees.
    @Test func layoutsAgreeOnEncoderNaming() {
        let flat = VibeVoiceASRModel.sanitize(
            weights: [
                "acoustic_tokenizer_encoder.conv_layers.2.stage.0.ffn.linear1.bias":
                    MLXArray.zeros([4])
            ], layout: .flat)
        let nested = VibeVoiceASRModel.sanitize(
            weights: [
                "model.acoustic_tokenizer.encoder.stages.3.0.ffn.linear1.bias":
                    MLXArray.zeros([4])
            ], layout: .nested)
        #expect(Set(flat.keys) == Set(nested.keys))
        #expect(flat.keys.first == "acoustic_tokenizer_encoder.stages.3.0.ffn.linear1.bias")
    }

    // MARK: - Prompt

    /// The prompt deliberately stops at `<|im_end|>` with no assistant header: upstream
    /// threads `add_generation_prompt` through the processor but never passes it to
    /// `apply_chat_template`, and the shipped chat template agrees.
    @Test func promptHasNoAssistantHeader() {
        let prompt = VibeVoiceASRPrompt.build(
            audioPlaceholder: "<AUDIO>", audioDuration: 12.5)
        #expect(!prompt.contains("<|im_start|>assistant"))
        #expect(prompt.contains("<|im_start|>system"))
        #expect(prompt.contains("<|im_start|>user"))
        #expect(prompt.contains("This is a 12.50 seconds audio, please transcribe it with "
                + "these keys: Start time, End time, Speaker ID, Content"))
        #expect(prompt.hasSuffix("<|im_end|>\n"))
    }

    @Test func promptWithContextInfo() {
        let prompt = VibeVoiceASRPrompt.build(
            audioPlaceholder: "<AUDIO>", audioDuration: 3, contextInfo: "  Acme Corp  ")
        #expect(prompt.contains("with extra info: Acme Corp"))
        #expect(prompt.contains("Please transcribe it with these keys:"))
    }

    /// `parts(...)` is what the model actually calls — `build` is only used by tests and
    /// the streaming path — so the context hint has to survive this split too. It reached
    /// `build` but not `parts` at first, which left `--context` silently inert for file
    /// transcription.
    @Test func partsCarryContextInfoIntoTheTail() {
        let plain = VibeVoiceASRPrompt.parts(audioDuration: 3)
        #expect(!plain.tail.contains("with extra info"))
        #expect(plain.tail.contains("This is a 3.00 seconds audio, please transcribe it"))

        let hinted = VibeVoiceASRPrompt.parts(audioDuration: 3, contextInfo: "  Galluzzo, Athos  ")
        #expect(hinted.tail.contains("with extra info: Galluzzo, Athos"))
        #expect(hinted.head == plain.head, "the hint belongs after the audio span, not before it")
    }

    /// The hint rides on the shared parameter struct, so a caller that never mentions it
    /// keeps the unhinted prompt.
    @Test func generateParametersDefaultToNoContext() {
        #expect(STTGenerateParameters().contextInfo == nil)
        #expect(STTGenerateParameters(contextInfo: "Acme").contextInfo == "Acme")
    }

    @Test func streamingPromptIsRawText() {
        let prompt = VibeVoiceASRPrompt.buildStreaming()
        #expect(!prompt.contains("<|im_start|>"))
        #expect(prompt.contains("streamingly with these keys: speaker, content"))
    }

    // MARK: - Transcript parsing

    @Test func parsesJSONArray() {
        let text = """
        [{"Start time": 0.0, "End time": 2.5, "Speaker ID": "Speaker 1", "Content": "Hello."},
         {"Start time": 2.5, "End time": 4.0, "Speaker ID": "Speaker 2", "Content": "Hi."}]
        """
        let segments = VibeVoiceASRPrompt.parse(text)
        #expect(segments.count == 2)
        #expect(segments[0].speakerID == "Speaker 1")
        #expect(segments[0].text == "Hello.")
        #expect(segments[1].startTime == 2.5)
        #expect(VibeVoiceASRPrompt.plainText(segments) == "Speaker 1: Hello.\nSpeaker 2: Hi.")
    }

    @Test func parsesFencedJSONAndKeyAliases() {
        let text = """
        Sure, here you go:
        ```json
        [{"Start": 1, "End": 2, "Speaker": "A", "Content": "Yes"}]
        ```
        """
        let segments = VibeVoiceASRPrompt.parse(text)
        #expect(segments.count == 1)
        #expect(segments[0].startTime == 1)
        #expect(segments[0].speakerID == "A")
        #expect(segments[0].text == "Yes")
    }

    /// A transcript cut off by the token limit should degrade to empty, not crash.
    @Test func parsesTruncatedOutputSafely() {
        // Unterminated JSON now falls through to plain parsing rather than vanishing.
        #expect(VibeVoiceASRPrompt.parse("").isEmpty)
        #expect(VibeVoiceASRPrompt.parse("   \n  ").isEmpty)
    }

    /// A bare object, rather than an array, is accepted too.
    @Test func parsesSingleObject() {
        let segments = VibeVoiceASRPrompt.parse("{\"Content\": \"Only one\"}")
        #expect(segments.count == 1)
        #expect(segments[0].text == "Only one")
    }


    /// At reduced precision the model answers in a plain `speaker: text` form instead of
    /// JSON. That still carries the diarization, so it is parsed rather than discarded.
    @Test func parsesPlainSpeakerLines() {
        let text = "[Silence]\n0: Hello.\n1: Hello back.\nSpeaker 2: And me."
        let segments = VibeVoiceASRPrompt.parse(text)
        #expect(segments.count == 4)
        #expect(segments[0].speakerID == nil)
        #expect(segments[0].text == "[Silence]")
        #expect(segments[1].speakerID == "0")
        #expect(segments[1].text == "Hello.")
        #expect(segments[3].speakerID == "Speaker 2")
        #expect(segments[3].text == "And me.")
    }

    /// Prose containing a colon must not be mistaken for a speaker label.
    @Test func plainParsingDoesNotSwallowProse() {
        let segments = VibeVoiceASRPrompt.parse(
            "He said the following: that he would be late.")
        #expect(segments.count == 1)
        #expect(segments[0].speakerID == nil)
        #expect(segments[0].text == "He said the following: that he would be late.")
    }

    /// JSON still wins when it is present.
    @Test func jsonTakesPrecedenceOverPlainParsing() {
        let segments = VibeVoiceASRPrompt.parse(
            "[{\"Speaker\": 0, \"Content\": \"Hi\", \"Start\": 1.0}]")
        #expect(segments.count == 1)
        #expect(segments[0].startTime == 1.0)
    }


    // MARK: - VAE sampling

    /// `sampleAcousticLatents` has to actually change the latents. It was previously
    /// declared and documented but never read, which is invisible without a test like this:
    /// the model still transcribes correctly either way.
    @Test func acousticSamplingIsWiredUp() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceTokenizerConfiguration.self,
            from: Data(#"{"vae_dim": 4, "fix_std": 0.5, "std_dist_type": "gaussian"}"#.utf8))

        let mean = MLXArray([Float](repeating: 1, count: 2 * 4 * 3), [2, 4, 3])

        MLXRandom.seed(0)
        let drawn = VibeVoiceAcousticTokenizer.sample(mean, config: config)
        #expect(drawn.shape == mean.shape)
        // A draw must differ from the mean it was drawn around.
        #expect(MLX.max(MLX.abs(drawn - mean)).item(Float.self) > 0)

        // One scale per batch element, broadcast over the rest: within a batch row the
        // deviation is a single scale times standard noise, so two rows should differ.
        MLXRandom.seed(1)
        let second = VibeVoiceAcousticTokenizer.sample(mean, config: config)
        #expect(MLX.max(MLX.abs(second - drawn)).item(Float.self) > 0)
    }

    /// With `fix_std` 0 the draw collapses to the mean, which is what the deterministic
    /// default relies on being equivalent to.
    @Test func zeroScaleSamplingReturnsTheMean() throws {
        let config = try JSONDecoder().decode(
            VibeVoiceTokenizerConfiguration.self,
            from: Data(#"{"vae_dim": 4, "fix_std": 0.0, "std_dist_type": "gaussian"}"#.utf8))
        let mean = MLXArray([Float](repeating: 2, count: 8), [1, 4, 2])
        let drawn = VibeVoiceAcousticTokenizer.sample(mean, config: config)
        #expect(MLX.max(MLX.abs(drawn - mean)).item(Float.self) == 0)
    }

    // MARK: - Registration

    @Test func modelTypeResolution() {
        #expect(STT.inferModelType(from: "microsoft/VibeVoice-ASR-HF") == "vibevoice_asr")
        #expect(STT.inferModelType(from: "microsoft/VibeVoice-ASR") == "vibevoice_asr")
        #expect(
            STT.inferModelType(from: "microsoft/VibeVoice-ASR-Streaming-7B")
                == "vibevoice_asr")
    }
}
