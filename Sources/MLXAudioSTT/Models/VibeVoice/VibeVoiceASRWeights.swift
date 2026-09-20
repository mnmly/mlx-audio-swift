import Foundation
@preconcurrency import MLX
import MLXAudioCodecs

extension VibeVoiceASRModel {

    /// Normalises either checkpoint layout onto this module tree.
    ///
    /// The nested layout is taken as canonical because its encoder naming already matches
    /// the shared tokenizer modules. The flat layout is not merely renamed — it regroups
    /// each downsampling conv with the stage that follows it, and it drops a level of
    /// wrapping around the head and mixer convolutions — so it needs a structural remap.
    public static func sanitize(
        weights: [String: MLXArray],
        layout: VibeVoiceASRLayout
    ) -> [String: MLXArray] {
        switch layout {
        case .nested: return sanitizeNested(weights)
        case .flat: return sanitizeFlat(weights)
        }
    }

    /// `microsoft/VibeVoice-ASR` and `microsoft/VibeVoice-ASR-Streaming-7B`.
    private static func sanitizeNested(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, value) in weights {
            // The acoustic decoder is ~344M parameters that the ASR path never touches.
            if key.hasPrefix("model.acoustic_tokenizer.decoder.") { continue }
            if key.hasPrefix("model.acoustic_tokenizer.encoder.") { continue }
            if key.hasPrefix("model.semantic_tokenizer.encoder.") { continue }
            if key.hasSuffix("fix_std") { continue }

            if key.hasPrefix("model.") {
                out[String(key.dropFirst("model.".count))] = value
            } else {
                out[key] = value
            }
        }

        // Reuse the tokenizer's own remapping for the two encoder sub-trees: it unwraps the
        // single-element Sequential around each conv and transposes conv weights into MLX's
        // channel-last kernel layout.
        for (prefix, destination) in [
            ("model.acoustic_tokenizer.encoder.", "acoustic_tokenizer_encoder"),
            ("model.semantic_tokenizer.encoder.", "semantic_tokenizer_encoder"),
        ] {
            for (key, value) in VibeVoiceAcousticTokenizer.sanitize(
                weights: weights, prefix: prefix) {
                out["\(destination).\(key)"] = value
            }
        }

        return out
    }

    /// `microsoft/VibeVoice-ASR-HF`.
    private static func sanitizeFlat(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, value) in weights {
            var mapped: String?

            if key.hasPrefix("language_model.model.") {
                mapped = "language_model." + String(key.dropFirst("language_model.model.".count))
            } else if key == "language_model.lm_head.weight" {
                mapped = "lm_head.weight"
            } else if key.hasPrefix("multi_modal_projector.") {
                mapped = mapProjector(String(key.dropFirst("multi_modal_projector.".count)))
            } else if key.hasPrefix("acoustic_tokenizer_encoder.") {
                mapped = mapFlatEncoder(
                    key, prefix: "acoustic_tokenizer_encoder.")
            } else if key.hasPrefix("semantic_tokenizer_encoder.") {
                mapped = mapFlatEncoder(
                    key, prefix: "semantic_tokenizer_encoder.")
            } else {
                mapped = key
            }

            guard let destination = mapped else { continue }

            var v = value
            if destination.hasSuffix(".conv.conv.weight"), v.ndim == 3 {
                v = v.transposed(0, 2, 1)
            }
            out[destination] = v
        }

        return out
    }

    /// `acoustic_linear_1` / `acoustic_norm` / `acoustic_linear_2` name the same three pieces
    /// the nested layout calls `acoustic_connector.{fc1,norm,fc2}`.
    private static func mapProjector(_ suffix: String) -> String? {
        for family in ["acoustic", "semantic"] {
            guard suffix.hasPrefix("\(family)_") else { continue }
            let rest = String(suffix.dropFirst(family.count + 1))
            if rest.hasPrefix("linear_1.") {
                return "\(family)_connector.fc1." + String(rest.dropFirst("linear_1.".count))
            }
            if rest.hasPrefix("linear_2.") {
                return "\(family)_connector.fc2." + String(rest.dropFirst("linear_2.".count))
            }
            if rest.hasPrefix("norm.") {
                return "\(family)_connector.norm." + String(rest.dropFirst("norm.".count))
            }
        }
        return nil
    }

    /// Rewrites the flat encoder tree onto the canonical one.
    ///
    /// | flat | canonical |
    /// | --- | --- |
    /// | `stem.conv.conv.X` | `stem.conv.conv.X` |
    /// | `stem.stage.J.Y` | `stages.0.J.Y` |
    /// | `conv_layers.I.conv.conv.X` | `downsample_layers.I.conv.conv.X` |
    /// | `conv_layers.I.stage.J.Y` | `stages.I+1.J.Y` |
    /// | `head.conv.X` | `head.conv.conv.X` |
    /// | `...mixer.conv.X` | `...mixer.conv.conv.conv.X` |
    private static func mapFlatEncoder(_ key: String, prefix: String) -> String? {
        let base = String(key.dropFirst(prefix.count))
        let destination: String

        if base.hasPrefix("stem.stage.") {
            destination = "stages.0." + String(base.dropFirst("stem.stage.".count))
        } else if base.hasPrefix("stem.") {
            destination = base
        } else if base.hasPrefix("head.conv.") {
            destination = "head.conv.conv." + String(base.dropFirst("head.conv.".count))
        } else if base.hasPrefix("conv_layers.") {
            let rest = String(base.dropFirst("conv_layers.".count))
            let parts = rest.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, let index = Int(parts[0]) else { return nil }
            let tail = String(parts[1])
            if tail.hasPrefix("stage.") {
                destination = "stages.\(index + 1)." + String(tail.dropFirst("stage.".count))
            } else if tail.hasPrefix("conv.") {
                destination = "downsample_layers.\(index)." + tail
            } else {
                return nil
            }
        } else {
            return nil
        }

        // The flat layout collapses Convlayer -> SConv1d -> NormConv1d -> Conv1d into a
        // single `mixer.conv`, so the nesting has to be restored.
        let restored = destination.replacingOccurrences(
            of: ".mixer.conv.", with: ".mixer.conv.conv.conv.")
        return prefix + restored
    }
}
