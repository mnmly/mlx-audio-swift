import Foundation
import MLX
import MLXNN

/// VibeVoice's acoustic σ-VAE tokenizer: 24 kHz audio ⟷ 7.5 Hz, 64-dimensional latents.
///
/// The encoder emits the distribution *mean*; there is no learned variance. Upstream draws
/// a per-utterance scale (`std_dist_type: "gaussian"`, `fix_std / 0.8`) at inference, which
/// this port exposes through `sample(_:scale:)` but does not apply by default — see `encode`.
///
/// Tensors are NCL (`[batch, channels, time]`) throughout, matching the Mimi codec.
public final class VibeVoiceAcousticTokenizer: Module, AudioCodecModel {
    public typealias EncodedAudio = MLXArray

    @ModuleInfo(key: "encoder") public var encoder: VibeVoiceTokenizerEncoder
    @ModuleInfo(key: "decoder") public var decoder: VibeVoiceTokenizerDecoder

    public let config: VibeVoiceTokenizerConfiguration
    public var codecSampleRate: Double? { 24000 }

    /// Samples of audio per latent frame (3200 for the shipped checkpoints).
    public var compressionRatio: Int { config.compressionRatio }

    public init(config: VibeVoiceTokenizerConfiguration) {
        self.config = config
        self._encoder = ModuleInfo(wrappedValue: VibeVoiceTokenizerEncoder(
            config: config, dimension: config.vaeDim), key: "encoder")
        self._decoder = ModuleInfo(wrappedValue: VibeVoiceTokenizerDecoder(
            config: config, dimension: config.vaeDim), key: "decoder")
    }

    // MARK: - Encode / decode

    /// Audio `[B, 1, T]` to latent means `[B, vaeDim, ceil(T / compressionRatio)]`.
    ///
    /// Returns the mean, i.e. the distribution mode. Upstream samples here; sampling makes
    /// output non-deterministic for no quality gain at inference, so callers that want
    /// reference-matching behaviour opt in via `sample(_:scale:)`.
    public func encode(_ audioNCL: MLXArray) -> MLXArray {
        encoder(audioNCL)
    }

    /// Latents `[B, vaeDim, frames]` to audio `[B, 1, frames * compressionRatio]`.
    public func decode(_ latentsNCL: MLXArray) -> MLXArray {
        decoder(latentsNCL)
    }

    public func encodeAudio(_ waveform: MLXArray) -> MLXArray {
        encode(normalizedToNCL(waveform))
    }

    public func decodeAudio(_ input: MLXArray) -> MLXArray {
        decode(input)
    }

    /// One streaming step. Successive calls continue the convolution state, so a caller can
    /// feed one latent frame at a time and get `compressionRatio` new samples back.
    public func decodeStep(_ latentsNCL: MLXArray) -> MLXArray {
        decoder.step(latentsNCL)
    }

    /// One streaming encode step. `resetState()` must be called between independent streams.
    public func encodeStep(_ audioNCL: MLXArray) -> MLXArray {
        encoder.step(audioNCL)
    }

    public func resetState() {
        encoder.resetState()
        decoder.resetState()
    }

    /// Draw from the encoder distribution the way upstream does at inference.
    ///
    /// With `std_dist_type == "gaussian"` the scale is itself random, one draw per batch
    /// element: `std = randn(B) * (fix_std / 0.8)`, then `mean + std * randn_like(mean)`.
    public func sample(_ mean: MLXArray, scale: Float? = nil) -> MLXArray {
        let fixStd = scale ?? config.fixStd
        switch config.stdDistType {
        case "gaussian":
            let value = fixStd / 0.8
            var std = MLXRandom.normal([mean.shape[0]]).asType(mean.dtype) * value
            while std.ndim < mean.ndim { std = std.expandedDimensions(axis: -1) }
            return mean + std * MLXRandom.normal(mean.shape).asType(mean.dtype)
        default:
            return mean + fixStd * MLXRandom.normal(mean.shape).asType(mean.dtype)
        }
    }

    private func normalizedToNCL(_ waveform: MLXArray) -> MLXArray {
        switch waveform.ndim {
        case 1: return waveform.reshaped([1, 1, waveform.shape[0]])
        case 2: return waveform.expandedDimensions(axis: 1)
        default: return waveform
        }
    }

    // MARK: - Weight loading

    /// Maps upstream checkpoint keys onto this module tree.
    ///
    /// Upstream wraps every stem/downsample/upsample conv in a single-element `nn.Sequential`,
    /// so its keys carry an extra `.0`; the stem shares the list with the resampling convs.
    /// PyTorch conv weights also need transposing into MLX's channel-last kernel layout.
    public static func sanitize(weights: [String: MLXArray], prefix: String = "") -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (rawKey, value) in weights {
            var key = rawKey
            if !prefix.isEmpty {
                guard key.hasPrefix(prefix) else { continue }
                key = String(key.dropFirst(prefix.count))
            }

            // `norm` is an Identity when disable_last_norm is set, and fix_std is a
            // non-persistent buffer; neither should reach the module tree.
            if key.contains(".norm.weight"), key.hasSuffix("encoder.norm.weight") { continue }
            if key.hasSuffix("fix_std") { continue }

            guard let remapped = remapSequentialIndex(key) else { continue }
            key = remapped

            var v = value
            if key.hasSuffix(".conv.conv.weight"), v.ndim == 3 {
                // (out, in/groups, k) -> (out, k, in/groups)
                v = v.transposed(0, 2, 1)
            } else if key.hasSuffix(".convtr.convtr.weight"), v.ndim == 3 {
                // (in, out/groups, k) -> (out, k, in)
                v = v.transposed(1, 2, 0)
            }
            out[key] = v
        }

        return out
    }

    /// `downsample_layers.0.0.x` -> `stem.x`, `downsample_layers.N.0.x` -> `downsample_layers.N-1.x`
    /// (and the same for `upsample_layers`). Returns nil for keys that should be dropped.
    private static func remapSequentialIndex(_ key: String) -> String? {
        for (listName, isDecoder) in [("downsample_layers", false), ("upsample_layers", true)] {
            let marker = "\(listName)."
            guard let range = key.range(of: marker) else { continue }
            let head = String(key[key.startIndex ..< range.lowerBound])
            let tail = String(key[range.upperBound...])
            let parts = tail.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
            guard parts.count == 3, let index = Int(parts[0]), parts[1] == "0" else { return key }
            let rest = String(parts[2])
            _ = isDecoder
            if index == 0 {
                return "\(head)stem.\(rest)"
            }
            return "\(head)\(listName).\(index - 1).\(rest)"
        }
        return key
    }
}
