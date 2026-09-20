import Foundation
@preconcurrency import MLX
import MLXLMCommon

/// A prefilled voice prompt: the conversation state a speaker's reference audio leaves
/// behind, rather than the audio itself.
///
/// VibeVoice-Realtime ships voices as prefilled KV caches — the realtime checkpoint has no
/// acoustic *encoder* at all, so there is deliberately no way to build one of these from a
/// new recording. Each file carries four branches; `negativeLanguageModel` is stored by
/// upstream but never read during generation, because the negative branch is only ever fed
/// speech embeddings and so never reaches the lower, text-only stack.
public struct VibeVoiceVoicePrompt: Sendable {
    public struct Branch: Sendable {
        public var lastHiddenState: MLXArray
        public var keys: [MLXArray]
        public var values: [MLXArray]

        /// Positions already in the cache.
        public var length: Int { keys.first?.dim(2) ?? 0 }
    }

    public var languageModel: Branch
    public var ttsLanguageModel: Branch
    public var negativeTTSLanguageModel: Branch

    public enum Error: Swift.Error, LocalizedError {
        case missingTensor(String)
        case noVoices(URL)
        case unknownVoice(name: String, available: [String])

        public var errorDescription: String? {
            switch self {
            case .missingTensor(let key):
                return "Voice prompt is missing '\(key)'"
            case .noVoices(let url):
                return "No voice prompts (*.safetensors) found in \(url.path)"
            case .unknownVoice(let name, let available):
                return "Unknown voice '\(name)'. Available: \(available.joined(separator: ", "))"
            }
        }
    }

    /// Loads a converted voice prompt. See `Sources/MLXAudioTTS/Models/VibeVoice/README.md`
    /// for the layout and the conversion step from upstream's `.pt` files.
    public static func load(from url: URL) throws -> VibeVoiceVoicePrompt {
        let weights = try MLX.loadArrays(url: url)

        func branch(_ name: String) throws -> Branch {
            let hiddenKey = "\(name).last_hidden_state"
            guard let hidden = weights[hiddenKey] else { throw Error.missingTensor(hiddenKey) }

            var keys: [MLXArray] = []
            var values: [MLXArray] = []
            var layer = 0
            while let k = weights["\(name).key.\(layer)"], let v = weights["\(name).value.\(layer)"] {
                keys.append(k)
                values.append(v)
                layer += 1
            }
            guard !keys.isEmpty else { throw Error.missingTensor("\(name).key.0") }
            return Branch(lastHiddenState: hidden, keys: keys, values: values)
        }

        return VibeVoiceVoicePrompt(
            languageModel: try branch("lm"),
            ttsLanguageModel: try branch("tts_lm"),
            negativeTTSLanguageModel: try branch("neg_tts_lm")
        )
    }

    /// Resolves a voice by name within a directory, accepting either a bare name
    /// (`en-Carter_man`) or a full filename.
    public static func resolve(name: String?, in directory: URL) throws -> URL {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []

        guard !files.isEmpty else { throw Error.noVoices(directory) }

        guard let name, !name.isEmpty else { return files[0] }

        if let match = files.first(where: {
            $0.deletingPathExtension().lastPathComponent == name || $0.lastPathComponent == name
        }) {
            return match
        }
        throw Error.unknownVoice(
            name: name,
            available: files.map { $0.deletingPathExtension().lastPathComponent })
    }

    /// Builds caches seeded with this branch's prefilled keys and values.
    ///
    /// `KVCacheSimple.update` appends and advances the offset, so a single call per layer
    /// leaves each cache holding the prompt with its position counter already correct.
    public static func makeSeededCache(_ branch: Branch, dtype: DType) -> [KVCache] {
        zip(branch.keys, branch.values).map { k, v in
            let cache = KVCacheSimple()
            _ = cache.update(keys: k.asType(dtype), values: v.asType(dtype))
            return cache
        }
    }
}
