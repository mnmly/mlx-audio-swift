import Foundation

/// One diarized transcript segment.
public struct VibeVoiceTranscriptSegment: Sendable, Equatable {
    public var startTime: Double?
    public var endTime: Double?
    public var speakerID: String?
    public var text: String?
}

/// Prompt construction and transcript parsing for VibeVoice ASR.
///
/// The model is asked, in plain ChatML, to emit a JSON array with a fixed set of keys; the
/// diarization comes back as ordinary text in that JSON rather than through special tokens.
public enum VibeVoiceASRPrompt {
    public static let systemPrompt =
        "You are a helpful assistant that transcribes audio input into text output in JSON format."

    static let keys = ["Start time", "End time", "Speaker ID", "Content"]

    /// Builds the prompt text, with `audioPlaceholder` standing in for the run of audio
    /// tokens that the model's embeddings replace.
    ///
    /// Deliberately ends at `<|im_end|>` with **no** `<|im_start|>assistant` header:
    /// upstream threads an `add_generation_prompt` argument through the processor but never
    /// passes it to `apply_chat_template`, and the checkpoint's own `chat_template.jinja`
    /// matches that. Adding the header changes what the model generates.
    public static func build(
        audioPlaceholder: String,
        audioDuration: Double,
        contextInfo: String? = nil
    ) -> String {
        let (head, tail) = parts(audioDuration: audioDuration, contextInfo: contextInfo)
        return head + audioPlaceholder + tail
    }

    /// The prompt either side of the audio span.
    ///
    /// Splitting here lets the caller insert the audio token ids directly instead of
    /// embedding them in text and trusting the tokenizer to recognise them again. The seam
    /// sits against a special token, which always breaks a BPE sequence, so tokenizing the
    /// two halves separately matches tokenizing the whole.
    public static func parts(
        audioDuration: Double,
        contextInfo: String? = nil
    ) -> (head: String, tail: String) {
        let keyList = keys.joined(separator: ", ")
        let duration = String(format: "%.2f", audioDuration)

        let suffix: String
        if let contextInfo, !contextInfo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let trimmed = contextInfo.trimmingCharacters(in: .whitespacesAndNewlines)
            suffix = "This is a \(duration) seconds audio, with extra info: \(trimmed)\n\n"
                + "Please transcribe it with these keys: \(keyList)"
        } else {
            suffix = "This is a \(duration) seconds audio, please transcribe it with these "
                + "keys: \(keyList)"
        }

        let head = """
            <|im_start|>system
            \(systemPrompt)<|im_end|>
            <|im_start|>user

            """
        let tail = "\n\(suffix)<|im_end|>\n"
        return (head, tail)
    }

    /// The prompt the streaming path uses, which is raw text rather than ChatML.
    public static func buildStreaming(contextInfo: String? = nil) -> String {
        let base = "You are a helpful assistant that transcribes audio input into text output. "
        if let contextInfo, !contextInfo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return base
                + "Please transcribe the following audios streamingly with these keys: "
                + "speaker, content\nand extra info: "
                + contextInfo.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
        }
        return base
            + "Please transcribe the following audios streamingly with these keys: "
            + "speaker, content\n"
    }

    /// Parses a transcript into segments.
    ///
    /// The model emits one of two formats. Asked for the four keys, it normally returns a
    /// JSON array with timings. At reduced precision it often answers in a plain
    /// `speaker: text` form instead — the content is equivalent, but there are no timings —
    /// so that is parsed too rather than being dropped on the floor.
    public static func parse(_ text: String) -> [VibeVoiceTranscriptSegment] {
        guard let json = extractJSON(from: text),
            let data = json.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data)
        else { return parseSpeakerLines(text) }

        let items: [[String: Any]]
        if let array = parsed as? [[String: Any]] {
            items = array
        } else if let object = parsed as? [String: Any] {
            items = [object]
        } else {
            return parseSpeakerLines(text)
        }

        return items.compactMap { item in
            var segment = VibeVoiceTranscriptSegment()
            var matched = false

            for key in ["Start time", "Start"] where item[key] != nil {
                segment.startTime = number(item[key])
                matched = true
                break
            }
            for key in ["End time", "End"] where item[key] != nil {
                segment.endTime = number(item[key])
                matched = true
                break
            }
            for key in ["Speaker ID", "Speaker"] where item[key] != nil {
                segment.speakerID = string(item[key])
                matched = true
                break
            }
            if let content = item["Content"] {
                segment.text = string(content)
                matched = true
            }
            return matched ? segment : nil
        }
    }

    /// Parses the plain `0: text` / `Speaker 1: text` form.
    ///
    /// Lines without a speaker prefix (such as a bare `[Silence]`) are kept as text-only
    /// segments so nothing is silently lost.
    static func parseSpeakerLines(_ text: String) -> [VibeVoiceTranscriptSegment] {
        var segments: [VibeVoiceTranscriptSegment] = []

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }

            var segment = VibeVoiceTranscriptSegment()
            if let colon = line.firstIndex(of: ":") {
                let label = line[line.startIndex ..< colon].trimmingCharacters(in: .whitespaces)
                let body = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                // Only treat it as a speaker label if it looks like one; this must not
                // swallow ordinary prose that happens to contain a colon.
                let isSpeaker = !label.isEmpty && label.count <= 20
                    && (Int(label) != nil
                        || label.lowercased().hasPrefix("speaker"))
                if isSpeaker, !body.isEmpty {
                    segment.speakerID = label
                    segment.text = body
                    segments.append(segment)
                    continue
                }
            }
            segment.text = line
            segments.append(segment)
        }

        return segments
    }

    /// Flattens parsed segments into a plain transcript.
    public static func plainText(_ segments: [VibeVoiceTranscriptSegment]) -> String {
        segments.compactMap { segment in
            guard let text = segment.text?.trimmingCharacters(in: .whitespaces), !text.isEmpty
            else { return nil }
            if let speaker = segment.speakerID, !speaker.isEmpty {
                return "\(speaker): \(text)"
            }
            return text
        }.joined(separator: "\n")
    }

    private static func number(_ value: Any?) -> Double? {
        if let d = value as? Double { return d }
        if let i = value as? Int { return Double(i) }
        if let s = value as? String { return Double(s) }
        return nil
    }

    private static func string(_ value: Any?) -> String? {
        if let s = value as? String { return s }
        if let i = value as? Int { return String(i) }
        if let d = value as? Double { return String(d) }
        return nil
    }

    /// Mirrors upstream's brace matching: prefer a ```json fence, else the first bracket and
    /// its partner.
    private static func extractJSON(from text: String) -> String? {
        if let fence = text.range(of: "```json") {
            let rest = text[fence.upperBound...]
            if let close = rest.range(of: "```") {
                return String(rest[..<close.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return String(rest).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let characters = Array(text)
        guard let start = characters.firstIndex(where: { $0 == "[" || $0 == "{" }) else {
            return nil
        }

        var depth = 0
        for index in start ..< characters.count {
            if characters[index] == "[" || characters[index] == "{" {
                depth += 1
            } else if characters[index] == "]" || characters[index] == "}" {
                depth -= 1
                if depth == 0 {
                    return String(characters[start ... index])
                }
            }
        }
        // Unterminated (the model hit a token limit mid-array).
        return nil
    }
}
