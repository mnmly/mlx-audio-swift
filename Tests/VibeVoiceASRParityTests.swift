import Foundation
@preconcurrency import MLX
import Testing

import MLXAudioCore
@testable import MLXAudioSTT

/// Compares the Swift ASR port against the PyTorch reference on real weights.
///
/// Gated on environment variables because it needs the ~17 GB checkpoint:
///
/// ```
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_DIR=/path/to/VibeVoice-ASR-HF \
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_FIXTURE=/path/to/asr_parity.safetensors \
/// xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
///   -only-testing:'MLXAudioTests/VibeVoiceASRParityTests'
/// ```
///
/// The fixture is produced by `gen_asr_fixture.py` (see the model README) and holds the
/// resampled waveform alongside the reference's acoustic latents for it.
@Suite("VibeVoice ASR reference parity", .serialized)
struct VibeVoiceASRParityTests {

    /// The acoustic encoder is the whole audio front end minus the connectors: if dB-FS
    /// normalisation, causal padding, or the frame count were wrong, it would show here
    /// rather than as a subtly different transcript.
    @Test func acousticLatentsMatchReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MLXAUDIO_VIBEVOICE_ASR_DIR"],
            let fixturePath = env["MLXAUDIO_VIBEVOICE_ASR_FIXTURE"]
        else {
            print("Skipping VibeVoice ASR parity test; set MLXAUDIO_VIBEVOICE_ASR_DIR and "
                + "MLXAUDIO_VIBEVOICE_ASR_FIXTURE to run it.")
            return
        }

        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let audio = try #require(fixture["audio"]).asType(.float32)
        let expected = try #require(fixture["features"]).asType(.float32)

        let model = try await VibeVoiceASRModel.fromModelDirectory(
            URL(fileURLWithPath: modelDir))
        // The reference ran in float32. Set MLXAUDIO_VIBEVOICE_KEEP_BF16 to measure the
        // checkpoint's native precision instead, which is what the CLI actually runs.
        if env["MLXAUDIO_VIBEVOICE_KEEP_BF16"] == nil {
            model.update(parameters: model.parameters().mapValues { $0.asType(.float32) })
            eval(model)
        }

        let frames = Int(ceil(Double(audio.size) / Double(model.config.compressionRatio)))
        #expect(expected.dim(0) == frames)

        // Goes through the same preparation `encodeAudio` uses, so this measures the real
        // path rather than a shortcut around it.
        let prepared = model.preparedSamples(audio)
        let latents = swappedAxes(model.acousticLatents(prepared), 1, 2)
        let produced = latents.reshaped([-1, latents.dim(2)])

        print("VibeVoice ASR parity: latents \(produced.shape) vs reference \(expected.shape)")
        #expect(produced.dim(0) == expected.dim(0))
        #expect(produced.dim(1) == expected.dim(1))

        // Max-abs can be a single outlier; the error's *shape* is what distinguishes
        // accumulated kernel noise from a structural mistake.
        let scale = MLX.max(MLX.abs(expected)).item(Float.self)
        let diff = MLX.max(MLX.abs(produced - expected)).item(Float.self)
        let errorRMS = MLX.sqrt(MLX.mean((produced - expected) * (produced - expected)))
            .item(Float.self)
        let signalRMS = MLX.sqrt(MLX.mean(expected * expected)).item(Float.self)
        let snr = 20 * log10(signalRMS / Swift.max(errorRMS, 1e-20))

        // Correlation over the flattened latents.
        let a = produced.reshaped([-1])
        let b = expected.reshaped([-1])
        let am = a - MLX.mean(a)
        let bm = b - MLX.mean(b)
        let correlation = (MLX.sum(am * bm)
            / (MLX.sqrt(MLX.sum(am * am)) * MLX.sqrt(MLX.sum(bm * bm)))).item(Float.self)

        print("""
            VibeVoice ASR parity: peak \(scale), max abs diff \(diff), \
            relative \(diff / Swift.max(scale, 1e-9)), SNR \(String(format: "%.1f", snr)) dB, \
            correlation \(correlation)
            """)

        // Where the error sits matters: concentrated at a multiple of the 450-frame
        // segmentation boundary would mean the chunked encode is wrong, whereas spread
        // evenly is just accumulated kernel noise.
        let perFrameError = MLX.max(MLX.abs(produced - expected), axis: 1)
        let worstFrame = MLX.argMax(perFrameError, axis: 0).item(Int.self)
        let framesPerSegment = 1_440_000 / 3200
        print("""
            VibeVoice ASR parity: worst frame \(worstFrame) of \(produced.dim(0)) \
            (segment boundary every \(framesPerSegment) frames, \
            distance to nearest boundary \
            \(Swift.min(worstFrame % framesPerSegment, framesPerSegment - worstFrame % framesPerSegment)))
            """)

        // The final frame can cover only a handful of real samples when the audio length
        // is not a multiple of the 3200-sample hop; measure it separately from the rest.
        if produced.dim(0) > 1 {
            let head = produced[0 ..< (produced.dim(0) - 1), 0...]
            let headRef = expected[0 ..< (expected.dim(0) - 1), 0...]
            let e = MLX.sqrt(MLX.mean((head - headRef) * (head - headRef))).item(Float.self)
            let sig = MLX.sqrt(MLX.mean(headRef * headRef)).item(Float.self)
            print("VibeVoice ASR parity: excluding the final frame, SNR "
                + String(format: "%.1f", 20 * log10(sig / Swift.max(e, 1e-20))) + " dB")
        }

        // Deep convolution stacks accumulate MLX's reduced-precision Metal kernels, so this
        // is bounded on error energy rather than a single worst sample.
        #expect(correlation > 0.999)
        #expect(snr > 30)
    }

    /// Greedy decoding against the reference.
    ///
    /// Two different assertions, because the model's output mixes two kinds of content.
    /// The JSON scaffolding is discrete and must match token for token. The timestamps are
    /// continuous values predicted *as text*, so the ~1e-3 latent difference MLX's Metal
    /// kernels leave behind is enough to flip a digit — on 90 s of audio the second
    /// timestamp comes out `6.49` against the reference's `6.52`. Demanding an exact token
    /// stream there would be asserting that two float pipelines agree bit for bit. What has
    /// to match is the transcribed *words*.
    @Test func greedyDecodeMatchesReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MLXAUDIO_VIBEVOICE_ASR_DIR"],
            let fixturePath = env["MLXAUDIO_VIBEVOICE_ASR_FIXTURE"]
        else { return }

        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let audio = try #require(fixture["audio"]).asType(.float32)
        let referenceIDs = try #require(fixture["generated_ids"]).asArray(Int32.self).map(Int.init)

        let model = try await VibeVoiceASRModel.fromModelDirectory(
            URL(fileURLWithPath: modelDir))
        model.update(parameters: model.parameters().mapValues { $0.asType(.float32) })
        eval(model)

        var produced: [Int] = []
        let output = model.transcribe(
            audio: audio,
            generationParameters: STTGenerateParameters(maxTokens: 2048, temperature: 0),
            onToken: { _ in },
            onTokenID: { produced.append($0) })

        // The opening is `<|im_start|>assistant\n[{"Start"` — discrete structure that
        // depends on the prompt, the spliced features and the decode loop all being right.
        let prefix = 8
        let matchingPrefix = zip(produced.prefix(prefix), referenceIDs.prefix(prefix))
            .prefix { $0 == $1 }.count
        print("VibeVoice ASR parity: structural prefix \(matchingPrefix)/\(prefix) ids match")
        #expect(matchingPrefix == prefix)

        let referenceText = model.tokenizer.decode(tokens: referenceIDs)
        let referenceSegments = VibeVoiceASRPrompt.parse(referenceText)
        let producedSegments = VibeVoiceASRPrompt.parse(model.tokenizer.decode(tokens: produced))

        let referenceWords = referenceSegments.compactMap(\.text).joined(separator: " ")
        let producedWords = producedSegments.compactMap(\.text).joined(separator: " ")

        print("""
            VibeVoice ASR parity: \(producedSegments.count) segments vs \
            \(referenceSegments.count) reference
              swift     \(producedWords.prefix(90).debugDescription)
              reference \(referenceWords.prefix(90).debugDescription)
            """)

        #expect(!referenceSegments.isEmpty)
        #expect(producedSegments.count == referenceSegments.count)
        #expect(producedWords == referenceWords)
        #expect(!output.text.isEmpty)
    }
}

/// Exercises the real-time streaming path, which needs its own checkpoint
/// (`microsoft/VibeVoice-ASR-Streaming-7B`) and the nested weight layout.
///
/// ```
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_STREAMING_DIR=/path/to/VibeVoice-ASR-Streaming-7B \
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_AUDIO=/path/to/audio.wav \
/// xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
///   -only-testing:'MLXAudioTests/VibeVoiceASRStreamingTests'
/// ```
@Suite("VibeVoice ASR streaming", .serialized)
struct VibeVoiceASRStreamingTests {

    @Test func transcribesIncrementally() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MLXAUDIO_VIBEVOICE_ASR_STREAMING_DIR"],
            let audioPath = env["MLXAUDIO_VIBEVOICE_ASR_AUDIO"]
        else {
            print("Skipping VibeVoice ASR streaming test; set "
                + "MLXAUDIO_VIBEVOICE_ASR_STREAMING_DIR and MLXAUDIO_VIBEVOICE_ASR_AUDIO.")
            return
        }

        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: audioPath))
        let audio = try #require(fixture["audio"]).asType(.float32)

        let model = try await VibeVoiceASRModel.fromModelDirectory(
            URL(fileURLWithPath: modelDir))

        // This checkpoint is the nested layout, so loading it at all exercises that mapping.
        #expect(model.config.layout == .nested)
        // And it declares its own geometry, which must override the defaults.
        #expect(model.streamingChunkFrames == 22)
        #expect(model.streamingLookaheadFrames == 4)
        // The streaming checkpoint is trained on un-normalised audio.
        #expect(model.normalizeAudio == false)

        let session = VibeVoiceASRStreamSession(model: model)
        #expect(session.chunkSamples == 22 * 3200)
        #expect(session.lookaheadSamples == 4 * 3200)

        // Feed the audio the way a microphone would: in small, uneven pieces.
        let samples = audio.asArray(Float.self)
        var chunks: [VibeVoiceASRStreamSession.Chunk] = []
        var offset = 0
        let feed = 24000  // one second at a time
        while offset < samples.count {
            let end = Swift.min(offset + feed, samples.count)
            chunks.append(contentsOf: session.append(MLXArray(Array(samples[offset ..< end]))))
            offset = end
        }
        chunks.append(contentsOf: session.finish())

        let transcript = chunks.map(\.text).joined(separator: " ")
        print("""
            VibeVoice ASR streaming: \(chunks.count) chunks over \
            \(String(format: "%.1f", Double(samples.count) / 24000))s
              \(transcript.prefix(300).debugDescription)
            """)

        // Chunks must advance by the trained stride, and cover the audio.
        let expectedChunks = Int(
            ceil(Double(samples.count) / Double(session.chunkSamples)))
        #expect(chunks.count == expectedChunks)
        #expect(zip(chunks, chunks.dropFirst()).allSatisfy { $0.startTime < $1.startTime })
        #expect(!transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}

/// Guards the failure mode that long audio exposed: greedy decoding collapsing into a
/// repeated phrase instead of transcribing.
///
/// ```
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_DIR=/path/to/VibeVoice-ASR-HF \
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_WAV=/path/to/long.wav \
/// xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
///   -only-testing:'MLXAudioTests/VibeVoiceASRLongAudioTests'
/// ```
@Suite("VibeVoice ASR long audio", .serialized)
struct VibeVoiceASRLongAudioTests {

    @Test func longAudioTranscribesWithoutDegenerating() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MLXAUDIO_VIBEVOICE_ASR_DIR"],
            let wavPath = env["MLXAUDIO_VIBEVOICE_ASR_WAV"]
        else {
            print("Skipping VibeVoice ASR long-audio test; set MLXAUDIO_VIBEVOICE_ASR_DIR "
                + "and MLXAUDIO_VIBEVOICE_ASR_WAV.")
            return
        }

        let (sampleRate, raw) = try loadAudioArray(from: URL(fileURLWithPath: wavPath))
        let audio = sampleRate == 24000 ? raw : try resampleAudio(raw, from: sampleRate, to: 24000)
        let duration = Double(audio.size) / 24000
        #expect(duration > 150, "This test is about audio long enough to trigger the collapse")

        // At the checkpoint's own precision: the collapse this guards against came from a
        // half-precision-only GEMM defect, so a float32 run here would prove nothing.
        let model = try await VibeVoiceASRModel.fromModelDirectory(
            URL(fileURLWithPath: modelDir))

        let output = model.generate(
            audio: audio,
            generationParameters: STTGenerateParameters(maxTokens: 8192, temperature: 0))

        let segments = output.segments ?? []
        print("""
            VibeVoice ASR long audio: \(String(format: "%.0f", duration))s -> \
            \(segments.count) segments, \(output.generationTokens) tokens
              \(output.text.prefix(120).debugDescription)
            """)

        // A collapse yields one runaway "segment" of repeated text, or none at all.
        #expect(segments.count > 3)
        #expect(output.generationTokens > 100)

        // No sentence should repeat more than a couple of times; the failure looked like
        // "I mean, I mean, I mean" for thousands of tokens.
        let sentences = output.text
            .split(whereSeparator: { ".!?\n".contains($0) })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.count > 8 }
        let counts = Dictionary(grouping: sentences, by: { $0 }).mapValues(\.count)
        let worst = counts.max { $0.value < $1.value }
        #expect((worst?.value ?? 0) <= 3, "Repeated \(worst?.value ?? 0)x: \(worst?.key ?? "")")
    }
}
