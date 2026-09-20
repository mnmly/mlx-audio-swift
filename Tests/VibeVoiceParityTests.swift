import Foundation
@preconcurrency import MLX
import MLXLMCommon
import Testing

@testable import MLXAudioTTS

/// End-to-end comparison of the Swift model against the PyTorch reference on real weights.
///
/// Generation draws fresh diffusion noise per latent, so two correct implementations produce
/// different waveforms. The fixture records the exact noise the reference drew along with the
/// waveform it produced; replaying that noise makes the comparison deterministic.
///
/// Gated on environment variables because it needs the 2 GB checkpoint:
///
/// ```
/// MLXAUDIO_VIBEVOICE_MODEL_DIR=/path/to/vibevoice-realtime \
/// MLXAUDIO_VIBEVOICE_PARITY_FIXTURE=/path/to/parity.safetensors \
/// xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
///   -only-testing:'MLXAudioTests/VibeVoiceParityTests'
/// ```
///
/// The fixture is produced by `gen_parity_fixture.py` (see the model README).
@Suite("VibeVoice reference parity", .serialized)
struct VibeVoiceParityTests {

    @Test func matchesPyTorchReferenceWithSharedNoise() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MLXAUDIO_VIBEVOICE_MODEL_DIR"],
            let fixturePath = env["MLXAUDIO_VIBEVOICE_PARITY_FIXTURE"]
        else {
            print("Skipping VibeVoice parity test; set MLXAUDIO_VIBEVOICE_MODEL_DIR and "
                + "MLXAUDIO_VIBEVOICE_PARITY_FIXTURE to run it.")
            return
        }

        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let expected = try #require(fixture["audio"]).asType(.float32)

        var noise: [MLXArray] = []
        var index = 0
        while let n = fixture["noise.\(index)"] {
            noise.append(n.asType(.float32))
            index += 1
        }
        #expect(!noise.isEmpty)

        let model = try await VibeVoiceModel.fromModelDirectory(
            URL(fileURLWithPath: modelDir))

        // The reference ran in float32; the checkpoint on disk is bfloat16, and bf16's
        // ~3 significant digits would swamp any real discrepancy.
        model.update(parameters: model.parameters().mapValues { $0.asType(.float32) })
        eval(model)

        var next = 0
        model.noiseProvider = {
            defer { next += 1 }
            return noise[Swift.min(next, noise.count - 1)]
        }

        let audio = try await model.generate(
            text: "Hello from MLX Swift.",
            voice: "en-Carter_man",
            refAudio: nil,
            refText: nil,
            language: nil,
            generationParameters: model.defaultGenerationParameters)

        let produced = audio.asArray(Float.self)
        let reference = expected.asArray(Float.self)

        let ratio = Double(produced.count) / Double(reference.count)
        print("""
            VibeVoice parity: produced \(produced.count) samples \
            (\(String(format: "%.2f", Double(produced.count) / 24000))s), \
            reference \(reference.count) (\(String(format: "%.2f", Double(reference.count) / 24000))s), \
            ratio \(String(format: "%.3f", ratio))
            """)

        // Same noise and same weights must mean the EOS head fires at the same frame.
        #expect(produced.count == reference.count)

        let n = Swift.min(produced.count, reference.count)
        guard n > 0 else { return }

        var peak: Float = 0
        var worst: Float = 0
        var sumSquaredError: Double = 0
        var sumSquaredRef: Double = 0
        for i in 0 ..< n {
            peak = Swift.max(peak, Swift.abs(reference[i]))
            worst = Swift.max(worst, Swift.abs(produced[i] - reference[i]))
            let d = Double(produced[i] - reference[i])
            sumSquaredError += d * d
            sumSquaredRef += Double(reference[i]) * Double(reference[i])
        }
        let snr = 10 * log10(sumSquaredRef / Swift.max(sumSquaredError, 1e-20))
        print("VibeVoice parity: peak \(peak), max abs diff \(worst), SNR \(String(format: "%.1f", snr)) dB")

        // Per-frame SNR separates a structural error (bad from frame 0) from ordinary
        // numerical drift (good early, degrading as latents feed back into the backbone).
        let frameSize = 3200
        var perFrame: [String] = []
        for frame in 0 ..< (n / frameSize) {
            var err: Double = 0
            var ref: Double = 0
            for i in (frame * frameSize) ..< ((frame + 1) * frameSize) {
                let d = Double(produced[i] - reference[i])
                err += d * d
                ref += Double(reference[i]) * Double(reference[i])
            }
            perFrame.append(String(format: "%.0f", 10 * log10(ref / Swift.max(err, 1e-20))))
        }
        print("VibeVoice parity: per-frame SNR (dB) \(perFrame.joined(separator: " "))")

        // Twenty chained diffusion steps per latent, each latent feeding back into the
        // backbone, on top of MLX's reduced-precision Metal convolutions: the waveforms
        // track the reference closely but are not bit-identical.
        #expect(snr > 20)
    }
}
