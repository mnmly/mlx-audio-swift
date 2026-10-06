import Foundation
@preconcurrency import MLX
import MLXNN
import Testing

import MLXAudioCore
@testable import MLXAudioCodecs
@testable import MLXAudioSTT

/// Batched decoding left-pads prompts and hides the padding with a mask. If the mask let a
/// real token see padding, or RoPE's shifted positions changed attention, every batched
/// transcript would drift from the single-row one; this checks the language model directly,
/// on random weights, so it needs no checkpoint.
@Suite("VibeVoice ASR batched decoding")
struct VibeVoiceASRBatchTests {

    @Test func leftPaddedBatchMatchesSingleRows() {
        MLXRandom.seed(7)
        let config = VibeVoiceQwen2Configuration(
            hiddenSize: 64, numHiddenLayers: 2, intermediateSize: 128,
            numAttentionHeads: 4, numKeyValueHeads: 2, vocabSize: 97)
        let model = VibeVoiceQwen2Model(config)
        eval(model)

        let lengths = [9, 5, 7]
        let prompts = lengths.map { MLXRandom.normal([1, $0, config.hiddenSize]) }
        let decodeSteps = [MLXRandom.normal([1, 1, config.hiddenSize]),
                           MLXRandom.normal([1, 1, config.hiddenSize])]

        // Each row on its own: prompt, then two single-token steps.
        let single = prompts.map { prompt -> [MLXArray] in
            let cache = model.makeCache()
            var outs = [model(inputsEmbeds: prompt, cache: cache)[0..., -1, 0...]]
            for step in decodeSteps { outs.append(model(inputsEmbeds: step, cache: cache)[0..., -1, 0...]) }
            return outs
        }

        // The same rows left-padded into one batch, prefilled in two slices like
        // `generateBatch` does, then the same two steps. `hidePadding: false` builds the
        // control: identical except the padding is visible to the real tokens.
        let width = lengths.max()!
        let pads = lengths.map { width - $0 }
        let batch = concatenated(zip(prompts, pads).map { prompt, pad in
            pad == 0 ? prompt
                : concatenated([MLXArray.zeros([1, pad, config.hiddenSize]), prompt], axis: 1)
        }, axis: 0)
        func runBatch(hidePadding: Bool) -> [MLXArray] {
            let maskPads = hidePadding ? pads : pads.map { _ in 0 }
            func mask(_ start: Int, _ count: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
                VibeVoiceASRModel.leftPaddedCausalMask(padding: maskPads, queryStart: start, queryCount: count)
            }
            let cache = model.makeCache()
            let split = 4
            _ = model(inputsEmbeds: batch[0..., 0 ..< split, 0...], cache: cache, mask: mask(0, split))
            var outs = [model(inputsEmbeds: batch[0..., split..., 0...], cache: cache,
                              mask: mask(split, width - split))[0..., -1, 0...]]
            for (i, step) in decodeSteps.enumerated() {
                let rows = concatenated(Array(repeating: step, count: lengths.count), axis: 0)
                outs.append(model(inputsEmbeds: rows, cache: cache, mask: mask(width + i, 1))[0..., -1, 0...])
            }
            return outs
        }
        func maxDiff(_ outs: [MLXArray], rows: [Int]) -> Float {
            rows.flatMap { row in outs.indices.map { step in
                MLX.abs(outs[step][row] - single[row][step][0]).max().item(Float.self)
            } }.max()!
        }

        // Row 0 is unpadded, so its difference is pure float32 reordering (batched vs
        // single-row kernels): the noise floor. Padded rows must stay at that level, while
        // letting them see their padding must move them far beyond it.
        let masked = runBatch(hidePadding: true)
        let leaky = runBatch(hidePadding: false)
        let noise = maxDiff(masked, rows: [0])
        let padded = maxDiff(masked, rows: [1, 2])
        let leak = maxDiff(leaky, rows: [1, 2])
        print("VibeVoice batch mask: noise \(noise), padded rows \(padded), padding visible \(leak)")
        #expect(noise < 5e-3)
        #expect(padded < 4 * Swift.max(noise, 1e-6))
        #expect(leak > 50 * padded)
    }

    @Test func maskHidesPaddingButNeverAWholeRow() {
        let mask = VibeVoiceASRModel.leftPaddedCausalMask(padding: [0, 3], queryStart: 0, queryCount: 5)
            .mask!.asArray(Bool.self)
        // [rows 2][1][q 5][k 5], row-major
        func at(_ row: Int, _ q: Int, _ k: Int) -> Bool { mask[row * 25 + q * 5 + k] }
        #expect(at(0, 4, 0) && at(0, 4, 4) && !at(0, 2, 3))      // plain causal row
        #expect(!at(1, 4, 0) && !at(1, 4, 2) && at(1, 4, 3))     // padding hidden from real tokens
        #expect(at(1, 1, 1) && !at(1, 1, 0))                     // a padding query sees only itself
        for row in 0 ..< 2 { for q in 0 ..< 5 { #expect((0 ..< 5).contains { at(row, q, $0) }) } }
    }
}

/// `generateBatch` against `generate` on the real checkpoint, at float32.
///
/// ```
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_DIR=/path/to/VibeVoice-ASR-HF \
/// TEST_RUNNER_MLXAUDIO_VIBEVOICE_ASR_WAV=/path/to/speech-over-60s.wav \
/// xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
///   -only-testing:'MLXAudioTests/VibeVoiceASRBatchCheckpointTests'
/// ```
///
/// Batched matmuls sum in a different order than single-row ones, so a token the model is
/// almost evenly split on can in principle resolve differently; measured on eight 300 s
/// lecture chunks, batches of 2 and 4 matched exactly and 8 differed by one word in 5,833.
/// Two short clips at batch 2 are expected to match exactly.
@Suite("VibeVoice ASR batched decoding, checkpoint", .serialized)
struct VibeVoiceASRBatchCheckpointTests {

    @Test func generateBatchMatchesGenerate() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelDir = env["MLXAUDIO_VIBEVOICE_ASR_DIR"],
            let wavPath = env["MLXAUDIO_VIBEVOICE_ASR_WAV"]
        else {
            print("Skipping VibeVoice ASR batch test; set MLXAUDIO_VIBEVOICE_ASR_DIR "
                + "and MLXAUDIO_VIBEVOICE_ASR_WAV.")
            return
        }
        let (sampleRate, raw) = try loadAudioArray(from: URL(fileURLWithPath: wavPath))
        let audio = sampleRate == 24000 ? raw : try resampleAudio(raw, from: sampleRate, to: 24000)
        #expect(audio.size >= 60 * 24000, "needs at least 60 s of speech")

        // Two clips of different length, so the batch really is padded.
        let clips = [audio[0 ..< 40 * 24000], audio[40 * 24000 ..< 60 * 24000]]
        let model = try await VibeVoiceASRModel.fromModelDirectory(
            URL(fileURLWithPath: modelDir), precision: .float32)
        let parameters = STTGenerateParameters(maxTokens: 2048, temperature: 0)

        let single = clips.map { model.generate(audio: $0, generationParameters: parameters) }
        let batched = model.generateBatch(audios: clips, generationParameters: parameters)

        #expect(batched.count == clips.count)
        for (s, b) in zip(single, batched) {
            #expect(b.text == s.text)
            #expect(b.generationTokens == s.generationTokens)
            #expect((b.segments ?? []).count == (s.segments ?? []).count)
        }
    }
}
