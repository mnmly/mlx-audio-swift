import Foundation
import MLX
import MLXLMCommon

extension VibeVoiceASRModel {

    /// Attention mask for a batch whose rows are left-padded by `padding[row]` positions:
    /// causal, with each row's padding hidden from every query.
    ///
    /// A padding position may still see itself, so no softmax row is fully masked (which
    /// would give NaN); its output is never attended to by a real token. Shape
    /// `[rows, 1, queryCount, queryStart + queryCount]`, true where attention is allowed.
    static func leftPaddedCausalMask(
        padding: [Int], queryStart: Int, queryCount: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let keys = queryStart + queryCount
        let padStart = MLXArray(padding.map { Int32($0) }).reshaped([padding.count, 1, 1, 1])
        let k = MLXArray(Int32(0) ..< Int32(keys)).reshaped([1, 1, 1, keys])
        let q = MLXArray(Int32(queryStart) ..< Int32(keys)).reshaped([1, 1, queryCount, 1])
        return .array((k .<= q) .&& ((k .>= padStart) .|| (k .== q)))
    }

    /// Transcribes several recordings with one batched greedy decode.
    ///
    /// Decoding is bound by reading the weights once per step, not by arithmetic: at float32
    /// a step streams ~30 GB of language model for one token. A batch of B rows reads those
    /// weights once for B tokens, so throughput grows with B at the same precision and the
    /// same per-row math.
    ///
    /// Each recording is encoded and prompted exactly as `generate` does. The prompts are then
    /// left-padded to a common length so every row shares the cache position, and a mask hides
    /// the padding from the real tokens. RoPE scores depend only on relative position, so the
    /// shift padding introduces does not change attention.
    ///
    /// Exactness depends on the matmul precision, not on batching. On GPUs with NAX matrix
    /// units (M5), MLX runs float32 matmuls at TF32 unless `MLX_ENABLE_TF32=0`, and from MLX
    /// 0.32 that includes the M = 2…8 decode matmuls a batch makes, while single-row decode
    /// (M = 1) stays float32. With `MLX_ENABLE_TF32=0`, batches of 4 and 8 reproduced the
    /// single-row float32 transcripts exactly (eight 300 s lecture chunks); with TF32 on,
    /// batch 8 differed by one word in 5,833.
    ///
    /// Greedy without a repetition penalty only (what file transcription uses); rows that
    /// finish keep stepping with the batch and their extra tokens are discarded.
    public func generateBatch(
        audios: [MLXArray],
        generationParameters: STTGenerateParameters
    ) -> [STTOutput] {
        guard audios.count > 1 else {
            return audios.map { generate(audio: $0, generationParameters: generationParameters) }
        }
        precondition(
            generationParameters.temperature <= 0 && generationParameters.repetitionPenalty == 1.0,
            "generateBatch supports greedy decoding without a repetition penalty")

        let start = Date()
        let rows = audios.count

        // Per-row prompts, built as the single-row path builds them.
        var prompts: [MLXArray] = []
        for audio in audios {
            let features = encodeAudio(audio)
            let duration = Double(audio.size) / Double(config.samplingRate)
            let (embeds, _) = buildPromptEmbeddings(
                features: features, duration: duration,
                contextInfo: generationParameters.contextInfo)
            eval(embeds)
            prompts.append(embeds)
        }
        let lengths = prompts.map { $0.dim(1) }
        let width = lengths.max()!
        let pads = lengths.map { width - $0 }
        let hidden = prompts[0].dim(2)
        let batch = concatenated(
            zip(prompts, pads).map { prompt, pad in
                pad == 0
                    ? prompt
                    : concatenated(
                        [MLXArray.zeros([1, pad, hidden], dtype: prompt.dtype), prompt], axis: 1)
            }, axis: 0)
        func mask(from queryStart: Int, count: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
            Self.leftPaddedCausalMask(padding: pads, queryStart: queryStart, queryCount: count)
        }

        let cache = languageModel.makeCache()

        // Prefill in slices, as `prefill` does. The slice is divided by the batch so a
        // slice's matmul stays under the same M·N bound that keeps bfloat16 off the broken
        // NAX kernel (see `prefillStepSize`).
        let step = Swift.max(1, prefillStepSize / rows)
        var sliceStart = 0
        var last: MLXArray
        while true {
            let sliceEnd = Swift.min(sliceStart + step, width)
            last = languageModel(
                inputsEmbeds: batch[0..., sliceStart ..< sliceEnd, 0...], cache: cache,
                mask: mask(from: sliceStart, count: sliceEnd - sliceStart))
            if sliceEnd == width { break }
            eval(last)
            Memory.clearCache()
            sliceStart = sliceEnd
        }

        var next = MLX.argMax(lmHead(last[0..., -1, 0...]), axis: -1).asType(.int32)
        asyncEval(next)
        let prefillEnd = Date()

        let eos = tokenizer.eosTokenId ?? 151_643
        var pieces = [[Int]](repeating: [], count: rows)
        var done = [Bool](repeating: false, count: rows)
        var position = width
        var steps = 0

        // Same pipelining as the single-row loop: read step n, then queue step n+1.
        while true {
            let tokens = next.asArray(Int32.self)
            for row in 0 ..< rows where !done[row] {
                let token = Int(tokens[row])
                if token == eos {
                    done[row] = true
                    continue
                }
                pieces[row].append(token)
                if pieces[row].count >= generationParameters.maxTokens
                    || Self.isDegenerate(pieces[row], parameters: generationParameters)
                {
                    done[row] = true
                }
            }
            if !done.contains(false) { break }

            last = languageModel(
                inputsEmbeds: languageModel.embed(next.reshaped([rows, 1])), cache: cache,
                mask: mask(from: position, count: 1))
            position += 1
            next = MLX.argMax(lmHead(last[0..., -1, 0...]), axis: -1).asType(.int32)
            asyncEval(next)

            steps += 1
            if steps % 256 == 0 { Memory.clearCache() }
        }

        let elapsed = Date().timeIntervalSince(start)
        let prefill = prefillEnd.timeIntervalSince(start)
        return (0 ..< rows).map { row in
            makeOutput(
                text: tokenizer.decode(tokens: pieces[row]), promptLength: lengths[row],
                generated: pieces[row].count, prefill: prefill, elapsed: elapsed,
                parameters: generationParameters)
        }
    }
}
