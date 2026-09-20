# VibeVoice ASR

Transcription with speaker diarization, from Microsoft's VibeVoice ASR checkpoints. The
model returns a JSON array of segments — start time, end time, speaker, text — which this
port parses into `STTOutput.segments`.

```swift
let model = try await STT.loadModel(modelRepo: "microsoft/VibeVoice-ASR-HF")
let output = model.generate(audio: samples)   // mono, 24 kHz
print(output.text)                            // "Speaker 1: …\nSpeaker 2: …"
```

## How it works

Audio runs through **two** σ-VAE encoders at 7.5 Hz — acoustic (64-dim) and semantic
(128-dim) — each with its own connector (`fc1 → RMSNorm → fc2`). The two projections are
**summed**, not concatenated, and spliced into a ChatML prompt over a contiguous run of
`<|box_start|>` placeholders, one per `ceil(samples / 3200)`. A Qwen2.5-7B decoder then
generates the transcript.

Two details that are easy to get wrong:

- The prompt ends at `<|im_end|>` with **no** `<|im_start|>assistant` header. Upstream
  threads an `add_generation_prompt` argument through its processor but never passes it to
  `apply_chat_template`, and the shipped `chat_template.jinja` agrees. Adding the header
  changes what the model generates.
- The acoustic branch *samples* its VAE upstream (two `randn` draws per encode), making
  transcripts non-deterministic. This port takes the mean by default, as upstream's own vLLM
  plugin does via `VIBEVOICE_USE_MEAN`; set `sampleAcousticLatents = true` to match the
  reference exactly.

`STTOutput.segments` carries `start`/`end` (what this package's CLI and its SRT/VTT writers
read) alongside `start_time`/`end_time` (what upstream's post-processing returns), plus
`speaker_id` where the model assigned one. `STTOutput.text` is the flattened
`speaker: text` transcript.

```
$ mlx-audio-swift-stt --model microsoft/VibeVoice-ASR-HF --audio two-people.wav \
    --output-path out --format json
  [ 0.00- 6.46]  speaker -    [Silence]
  [ 6.46- 7.22]  speaker 0    Hello.
  [ 7.34- 8.41]  speaker 1    Hello.
  [ 8.06-10.04]  speaker 0    Oh, hello, I didn't know you were there.
```

Note that SRT and VTT have no way to express a speaker, so those formats carry the text and
timings only; use `--format json` to keep the attribution.

Audio longer than 60 s is encoded in segments (`acoustic_tokenizer_chunk_size`), carrying
convolution state across them, which is upstream's workaround for the convolution stack's
32-bit indexing. Input is zero-padded to a whole number of 3200-sample frames first: the
offline convolution path ceil-pads internally but the segmented one does not, and a final
frame holding only a handful of real samples came out badly wrong without it.

## Precision, and long recordings

The checkpoint is bfloat16 and this port runs it at that precision by default, on recordings
of any length. Earlier versions collapsed into a repetition loop (`"I mean, I mean, I
mean…"`) on audio past about two minutes at bfloat16, and only float32 was reliable. That was
not a precision problem: mlx-swift 0.31.x JIT-compiles the "NAX" split-K GEMM that M5-class
GPUs use for a matmul once `M·N ≥ 2048²`, `K ≥ 10240` and `K ≥ 3·max(M, N)` — this model's
`down_proj` (K = 18944, N = 3584) on any prompt of ~1171 tokens or more — with the float32
accumulator's type in place of the input's, so it read bfloat16 activations as float32 and
returned garbage. Fixed upstream in MLX 0.32.0 (ml-explore/mlx#3810), which no mlx-swift
release bundles yet. The prompt is therefore prefilled in slices whose `M·N` stays under
2048² (1024 tokens for this hidden size), which never reaches that kernel and, as in mlx-lm,
also keeps prefill memory flat. `precision: .float32` remains available for comparisons
against the reference, at twice the memory:

```swift
let model = try await VibeVoiceASRModel.fromPretrained(
    "microsoft/VibeVoice-ASR-HF", precision: .float32)
```

The audio encoders are always float32 regardless, because they are ~33 convolutions deep and
bfloat16 accumulates through them badly — 40.7 dB SNR against the reference versus 68.2 dB.
The two encoders are a small part of an 8B model, so this costs about a gigabyte.

Two safeguards apply either way: `repetitionPenalty` / `repetitionContextSize` from
`STTGenerateParameters` are honoured, and for greedy callers (penalty 1.0, which has no way
to escape a loop) generation stops once the last 24 tokens contain almost no distinct
values, rather than spending the whole token budget repeating a phrase.

Generation cost grows with context, so one pass over hours of audio is far slower than
transcribing it in parts.

## Verifying against the reference

`Tests/VibeVoiceASRParityTests.swift` compares against PyTorch on real weights, gated behind
two environment variables. Set `vae_std = 0` on the reference first — the acoustic VAE
samples at inference, so without that its own latents differ run to run by ~2.6% and the
comparison measures noise rather than correctness. The fixture (`gen_asr_fixture.py` in the
test's doc comment) is a safetensors file holding `audio` (24 kHz float32), `features` (the
reference's acoustic latents, `[frames, 64]`) and `generated_ids`, produced with
`transformers`' native `VibeVoiceAsrForConditionalGeneration` in float32:

```python
model.config.acoustic_tokenizer_encoder_config.vae_std = 0.0
inputs = processor.apply_transcription_request(audio=wav)
latents = model.get_audio_features(input_values=inputs["input_values"]).last_hidden_state
ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)[0, inputs["input_ids"].shape[1]:]
```

With sampling disabled the acoustic latents match at **66 dB SNR, correlation 1.0** — both
for a 30 s clip and for a 90 s one, which crosses the 60 s segmentation threshold and comes
out at the expected 675 frames.

Two different assertions cover decoding, because the output mixes two kinds of content. The
JSON scaffolding is discrete and matches the reference token for token. The timestamps are
continuous values predicted *as text*, so the ~1e-3 difference MLX's Metal kernels leave in
the latents is enough to flip a digit — over 90 s the second timestamp comes out `6.49`
against the reference's `6.52`. Requiring an exact token stream there would be asserting
that two float pipelines agree bit for bit, so what is checked is the transcribed *words*,
which match exactly (30 segments on both sides).

## Checkpoints

Both weight layouts load; `config.json` selects between them.

| Repo | Layout | Notes |
|---|---|---|
| `microsoft/VibeVoice-ASR-HF` | flat | Recommended. 8.33B params, bundles a tokenizer and chat template, acoustic decoder already stripped |
| `microsoft/VibeVoice-ASR` | nested | 8.67B. No tokenizer — copy the files from `Qwen/Qwen2.5-7B`. Carries ~344M params of unused acoustic decoder, which this port skips |
| `microsoft/VibeVoice-ASR-Streaming-7B` | nested | The only streaming-capable checkpoint |

The flat layout is not a rename of the nested one: it regroups each downsampling convolution
with the stage that follows it (`conv_layers.I.stage.J` against `stages.I+1.J`) and drops a
level of wrapping around the head and mixer convolutions. `VibeVoiceASRWeights.swift` maps
both onto one module tree.

At bf16 the weights are ~17 GB, so this wants a 32 GB machine or better. For long recordings
the KV cache dominates — 28 layers × 4 KV heads × 128 dims is roughly 57 KB per token, so an
hour of audio is ~1.6 GB before any generated text. `STTGenerateParameters` carries
`kvBits` / `kvGroupSize` / `quantizedKVStart` for that.

## Streaming

`VibeVoiceASRStreamSession` transcribes audio that is still arriving. It is **not** a chunked
audio encoder: the encoder runs fresh on each overlapping window, and what persists is the
language model's KV cache.

```swift
let session = VibeVoiceASRStreamSession(model: model)
for chunk in session.append(newSamples) { print(chunk.text) }
for chunk in session.finish() { print(chunk.text) }
```

Each window is `chunk_frames` of advance plus `lookahead_frames` of future context (2.93 s
and 0.53 s for the shipped checkpoint), decoded until the model emits `<|text_chunk_end|>`.
That separator is then appended to the cache **whether or not the model produced it**, which
is what keeps successive chunks aligned. A checkpoint only works at the geometry it was
trained on, so both values are read from its `preprocessor_config.json` — which is also
where `normalize_audio: false` comes from, since the streaming model is trained on
un-normalised audio.

Verified against `microsoft/VibeVoice-ASR-Streaming-7B`: feeding 30 s of audio one second at
a time yields 11 chunks (30 / 2.93) of coherent, speaker-labelled text. That checkpoint uses
the nested weight layout, so it exercises that mapping end to end.
