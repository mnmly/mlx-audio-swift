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
`speaker_id`. `STTOutput.text` is the flattened `speaker: text` transcript.

Audio longer than 60 s is encoded in segments (`acoustic_tokenizer_chunk_size`), carrying
convolution state across them, which is upstream's workaround for the convolution stack's
32-bit indexing.

## Verifying against the reference

`Tests/VibeVoiceASRParityTests.swift` compares against PyTorch on real weights, gated behind
two environment variables. Set `vae_std = 0` on the reference first — the acoustic VAE
samples at inference, so without that its own latents differ run to run by ~2.6% and the
comparison measures noise rather than correctness.

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
