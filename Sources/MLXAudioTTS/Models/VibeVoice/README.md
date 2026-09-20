# VibeVoice Realtime

Streaming, single-speaker text to speech from [microsoft/VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B).

Audio starts playing long before the text is exhausted: the model interleaves five text
tokens with six acoustic latents, decoding each latent to 24 kHz audio as soon as it is
sampled.

```swift
let model = try await TTS.loadModel(modelRepo: "mlx-community/VibeVoice-Realtime-0.5B")
for try await event in model.generateStream(text: "Hello there.", voice: "en-Carter_man") {
    if case .audio(let chunk) = event { player.enqueue(chunk) }
}
```

```
swift run mlx-audio-swift-tts --model mlx-community/VibeVoice-Realtime-0.5B \
    --voice en-Carter_man --benchmark "Hello there."
```

## How it works

- **Backbone** — Qwen2 0.5B (hidden 896, 24 layers, GQA 14/2) split in two. The lower 4
  layers encode text only; their last hidden state is spliced into the tail of the upper 20
  layers, which see both text and speech. A two-entry embedding marks each position as text
  or speech.
- **Acoustic tokenizer** — a causal convolutional σ-VAE at **7.5 Hz** (ratios
  `[8, 5, 5, 4, 2, 2]`, so 3200 samples per latent at 24 kHz), 64-dimensional latents. Only
  the decoder ships in this checkpoint.
- **Diffusion head** — 4 adaLN-modulated SwiGLU blocks predicting one latent at a time,
  sampled with DPM-Solver++ (multistep, order 2) under classifier-free guidance.
- **Stopping** — a binary classifier over the backbone's last hidden state.

Two knobs matter for the latency/quality trade-off:

| Property | Default | Notes |
|---|---|---|
| `diffusionSteps` | 20 (from `config.json`) | Upstream's own realtime demo drops this to 5 |
| `cfgScale` | 1.5 | Matches upstream's demo default |

## Voices

This model does **not** do voice cloning, by design — the checkpoint contains no acoustic
*encoder*, so there is no path from a recording to a speaker. Voices ship as prefilled
conversation state: a `last_hidden_state` plus a KV cache for each of the `lm`, `tts_lm` and
`neg_tts_lm` branches. (Upstream files carry a fourth branch, `neg_lm`, which is never read:
the negative branch is only ever fed speech embeddings, so it never reaches the text-only
stack.)

Upstream distributes these as `.pt` pickles in `demo/voices/streaming_model/`. Convert them
to safetensors and place them in `voices/` next to the weights:

```python
# flatten each branch to <branch>.last_hidden_state / <branch>.key.<layer> / <branch>.value.<layer>
prompt = torch.load(src, map_location="cpu", weights_only=True)
flat = {}
for branch in ["lm", "tts_lm", "neg_lm", "neg_tts_lm"]:
    part = prompt[branch]
    flat[f"{branch}.last_hidden_state"] = part["last_hidden_state"].contiguous()
    cache = part["past_key_values"]
    for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
        flat[f"{branch}.key.{i}"] = k.contiguous()
        flat[f"{branch}.value.{i}"] = v.contiguous()
save_file(flat, dst)
```

Loading `weights_only=True` needs `torch.serialization.add_safe_globals([BaseModelOutputWithPast, DynamicCache])`
on current PyTorch.

The expected model directory:

```
config.json
preprocessor_config.json
model.safetensors
tokenizer.json, tokenizer_config.json, vocab.json, merges.txt   # from Qwen/Qwen2.5-0.5B
voices/en-Carter_man.safetensors, ...
```

The tokenizer is not part of the upstream repo; `preprocessor_config.json` points at
`Qwen/Qwen2.5-0.5B`.

## Verifying against the reference

Generation samples fresh diffusion noise per latent, so a correct port does not reproduce
the reference waveform. `Tests/VibeVoiceParityTests.swift` makes the comparison
deterministic by replaying the exact noise the PyTorch reference drew:

```
TEST_RUNNER_MLXAUDIO_VIBEVOICE_MODEL_DIR=/path/to/model \
TEST_RUNNER_MLXAUDIO_VIBEVOICE_PARITY_FIXTURE=/path/to/parity.safetensors \
xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:'MLXAudioTests/VibeVoiceParityTests'
```

With shared noise the two implementations stop on the same frame and track each other at
about 26 dB SNR overall — 45 dB on the first latent, degrading as each latent feeds back
into the backbone. The floor is MLX's Metal kernels, which accumulate at roughly bf16
precision (a 256-wide `matmul` is ~7e-4 relative against PyTorch, `conv1d` ~1e-3, while the
same calls on the CPU backend agree to ~1e-7).
