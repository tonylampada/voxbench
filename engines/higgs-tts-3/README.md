# Higgs TTS 3 engine

Wraps `bosonai/higgs-tts-3-4b` (4B autoregressive decoder, 102 languages,
Boson Research & Non-Commercial licence, released 2026-07) behind the shared
bench contract. English and Portuguese both sit in the model's sub-5% WER/CER
production tier; the model card's Elo places it near the top of open weights.

Served by **SGLang-Omni** — the stack Boson documents for this model.
`server.py` launches `sgl-omni serve` (OpenAI-compatible `/v1/audio/speech`)
as a subprocess and adapts the contract on 8877.

## Build (from the repo root)

```
docker build -f engines/higgs-tts-3/Dockerfile -t tts-higgs-tts-3:gpu .
```

Base image is `lmsysorg/sglang-omni:dev` (multi-GB pull; the layer pull, not
the build, is what takes the time).

## Run

```
engines/tts up higgs-tts-3
```

The registry entry uses extra-mount `voices+shm`: the repo `voices/` clips are
mounted at `/app/voices` for zero-shot cloning, and the container gets
`--shm-size 32g` (vendor requirement for the SGLang stack; the 64 MB docker
default starves torch IPC).

## What it loads

`bosonai/higgs-tts-3-4b` from the mounted host HF cache (~9.3 GB safetensors,
public, ungated). Nothing is baked into the image. Sample rate 24 kHz, 8
codebooks at 25 fps, Higgs tokenizer + vocoder bundled in the serving stack.

## Contract notes

- **Language**: Higgs has no `lang` parameter — the model infers language from
  the text itself. The adapter accepts `en` and `pt` (both top-tier for this
  model) and returns 400 for anything else.
- `/voices`: `"default"` (the model's builtin voice) plus every clip mounted
  at `/app/voices` — those are used as zero-shot cloning references
  (`references[].audio_path`). A sibling `.txt` transcript is passed along
  when present (improves cloning fidelity; the repo clips have none).
- `/tts/stream` is **native streaming**: upstream emits raw 16-bit mono PCM
  as the vocoder produces them (`stream: true, response_format: "pcm"`); the
  adapter wraps chunks (coalesced to ≥0.2 s) into framed WAVs. `chunk_size`
  is accepted but ignored — it is a knob for sentence-chunked fake streaming.
- Optional body fields: `temperature`, `top_k`, `max_new_tokens`.
- Inline control tokens go straight in `text`: `<|emotion:amusement|>`,
  `<|style:whispering|>`, `<|prosody:pause|>`, `<|sfx:laughter|>Haha` — see
  the model card's PROMPTING.md.

## Pins and why

- Base `lmsysorg/sglang-omni:dev` (44.5 GB extracted, 18 GB pull). No package
  is installed by the build — the stack is already in the image, but **only**
  inside the `/opt/omni` python3.11 venv (`/opt/omni/bin/sgl-omni`); the
  system `python3` has neither `sgl_omni` nor the CLI, and the model card's
  "clone and pip install" step is unnecessary. Both the adapter and the
  upstream launch therefore use the venv's binaries.
- `sgl-omni serve` has **no `--context-length` flag** (the sibling trick of
  capping declared context does not apply here). The memory headroom knob is
  `--mem-fraction-static`, exposed as env `HIGGS_MEM_FRACTION` — unset by
  default so SGLang auto-sizes.

## Measurements

Observed on this box (RTX 4090 24 GB, nvdrv workaround, weights pre-cached):

- **Boot to healthy: ~71 s** from container start (model load; no downloads).
- **VRAM: 21.5 / 24.0 GiB** — SGLang preallocates its KV pool from free VRAM,
  so the card reads full regardless of load. This is the auto-picked
  mem fraction; `HIGGS_MEM_FRACTION` shrinks it if something must coexist.
  Container RSS 3.7 GiB.
- **Warm `/tts`: RTF ≈ 0.26** — 5.9 s of audio in 1.5 s wall. Cold first call
  after boot ~5 s.
- **`/tts/stream` warm: first frame ≈ 1.5 s**, then a continuous trickle;
  22 s of audio completed in 5.4 s wall (RTF 0.24).
- **Duration stability**: same English sentence 4× → 5.0 / 5.6 / 6.9 / 5.9 s.
  Sampling variance (default temperature), never silence or truncation.

## Warts

- **Short sentences "stream" as one burst**: anything the model finishes
  inside the ~1.5 s first-audio window (i.e. most bench sentences) arrives as
  all frames at once. Genuine incremental delivery shows on longer inputs.
- **Degenerate repetitive input can loop**: a test string with one sentence
  repeated 8× produced 80+ s of rambling audio. Natural text is fine.
- **The card reads full while the engine is up** (KV preallocation, above) —
  nothing else fits on the GPU next to it; `engines/tts down` it when done.
- Logs show `Failed to import nixl` at ERROR level — harmless; nixl is the
  multi-node relay transport, unused single-GPU.
- Output is 24 kHz mono 16-bit WAV. Identical calls give different takes
  (autoregressive sampling); pass `temperature: 0` territory at your own risk,
  it is not pinned here.
