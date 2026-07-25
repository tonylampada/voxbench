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

(filled after first successful boot)

## Measurements

(VRAM, warm-call latency, duration-stability — filled after smoke test)

## Warts

(filled after smoke test)
