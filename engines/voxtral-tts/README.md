# Voxtral TTS engine

Wraps `mistralai/Voxtral-4B-TTS-2603` (4B autoregressive decoder, 9 languages,
CC-BY-NC-4.0 — the weights inherit the preset voices' non-commercial licence,
released 2026-03) behind the shared bench contract. Mistral reports 70–90 ms
to first audio on their serving stack; the point of this engine is to check
whether that responsiveness survives contact with this box.

Served by **vLLM-Omni** — the stack Mistral's model card documents for this
model. `server.py` launches `vllm serve --omni` (OpenAI-compatible
`/v1/audio/speech`) as a subprocess and adapts the contract on 8877.

## Build (from the repo root)

```
docker build -f engines/voxtral-tts/Dockerfile -t tts-voxtral-tts:gpu .
```

Base image is `vllm/vllm-omni:v0.24.0` (multi-GB pull; the layer pull, not the
build, is what takes the time).

## Run

```
engines/tts up voxtral-tts
```

The registry entry uses extra-mount `voices`: the repo `voices/` clips are
mounted at `/app/voices` so the library appears in `/v1/voices`.

## What it loads

`mistralai/Voxtral-4B-TTS-2603` from the mounted host HF cache (~9 GB,
Mistral-format consolidated safetensors, public, ungated). Nothing is baked
into the image. Output is 24 kHz mono; vLLM-Omni runs it as a two-stage
pipeline (token generator + audio tokenizer) on one GPU.

## Contract notes

- **Languages**: the vendor claims en, fr, es, de, it, pt, nl, ar, hi; the
  adapter accepts exactly those and returns 400 for anything else. The model
  has no language parameter — it reads the language off the text — so `lang`
  is a validation gate only. en and pt are the two verified on this box.
- `/v1/voices`: the 20 preset speakers baked into the checkpoint are the
  builtins (gender is in the preset name); every library clip appears as a
  `reference` voice built from the manifest.
- **Voice cloning does not work on the public checkpoint** — see Warts. The
  adapter still accepts reference ids and forwards the clip as base64
  `ref_audio`, so it lights up if Mistral ever ships the encoder weights, but
  today every reference voice answers a 400 explaining the situation.
- `/v1/params`: `temperature`, `top_p`, `top_k` (forwarded via vLLM-Omni's
  `extra_params`, applied to the token stage's sampling), plus `seed` and
  `max_new_tokens` (top-level upstream fields). Defaults mirror the upstream
  deploy config: temperature 0, top_p 1, top_k -1, seed 42 — i.e. the stock
  engine is deterministic-leaning, unlike higgs.
- `stream: true` is **native streaming**: upstream emits raw 16-bit mono PCM
  as the codec stage produces it (`stream_format: "audio"`), passed through
  untouched.

## Pins and why

- Base `vllm/vllm-omni:v0.24.0` — the release tag, not `latest`; vLLM-Omni
  moves fast (0.20 → 0.25rc in four months) and the Voxtral pipeline config
  (`vllm_omni/deploy/voxtral_tts.yaml`) is auto-loaded from inside the wheel,
  so image tag == the whole serving behaviour. Nothing is pip-installed on
  top: vllm, vllm-omni, mistral-common, fastapi and uvicorn all come from the
  image, so they cannot drift apart.
- The upstream launch is exactly the model card's documented command
  (`vllm serve <model> --omni --trust-remote-code`); the deploy config
  auto-loads by model type. No flags are second-guessed here — the vendor's
  recipe on a 24 GB 4090 peaked at 20.5 GiB, which fits this card.

## Measurements

Observed on this box (RTX 4090 24 GB, nvdrv workaround, weights pre-cached):

- **Boot to healthy: ~TBD s** from container start.
- **VRAM: TBD** (deploy config: stage 0 at 0.8 GPU fraction, stage 1 at 0.1).
- **Warm synthesis: TBD**.
- **Streaming: TTFB TBD vs total TBD**.
- **Duration stability**: TBD (three identical English calls).

## Warts

- **The public checkpoint cannot clone.** `ref_audio` requires audio-encoder
  weights Mistral did not ship; upstream fails with `encode_waveforms
  requires encoder weights which are not available in the open-source
  checkpoint`. The adapter maps that to a 400 telling you to use a preset.
  Cloning-from-3-seconds is real only through Mistral's hosted API / gated
  weights.
- The preset voices are language-flavoured (`de_male`, `pt_female`, …) but
  the model speaks whatever language the input text is in, with the preset's
  timbre and accent. Cross-language use of a preset is legal and sounds like
  a foreigner speaking it.
- CC-BY-NC-4.0: the preset reference voices come from non-commercial corpora
  (EARS, CML-TTS, IndicVoices-R, …) and the weights inherit that — benchmark
  material, cannot ship commercially.
