# Gepard 1.0 — TTS engine

[nineninesix/gepard-1.0](https://huggingface.co/nineninesix/gepard-1.0) — 555M autoregressive TTS
(Qwen3.5 full-attention backbone → NVIDIA NeMo NanoCodec), 22.05 kHz, Apache-2.0.
Locales: en-US, en-UK, es-MX, pt-BR. Language is inferred from the text — there is no
language parameter in the model; the adapter validates `lang` ∈ {en, pt} and otherwise
just synthesizes.

## Build

```
cd engines/gepard
docker build -t tts-gepard:gpu .
```

## Run

```
docker run -d --name tts-gepard \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8886:8877 tts-gepard:gpu
```

Model load ≈ 40 s (NeMo import + checkpoint + codec download on first run); `/health`
answers only after load.

## What it loads

- Checkpoint `nineninesix/gepard-1.0` (public, ungated) via the vendor's
  [gepard-inference](https://github.com/nineninesix-ai/gepard-inference) package,
  pinned at commit `fa19f57`.
- Codec `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` (public), auto-selected from the
  checkpoint's `gepard_config.json`, loaded through NeMo's `AudioCodecModel`.

**Why not the plain `transformers` pipeline:** the task brief suggested
`pipeline("text-to-speech", model="nineninesix/gepard-1.0")` is the whole install. It is not:
the pipeline loads (transformers ≥ 5.15 dev; every released version 5.3.0–5.14.1 crashes on
Gepard's full-attention-only qwen3_5 config) but returns **raw FSQ token IDs**, not audio —
transformers has no NanoCodec decode for this model. The vendor's inference package is the
real minimal path: it carries the `FullAttnCache` workaround for the transformers bug and the
token→waveform codec step.

## Pins and why

- `transformers==5.3.0` — vendor's pin; their `FullAttnCache` workaround targets it.
- `gepard-inference @ fa19f57` — unreleased package installed from git; pinned so upstream
  pushes can't break the build.
- `nemo-toolkit[tts]==2.4.0` — vendor's pin; supplies `AudioCodecModel`. Installed **before**
  the gepard package so the transformers pin lands last (NeMo pins an older transformers).
- `uv` instead of pip for the NeMo/gepard layers — pip's resolver runs >10 min on
  nemo-toolkit's tree and then errors; vendor's own setup scripts use uv for the same reason.
- `build-essential` — nemo-toolkit pulls sdists (`texterrors`, `kaldi-python-io`) that compile
  C++ at install time; python:3.12-slim has no compiler.
- torch cu128 wheels, index-only — coherent set from the pytorch index; runs on the CUDA-13
  580 driver.

## Observed

- VRAM: **2264 MiB** (bf16 backbone + codec, per-process via NVML).
- `/tts`: ~2.1 s for 5.9 s EN audio, ~1.6 s for 6.5 s PT audio (RTX 4090 shared with other
  engines; eager PyTorch path — the ~25× realtime / 50 ms TTFA claims are the vLLM serving
  path, deliberately not used here).
- Image: 10.5 GB.

## Warts

- `/tts/stream` is a **sentence-chunked pseudo-stream** (one WAV per `chunk_size` sentences),
  not native streaming — Gepard's native streaming lives in the gepard-vllm path only.
- `/voices` lists the vendor repo's reference clips (`audio_en`, `audio_ru`, `nurisa_en`,
  `ulan_emo`); default voice is `audio_en`. There is **no pt-BR reference clip** in the vendor
  repo — pt text is synthesized with the en reference voice, which is exactly the "quality
  varies outside English" condition the bench is measuring. `voice` accepts any of the listed
  names; unknown names → 400.
- Voice cloning from arbitrary clips works via the vendor session but is not exposed —
  the contract has no voice-upload field, and the four listed clips cover the bench.
- Generation params default to the vendor's config (temperature 0.3, cfg_scale 1.0
  = single-pass, no CFG). Optional body fields `temperature`, `cfg_scale` are passed through.
