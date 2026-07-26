# Chatterbox Nano engine

Wraps `ResembleAI/chatterbox-nano` (110M, MIT, English only, released 2026-07-06,
re-pushed 2026-07-21) behind the shared bench contract. Loaded through
`ChatterboxTurboTTS.from_pretrained(device, nano=True)` — Nano shares Turbo's
architecture with a smaller GPT-2 backbone and a 1-step (distilled meanflow)
token-to-mel decoder.

## Build (from the repo root)

```
docker build -f engines/chatterbox-nano/Dockerfile -t tts-chatterbox-nano:gpu .
```

The image is based on the repo's shared dependency layer `tts-base:gpu` (build it
first with `docker build -f engines/base/Dockerfile -t tts-base:gpu .` if it is
not already local) — chosen over a fresh `python:3.11-slim` because this box's link
was at ~50 KB/s with several engine builds pulling in parallel, and the base
already carries torch
2.6.0+cu124, the full dependency tree, `setuptools<81`, and `voices/`.

## Run

GPU (broken-driver-stack recipe for this box):

```
docker run -d --name tts-chatterbox-nano \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv -e HF_HUB_DISABLE_XET=1 \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8891:8877 tts-chatterbox-nano:gpu
```

CPU (same image, no device passthrough — this is the interesting mode; Resemble
claim ~3x realtime on 8 cores):

```
docker run -d --name tts-chatterbox-nano \
  -e NANO_DEVICE=cpu -e HF_HUB_DISABLE_XET=1 \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8891:8877 tts-chatterbox-nano:gpu
```

`NANO_DEVICE=cpu|cuda`, default `cuda`. Weights come from the mounted HF cache
(~3 GB pulled on first start if missing); nothing is baked into the image.

## What it loads

`ResembleAI/chatterbox-nano`: `t3_nano_v1.safetensors` (110M backbone),
`s3gen.safetensors` + `s3gen_meanflow.safetensors` (distilled decoder),
`ve.safetensors`, builtin voice `conds.pt`, BPE tokenizer. All via the vendor's
own `from_pretrained(nano=True)` — no custom loader.

## Pins and why

- `chatterbox-tts @ .../archive/5de7a54aa4e5.tar.gz` — **PyPI's 0.1.7
  (2026-03-26) predates Nano** and has no `nano=` parameter; the pin is GitHub
  master's "Update ChatterboxNano" commit (2026-07-21), the same code the HF
  demo Space runs. Installed from the tarball (no git in the base image) with
  `--no-deps`: every pin in master's pyproject is already satisfied by the base
  (torch 2.6.0, transformers 5.2.0, safetensors 0.5.3, diffusers 0.29.0,
  omegaconf, resemble-perth 1.0.1) except gradio, which only the demo scripts
  import.
- `setuptools<81` — inherited from the base image; `resemble-perth` (the
  watermarker) imports `pkg_resources`, removed in setuptools 81; without it
  model load dies with `'NoneType' object is not callable`.
- torch 2.6.0 cu124 wheels run fine on CPU too, which is what makes the
  one-image cpu/cuda switch work.
- The base bakes `HF_HUB_OFFLINE=1` and an in-image `HF_HOME`; this Dockerfile
  overrides both (`HF_HOME=/root/.cache/huggingface`, `HF_HUB_OFFLINE=0`) so the
  standard host-cache mount works and Nano's weights can be pulled on first run.

## Contract notes

- **English only.** `lang: "pt"` → 400 with a clear message.
- `/tts/stream` is **not native streaming** — text is chunked by sentence
  (`chunk_size` per chunk, default 3), one framed WAV per chunk.
- `/voices` lists the reference clips in `voices/` (inherited from the base
  image; cloning via `audio_prompt_path`, clips must be >5 s). `voice: null`
  uses the builtin voice (`conds.pt`).
- **Paralinguistic tags are inline text markup** — put `[laugh]`, `[chuckle]`,
  `[cough]` etc. directly in `text`; no API surface needed.
- Optional body fields: `temperature` (default 0.8), `top_p` (0.95),
  `repetition_penalty` (1.2). Nano ignores `cfg_weight`/`exaggeration`
  (the package warns and drops them), so they are not exposed.
- Output is 24 kHz mono 16-bit WAV, peak-normalized.

## Warts

- **Perth neural watermark embedded in every output** by default (applied inside
  the package's `generate()`; not disabled here).
- **First-run weight download crawls under hf_xet** on this box: the
  unauthenticated xet backend pulled at ~0.2 MB/s; adding
  `-e HF_HUB_DISABLE_XET=1` to `docker run` switched to plain LFS at ~4.6 MB/s.
  Only matters until the ~3 GB are cached; harmless to leave on.
- Sentence splitting is a plain `[.!?…]` regex — decimals/abbreviations may
  split oddly in `/tts/stream`; `/tts` synthesizes in one shot and is unaffected.
- VRAM: could not isolate per-process (host `nvidia-smi` is broken pre-reboot).
  Container RSS observed: 3.2 GiB in GPU mode, 2.6 GiB in CPU mode.
- Observed on this box (24-core, contended): GPU warm call 0.4 s wall for ~5 s
  audio; CPU warm call 3.2 s wall for 5.2 s audio. First call after load is
  slower (6–8 s) in both modes.
