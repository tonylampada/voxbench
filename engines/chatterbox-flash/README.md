# Chatterbox Flash engine

Wraps `ResembleAI/chatterbox-flash` (released 2026-07-06, MIT, ~520M, English-only,
block-diffusion with parallel unmasking — not the autoregressive path of the other
Chatterbox variants) behind the shared bench contract. Vendor-reported numbers:
RTF 0.076, 103 ms time-to-first-packet, Seed-TTS similarity 0.704, WER 1.96.

## Build (from the repo root — the build copies `voices/`)

```
docker build -f engines/chatterbox-flash/Dockerfile -t tts-chatterbox-flash:gpu .
```

## Run (broken-driver-stack recipe for this box)

```
docker run -d --name tts-chatterbox-flash \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8890:8877 tts-chatterbox-flash:gpu
```

Weights come from the mounted HF cache (`ResembleAI/chatterbox-flash`:
`t3_flash.safetensors` + `s3gen.safetensors` + `ve.safetensors` + `tokenizer.json`,
~2 GB, downloaded on first start if missing). Nothing is baked into the image.

## What it loads

- `lang: "en"` → `ChatterboxFlashTTS.from_pretrained()` (`chatterbox-flash` 0.1.0 PyPI
  package, a separate package from `chatterbox-tts` with its own API). T3 runs in
  bfloat16 with CUDA graphs; S3Gen is the meanflow-distilled vocoder at 2 CFM steps
  (package defaults).
- `lang: "pt"` → **400** ("Chatterbox Flash is English-only"). By design, not a gap.

## Pins and why

- `chatterbox-flash` (0.1.0, unpinned — only release). Pulls `chatterbox-tts>=0.1.7`,
  which itself pins `torch==2.6.0` (cu124 wheels); nothing else to pin.
- `setuptools<81` — `resemble-perth` comes in via the `chatterbox-tts` dependency and
  imports `pkg_resources`, removed in setuptools 81; without the pin the import dies
  with `'NoneType' object is not callable`.

## Contract notes

- `/tts/stream` is **not native streaming**. The paper/README advertise streaming
  generation, but `chatterbox-flash` 0.1.0's public API only exposes one-shot
  `generate()` / `generate_batch()` (the internal `S3GenStreamer` is not exported), so
  the adapter chunks text by sentence (`chunk_size` sentences per chunk, default 3)
  and emits one framed WAV per chunk. First-audio latency is real.
- **No builtin voice**: the release ships no `conds.pt`, every generation clones from a
  reference clip. `voice: null` falls back to `voices/en_female.flac` (~9 s clip).
  `/voices` lists the reference clips copied from the repo root's `voices/` dir.
  Conditioning per (voice, exaggeration) is prepared once and cached.
- Optional body fields: `exaggeration` (default 0.5, baked into conditioning),
  `temperature` (default 0.6), `num_steps` (diffusion steps, default 10).
- Output is 24 kHz mono 16-bit WAV, peak-normalized.

## Warts

- **No Perth watermark**: unlike the other Chatterbox variants, `chatterbox_flash`'s
  pipeline never imports or applies the `resemble-perth` watermarker — output audio is
  unwatermarked. (The package still *depends* on it transitively via `chatterbox-tts`,
  hence the setuptools pin.)
- VRAM observed: ~2.5 GB (container reported via `torch.cuda.memory_reserved` after
  load + one generation; host `nvidia-smi` is broken pre-reboot).
- Sentence splitting is a plain `[.!?…]` regex — decimal numbers/abbreviations may
  split oddly in `/tts/stream`; `/tts` synthesizes the whole text in one shot.
- The English text normalizer (`en_us_cleaner`) runs inside `generate()` by default
  (numbers, times, phone numbers spelled out).
