# Qwen3-TTS engine

Wraps `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (Alibaba, Apache-2.0, not gated) behind the shared TTS bench contract.

## Build

```bash
cd engines/qwen3-tts
docker build -t tts-qwen3-tts:gpu .
```

## Run

```bash
../tts up qwen3-tts        # port 8882
```

(GPU device/driver flags and mounts live in `engines/tts`; see `engines/README.md`.)

First start downloads ~3.4 GB of weights into the mounted HF cache.

## Model / voices

- Checkpoint: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, loaded via the official `qwen-tts` pip package (`Qwen3TTSModel.generate_custom_voice`), bf16 on CUDA (fp16 fallback if bf16 unsupported).
- `/voices` lists the model's 9 built-in speakers (aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian). This CustomVoice checkpoint does speaker-id synthesis only — voice *cloning* from reference audio is the Base checkpoint's API (`generate_voice_clone`), so the repo's `voices/` clips are not used here.
- Default speaker when `voice` is null: **ryan** (English-native male) — used for both `en` and `pt`.

## Portuguese caveat (the thing we care about)

The model exposes a single `Portuguese` language id — **there is no pt-BR vs pt-PT selector anywhere in the API**, so nothing to expose as a body field. Brazilian reviewers report pt-BR/pt-PT phoneme confusion; whatever the bench hears with speaker `ryan` + language `Portuguese` is what you get. No native Portuguese speaker exists among the 9 built-ins.

## Streaming: NOT native

The model architecture supports streaming (~100 ms first packet per the paper), but `qwen-tts==0.1.1` does not expose any streaming *output* API — its `non_streaming_mode` flag only simulates streaming *text input*. So `/tts/stream` chunks text by sentence (`chunk_size` sentences per chunk, default 3) and emits one length-prefixed WAV per chunk. Revisit when the package grows a real streaming API.

## Pins and warts

- `qwen-tts==0.1.1` pins its own `transformers==4.57.3` and `accelerate==1.12.0` — don't fight it, give it its own image (which this is).
- `torch==2.8.0` from PyPI bundles CUDA 12.8 userspace libs; works with the host's 580.x kernel module via the mounted driver libs.
- flash-attn deliberately skipped (optional, ~10% faster, build-time swamp). Default SDPA attention is used.
- `torchaudio` must be pinned to the same version as torch (2.8.0) — unpinned, pip resolves torchaudio 2.11 (a CUDA 13 build) and the server dies at import with `libcudart.so.13: cannot open shared object file`.
- `sox` CLI installed via apt — the `pysox` dependency shells out to it.
- Extra body field on `/tts` and `/tts/stream`: `instruct` (string) — tone/emotion instruction, e.g. `"Very happy."`. Optional, never required.
- VRAM observed: **4.54 GiB** (RTX 4090, bf16, SDPA) after several syntheses — measured per-process via NVML.
- Synthesis speed observed: ~4.3–4.7 s for a ~6 s utterance (≈0.7× real-time on the 4090, without flash-attn).
