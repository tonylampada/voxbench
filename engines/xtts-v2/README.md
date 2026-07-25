# XTTS-v2 (control engine)

Coqui XTTS-v2 — frozen since 2023, still the pt-BR quality benchmark. Built as a
**control**, not a candidate: the model weights are CPML-licensed, which is
permanently non-commercial — and since Coqui Inc. no longer exists, there is
nobody to buy a commercial license from. Use it as the quality bar to judge the
other engines against, nothing more.

## Build

The image derives from the repo's own chatterbox image, so build that first if
it isn't already local:

```
docker build -t chatterbox-tts:gpu .            # repo root (skip if image exists)
docker build -f engines/xtts-v2/Dockerfile -t tts-xtts-v2:gpu .  # repo root, ~2 min on top of the base
```

Why not a pytorch/pytorch base: on benchmark night this box's WiFi couldn't
pull a 3 GB base image with five engine builds running in parallel (50 min in,
still pulling). `chatterbox-tts:gpu` was already local and carries torch
2.6.0+cu124, which the idiap fork handles fine.

## Run

```
../tts up xtts-v2        # port 8885
```

(GPU device/driver flags and mounts — including this engine's extra coqui TTS
cache mount — live in `engines/tts`; see `engines/README.md`.)

## Model / cache

Loads `tts_models/multilingual/multi-dataset/xtts_v2` via the coqui-tts model
manager. **Weights live in the TTS cache (`~/.local/share/tts`), not the HF
cache** — that's why the run command mounts both; the HF mount is just the
fleet standard. The host already has the ~1.8 GB model cached, so startup
downloads nothing. On a cold host the first start downloads into the mounted
cache (needs network). The TTS cache mount is read-write on purpose: the model
manager writes `tos_agreed.txt` and hash files into the model dir.

First load normally prompts to accept the CPML TOS on stdin, which would hang a
container. `COQUI_TOS_AGREED=1` (set in the Dockerfile) is the documented
non-interactive accept. On this host the cached model dir already contains
`tos_agreed.txt`, so the prompt never fired; the env var stays as the belt for
cold-cache runs. Verified: container reaches healthy with zero interaction.

## Pins and why

- `coqui-tts==0.27.5` — the maintained **idiap fork**. Do NOT install the
  `TTS` package; that's the dead upstream and it fights modern torch.
- `transformers>=4.57,<4.58` — the trap of this engine. The base image has
  transformers 5.2.0 and coqui-tts's own spec (`>=4.57`) *accepts* it, but
  XTTS's tortoise layers import `isin_mps_friendly`, which transformers 5.x
  removed — the unpinned install dies at `from TTS.api import TTS`. 4.57 is
  the newest release that both satisfies coqui-tts and still has the symbol.
- torch is whatever the base carries (2.6.0+cu124). The idiap fork ≥0.25.2
  handles torch 2.6's `weights_only=True` default when loading XTTS pickles.

## Contract notes

- `/voices` returns the 58 built-in XTTS studio speakers. Default speaker is
  `Ana Florence`. A `voice` naming a file under an optionally mounted `/voices`
  dir does reference-clip cloning instead (`speaker_wav`).
- `/tts/stream` is **sentence-chunked, not native streaming**: text is split
  into sentences, grouped `chunk_size` per chunk, one framed WAV per chunk.
  XTTS has native token streaming but it doesn't fit the framed-WAV contract;
  sentence chunking still measures honest first-audio latency.
- `pt` maps straight to XTTS `language="pt"`; anything other than `en`/`pt`
  gets a 400.

## Observed (RTX 4090, benchmark night)

- Startup to healthy: **~30 s** with cached weights. `/health` is 503 until
  the model is loaded — poll it, don't race it.
- VRAM: **2.5 GiB** resident (NVML per-process, after synthesis).
- Speed: 7.3 s of EN audio in 1.7 s wall; 8.3 s of PT audio in 1.1 s wall
  (RTF ≈ 0.15).
- Output: 24 kHz mono 16-bit WAV, peak-normalized.

## Warts

- The image is coupled to `chatterbox-tts:gpu` being buildable/local (see
  Build). Swap the FROM back to `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
  on a box with real bandwidth if that coupling ever bites.
- pip warns that chatterbox's own pins conflict with the downgraded
  transformers inside this image. Irrelevant here — chatterbox code is never
  imported in this container — but don't reuse this image to run chatterbox.
