# XTTS-v2 (control engine)

Coqui XTTS-v2 — frozen since 2023, still the pt-BR quality benchmark. Built as a
**control**, not a candidate: the model weights are CPML-licensed, which is
permanently non-commercial — and since Coqui Inc. no longer exists, there is
nobody to buy a commercial license from. Use it as the quality bar to judge the
other engines against, nothing more.

## Build

```
cd engines/xtts-v2
docker build -t tts-xtts-v2:gpu .
```

## Run

```
docker run -d --name tts-xtts-v2 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.local/share/tts:/root/.local/share/tts \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8885:8877 tts-xtts-v2:gpu
```

The `--device`/`nvdrv` incantation is the pre-reboot workaround for this box's
mismatched driver stack — see the fleet notes; `--gpus all` does not work here.

## Model / cache

Loads `tts_models/multilingual/multi-dataset/xtts_v2` via the coqui-tts model
manager. **Weights live in the TTS cache (`~/.local/share/tts`), not the HF
cache** — that's why the run command mounts both; the HF mount is just the fleet
standard. The host already has the ~1.8 GB model cached, so startup does no
downloading. On a cold host the first start downloads the model into the
mounted cache (needs network).

First load normally prompts to accept the CPML TOS on stdin, which hangs a
container. `COQUI_TOS_AGREED=1` (set in the Dockerfile) is the documented
non-interactive accept; additionally the cached model dir already contains
`tos_agreed.txt`, so no prompt fires. Verified: container reaches healthy with
no interaction.

## Pins and why

- `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` base — torch **2.5**, not
  ≥2.6, because 2.6 flipped `torch.load` to `weights_only=True` by default and
  XTTS checkpoints are pickle-based.
- `coqui-tts==0.27.5` — the maintained **idiap fork**. Do NOT install the `TTS`
  package; that's the dead upstream and fights modern torch.
- `torchaudio==2.5.1` — matches the base image's torch so pip doesn't drag a
  different torch in.

## Contract notes

- `/voices` returns the 58 built-in XTTS studio speakers. Default speaker is
  `Ana Florence`. Passing a `voice` that is a file under an optionally mounted
  `/voices` dir does reference-clip cloning instead (`speaker_wav`).
- `/tts/stream` is **sentence-chunked, not native streaming**: text is split
  into sentences, grouped `chunk_size` per chunk, one framed WAV per chunk.
  XTTS does have native token streaming but it doesn't fit the framed-WAV
  contract; sentence chunking still measures honest first-audio latency.
- `pt` maps straight to XTTS `language="pt"`; anything other than `en`/`pt`
  gets a 400.

## Observed

- VRAM: ~3.6 GB resident after load (fits the ~4 GB expectation).
- Output: 24 kHz mono 16-bit WAV, peak-normalized.

## Warts

- Startup takes ~1–2 min even with cached weights (big pickle load + GPU move).
  `/health` returns 503 until ready — poll it, don't race it.
- The TTS cache mount is read-write: the model manager wants to write
  `tos_agreed.txt`/hash checks into the model dir on first run.
