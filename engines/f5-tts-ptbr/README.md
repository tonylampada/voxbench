# F5-TTS pt-BR engine

[F5-TTS](https://github.com/SWivid/F5-TTS) wrapped behind the shared benchmark
HTTP contract, with Portuguese served by the community finetune
[`firstpixel/F5-TTS-pt-br`](https://huggingface.co/firstpixel/F5-TTS-pt-br).

## ⚠️ License — benchmark only

Code is MIT, but the official F5-TTS weights (and therefore the pt-BR finetune,
which inherits from them) are **CC-BY-NC-4.0**. Non-commercial only — **we cannot
ship this engine**. It is here as the pt-BR quality yardstick.

## Models loaded

| lang | checkpoint | arch |
|------|-----------|------|
| `en` | `SWivid/F5-TTS` v1 base (package default) | `F5TTS_v1_Base` |
| `pt` | `firstpixel/F5-TTS-pt-br` → `pt-br/model_last.safetensors` | `F5TTS_Base` (old arch — the finetune predates v1, loading it with the default config fails with a state-dict mismatch) |

Vocab: the finetune uses the **stock 2545-token vocab** bundled with the
`f5-tts` package ("Tokenizer is the same as original F5-TTS" per the model
card), so no separate `vocab.txt` download is needed.

## Build

```bash
docker build -t tts-f5-tts-ptbr:gpu engines/f5-tts-ptbr/
```

Base image is the official `ghcr.io/swivid/f5-tts:main` (torch + CUDA + f5-tts
preinstalled). Only additions: `fastapi uvicorn num2words` — nothing needed
pinning; the base image's torch stack is used as-is.

## Run

Uses the explicit-device-nodes GPU recipe for this box (broken nvidia-container-toolkit
pre-reboot; see repo notes). Weights come from the mounted host HF cache — first
run downloads ~1.4 GB (finetune) + base v1 model + vocos vocoder into it.

```bash
docker run -d --name tts-f5-tts-ptbr \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -v "$(pwd)/voices:/app/voices:ro" \
  -p 127.0.0.1:8884:8877 tts-f5-tts-ptbr:gpu
```

(The server binds 0.0.0.0 *inside* the container; the host publish is
loopback-only, as required.)

## Voices

F5-TTS clones from a reference clip (<12 s) **plus its transcript**. The
contract's `voice` field only carries a name, so the adapter looks for a
sibling `.txt` transcript next to each clip in `voices/` (e.g.
`pt_male.flac` + `pt_male.txt`). **The repo currently has no `.txt`
transcripts**, so `/voices` reports only `"default"` — the English reference
clip bundled inside the `f5-tts` package. Requests naming a voice without a
transcript fall back to the default voice (logged, not an error).

Wart: the default reference is an *English* clip, so pt output is voiced by an
English speaker timbre. Drop a `<clip>.txt` transcript into `voices/` to get a
proper pt-BR reference voice.

## Text normalization (pt)

The pt-BR finetune has no text normalization of its own. For `lang: "pt"` the
adapter lowercases the input and expands digits with `num2words` (pt_BR), per
the model card. `en` text is passed through untouched.

## Streaming

`/tts/stream` is **not native streaming** — F5-TTS has none. The adapter chunks
text by sentence (`chunk_size` sentences per chunk, default 3) and emits one
length-prefixed WAV per chunk, so first-audio latency is still honestly
measurable. Single generation is capped at ~30 s of audio by the engine (it
auto-chunks internally beyond that).

## Optional knobs

`speed` (float, default 1.0) and `nfe_step` (int, default 32 diffusion steps)
are accepted as optional `/tts` body fields.

## VRAM observed

~TODO GB (filled in after smoke test).
