# VoxCPM2 engine

Wraps [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0, ~2B params,
MiniCPM-4 backbone, tokenizer-free continuous-latent diffusion-AR) behind the shared TTS
contract (`/health`, `/voices`, `/tts`, `/tts/stream`).

**Output is 48 kHz** — higher than the other engines in this bench (22–24 kHz). Not
downsampled on purpose; keep that in mind when comparing.

## Build (from the repo root — the image bakes in `voices/`)

```
docker build -f engines/voxcpm2/Dockerfile -t tts-voxcpm2:gpu .
```

## Run (pre-reboot GPU workaround for this box)

```
docker run -d --name tts-voxcpm2 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8883:8877 tts-voxcpm2:gpu
```

Model weights come from the mounted HF cache (`openbmb/VoxCPM2`, public, ~5 GB); first run
downloads them into the cache.

## What it loads

- `openbmb/VoxCPM2` via `voxcpm==2.0.3` (`VoxCPM.from_pretrained`), device auto → CUDA.
- Denoiser disabled (`load_denoiser=False`): it would drag in modelscope + an extra
  ModelScope model just to denoise reference clips, and ours are clean.
- `optimize=True` by default: voxcpm runs torch.compile + a warmup generate at load, so
  startup takes a few minutes but RTF is much better. Set `-e VOXCPM_OPTIMIZE=0` to skip.
  `/health` only answers once loading (incl. warmup) is done.

## Pins and why

- `voxcpm==2.0.3` installed `--no-deps`: its declared dependencies include torchcodec,
  funasr, modelscope, gradio and datasets, which are only used by its training code,
  denoiser and demo UI. The Dockerfile installs the actual inference-path imports
  (verified against the 2.0.3 sources): einops, librosa, soundfile, transformers,
  huggingface_hub, regex, tqdm, inflect, wetext.
- `transformers<5`: voxcpm imports `LlamaTokenizerFast` from the 4.x API.
- Base image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`: voxcpm needs torch>=2.5,
  CUDA>=12. gcc/g++ added for torch.compile (inductor needs a host compiler).

## Contract mapping

- `voice` → a clip name from `voices/` (baked into the image; `GET /voices` lists them),
  fed to the model's **basic cloning** mode (`reference_wav_path`). VoxCPM2 also has two
  other cloning modes (ref-audio-plus-transcript via `prompt_wav_path`+`prompt_text`, and
  controllable style cloning) plus a voice-design-from-text-description mode — not wired,
  by design.
- `lang` is validated (`en`/`pt` → else 400) but not passed to the model: VoxCPM2 is
  multilingual (30 languages incl. both) and infers the language from the text.
- `/tts/stream` uses **native streaming** (`generate_streaming()`): text is split into
  `chunk_size`-sentence pieces, and every audio chunk the model natively yields per piece
  is framed as `[4-byte BE length][complete WAV]` — so frames are smaller and more
  frequent than one-per-sentence-chunk.
- Optional extra knobs on `/tts` and `/tts/stream`: `cfg_value` (guidance, default 2.0),
  `inference_timesteps` (diffusion steps, default 10).

## Observed

- VRAM: <!-- FILL AFTER SMOKE TEST -->
- Warts: text normalization (`normalize=True`) is off — numbers/abbreviations are spoken
  as the model reads them; enabling it needs the `wetext` normalizer per language.
  With no `voice`, the model picks an arbitrary voice per request (no fixed default
  speaker) — pass a voice for stable output.
