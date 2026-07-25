# OmniVoice pt-BR engine

`edwixx/omnivoice-brpt-v15` — a Brazilian Portuguese finetune (v1.5 `checkpoint-9000`) of
`k2-fsa/OmniVoice` (0.6B, Qwen3-0.6B backbone, diffusion-LM-style TTS). Best measured pt-BR
numbers in the survey: WER 0.040 / CER 0.038 on a 20-prompt pt-BR set, RTF ~0.025.

## ⚠️ License conflict — research-only until resolved

The finetune is tagged **Apache-2.0**, but the base `k2-fsa/OmniVoice` weights are
**CC-BY-NC**. A finetune cannot relicense the base weights, so until this is resolved
upstream, treat this engine as **non-commercial / research-only**. Not resolved here —
just recorded.

## Build

From the **repo root** (context needs `voices/`):

```
docker build -f engines/omnivoice-brpt/Dockerfile -t tts-omnivoice-brpt:gpu .
```

## Run

```
docker run -d --name tts-omnivoice-brpt \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  -e LD_LIBRARY_PATH=/opt/nvdrv \
  -v /home/ai/.local/share/nvdrv-580.159.03:/opt/nvdrv:ro \
  -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8887:8877 tts-omnivoice-brpt:gpu
```

First start downloads to the mounted HF cache: the finetune (~2.5 GB), the audio tokenizer
`eustlb/higgs-audio-v2-tokenizer` (the finetune repo ships no `audio_tokenizer/` folder;
the `omnivoice` library falls back to that repo automatically), and — only on the first
voice-cloning request — `openai/whisper-large-v3-turbo` (~1.6 GB) to auto-transcribe the
reference clip. Whisper runs on **CPU** (`asr_device="cpu"`): it transcribes a ~5 s clip
once per voice, and on GPU it OOMed when sibling engines filled the card.

## Model / language choices

- **Both `en` and `pt` are served from the finetune** (one model, one load). The base
  model was not loaded separately for English: the finetune is a light refine (LR 5e-6)
  of a 600+-language base, and English came out intelligible in the smoke test. If the
  bench shows degraded English, swap `MODEL_ID` to `k2-fsa/OmniVoice` for `en` — but that
  doubles VRAM, so it was not done preemptively.
- For `pt` with no voice, the adapter passes `instruct="portuguese accent"` — the exact
  instruction string the finetune was trained with (per the model card).
- Any other `lang` → 400.

## Voices

`GET /voices` lists the reference clips baked from the repo's `voices/` dir (~8 MB).
`voice` = a clip name (with or without extension) → zero-shot voice cloning against that
clip. The clip's transcript is not known, so the first request for each voice
auto-transcribes it with Whisper (slow, one-time; the resulting clone prompt is cached
in-process). `voice: null` → the model's auto-voice mode (it invents a voice).

## Streaming

**No native streaming upstream** — `/tts/stream` is the sentence-chunk fallback:
text split by sentence, `chunk_size` (default 3) sentences per generate call, one
length-prefixed WAV per chunk.

## Extra knobs (optional body fields)

- `speed` (float) — >1.0 faster, <1.0 slower.
- `num_step` (int) — diffusion steps, default 32; 16 is faster at some quality cost.
- `instruct` (str) — voice-design attributes (e.g. `"female, low pitch"`), used only
  when no `voice` is given; overrides the pt default instruction.

## Pins and why

- `omnivoice==0.2.1` — so the notes above stay true for what ships.
- Base image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` — omnivoice wants
  torch>=2.4, upstream docs recommend 2.8/cu128; using the official image avoids a
  10-minute pip torch install.
- transformers left floating (omnivoice requires >=5.3.0); nothing observed to need a pin.
- No `resemble-perth`/setuptools trap here — that is a Chatterbox-image issue.

## VRAM observed

~2.5 GB resident after load and during plain `/tts` synthesis (fp16, RTX 4090).
Brief extra allocations (tens of MB) during reference-clip encoding for voice cloning.
Whisper runs on CPU, so it adds no VRAM.

## Warts

- `flash_attn` is not installed; the model falls back to SDPA (upstream supports this
  explicitly). Fine for the bench.
- Endpoints are sync and GPU work is behind a single lock — concurrent requests queue
  rather than race. The bench hits it serially anyway.
- First `/tts` after cold start is slower (CUDA/graph warmup); the bench's `/health`
  poll does not warm the generate path.
- **Voice cloning was not fully verified on this box**: during smoke testing three
  sibling engines held 21+ of 24 GB, and the clone path OOMed on ~2–88 MiB allocations
  (ref-clip encode, then a generate step). Plain `/tts`, `/tts/stream`, `/voices`,
  `/health`, and the 400 path all passed under that same contention. With the card to
  itself (bench conditions) cloning needs only tens of MB beyond the resident 2.5 GB
  and is expected to pass — but it was not observed passing end-to-end.
