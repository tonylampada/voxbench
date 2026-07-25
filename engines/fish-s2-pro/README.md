# Fish Audio S2 Pro engine

Wraps `fishaudio/s2-pro` (5B dual-AR, #1 open-weights on Artificial Analysis, Fish Audio
Research License — non-commercial, fine here) behind the shared bench contract.

Thin stack: the official fish-speech API-server image runs `tools/api_server.py` (loads the
model, serves `/v1/tts` on localhost:8080 inside the container) and a stdlib-only adapter
(`server.py`) translates the bench contract on port 8877. No pip installs on top of the
image — the adapter deliberately uses only the Python standard library so the image's pinned
uv environment stays untouched.

## Checkpoint

`fishaudio/s2-pro` (public, not gated) from the host HF cache:
`~/.cache/huggingface/hub/models--fishaudio--s2-pro` (~10.5 GB: 2 safetensors shards + codec.pth).
Pre-download with `uvx --from 'huggingface_hub[cli]' hf download fishaudio/s2-pro --exclude overview.png`.
The entrypoint resolves the snapshot dir at runtime; nothing is baked into the image.

## Build

```
docker build -f engines/fish-s2-pro/Dockerfile -t tts-fish-s2-pro:gpu .
```

Base image: `fishaudio/fish-speech:server-cuda` (rebuilt 2026-06-09). Note the tag from the
model card docs (`latest-webui-cuda`) does not exist on Docker Hub — the real tags are
`webui-cuda` / `server-cuda`; we use the server one (no gradio). Torch imports verified
working in-container before anything was built on top (the round-one F5-TTS lesson).

## Run

```
engines/tts up fish-s2-pro
```

Registered on port 8892 with mounts `voices` (the live voice library, for reference
cloning) and `hf` (the host HF cache at `/hf:ro`).

The HF cache is mounted at `/hf` (not `/root/.cache`) because the container runs as uid 1000
(`fish`), which cannot traverse `/root`. Model load takes ~1–2 min; `/health` returns 503
until the upstream server is actually ready.

**KV-cache cap (the big wart):** the shipped `config.json` declares `max_seq_len: 32768`,
which makes the server preallocate a ~4.8 GB KV cache → ~16.5 GB total, and it OOMs next to
the resident 8 GB chatterbox-gpu on the 24 GB card. The entrypoint symlink-copies the
snapshot and patches `max_seq_len` to 4096 (~3 minutes of audio at the codec's 21 Hz —
far beyond any bench utterance). Weights are untouched; override with `-e FISH_MAX_SEQ_LEN=N`.

## Contract notes (v2)

- `lang`: both `en` (Tier 1) and `pt` (Tier 2) are supported; the model infers language from
  the text, there is no explicit language switch upstream.
- `/v1/voices` → the `default` builtin (the model's unconditioned voice) plus the full
  shared library as `reference` voices. **Cloning works for any library clip**: every clip
  reads the bench seed sentence in its own language (`bench/mkvoicelib.py`), so the adapter
  knows each clip's transcript by construction and sends `references=[{audio, text}]`
  (base64 audio over JSON — the fish-speech schema decodes it) to upstream `/v1/tts`.
- `stream: true` → **400**. fish-speech's `streaming: true` is cosmetic for s2-pro: measured
  against upstream directly, every PCM byte of an 8-sentence text arrives in one burst at
  the end (TTFB == total; its WAV-header and final-blob yields are numpy arrays its own
  async wrapper silently drops). The contract says don't fake it.
- `/v1/params`: `temperature`, `top_p`. No `speed` knob — `speed` other than 1.0 → 400.
- Inline control tags (`[whisper]`, `[laughing]`, …) pass straight through in the text.

## Serving path

Plain fish-speech PyTorch server (`tools/api_server.py`), not SGLang. The SGLang path is a
separate serving stack; the vendor's own docker server image is the laziest thing that works
and is what the round-one engines were also measured against (vendor-default serving).

## Observed

- VRAM: **14.27 GiB** for the server process (NVML per-process query in-container), with the
  4096-token KV cache. With the stock 32768 config it wants ~16.5 GB and OOMs under
  contention.
- Latency: ~6.5–7.6 s per ~6 s utterance, non-streaming, while sharing the GPU with
  chatterbox-gpu (not a clean benchmark number — the bench measures properly).
- `uv run` re-syncs the venv at container start (re-downloads ~300 MB of nvidia wheels,
  cu129→cu126 downgrade) unless `UV_NO_SYNC=1` is set — the entrypoint sets it.
- Warts: host `pyenv` shims are broken in this worktree (`.python-version` wants a missing
  3.11), so use `uvx` for the HF download, not `huggingface-cli`.
