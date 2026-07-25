# Chatterbox pt-BR engine

Wraps `ResembleAI/Chatterbox-Multilingual-pt-br` (the dedicated Brazilian Portuguese
finetune, shipped 2026-06-10, reported CER 0.35%, MIT, ungated) plus
`ResembleAI/chatterbox-turbo` for English, behind the shared bench contract.

## Build (from the repo root — the build copies `voices/`)

```
docker build -f engines/chatterbox-ptbr/Dockerfile -t tts-chatterbox-ptbr:gpu .
```

## Run

```
../tts up chatterbox-ptbr        # port 8881
```

(GPU device/driver flags and mounts live in `engines/tts`; see `engines/README.md`.)

Weights come from the mounted HF cache (downloaded on first start if missing —
~2 GB for the finetune). Nothing is baked into the image.

## What it loads

- `lang: "en"` → `ChatterboxTurboTTS.from_pretrained()` (`ResembleAI/chatterbox-turbo`), same as the repo-root server.
- `lang: "pt"` → the pt-BR finetune. `chatterbox-tts` 0.1.7's `from_pretrained()` only knows
  the base multilingual checkpoint, so `server.py` has a small custom loader that mirrors the
  official ResembleAI pt-BR HF Space: `t3_pt_br.safetensors` + `s3gen_v3.pt` from the finetune
  repo, voice encoder / tokenizer / builtin voice (`ve.pt`, `grapheme_mtl_merged_expanded_v1.json`,
  `conds.pt`) from the base `ResembleAI/chatterbox` repo. `s3gen_v3.pt` is loaded with
  `strict=False` (the Space does the same; a few buffers differ from 0.1.7's S3Gen).

## Pins and why

- `chatterbox-tts==0.1.7` — the PyPI release this adapter's custom loader was written against.
- `setuptools<81` — `resemble-perth` (the watermarker) imports `pkg_resources`, removed in
  setuptools 81; without the pin the model dies at load with `'NoneType' object is not callable`.
- torch is pinned by chatterbox-tts itself (`torch==2.6.0`, cu124 wheels).

## Contract notes

- `/tts/stream` is **not native streaming** — Chatterbox has no incremental decode in this API;
  the text is chunked by sentence (`chunk_size` sentences per chunk, default 3) and one framed
  WAV is emitted per chunk. First-audio latency is real.
- `/voices` lists the reference clips copied from the repo root's `voices/` dir (voice cloning
  via `audio_prompt_path`). `voice: null` uses each model's builtin voice (`conds.pt`).
- Optional body fields: `exaggeration`, `cfg_weight`, `temperature` (model defaults: pt 0.5/0.5/0.8, en Turbo 0.0/0.0/0.8).
- Output is 24 kHz mono 16-bit WAV, peak-normalized.

## Warts

- **Chatterbox embeds a Perth neural watermark in every output by default** (the package
  applies it inside `generate()`; not disabled here).
- `s3gen_v3.pt` load reports **2 missing / 0 unexpected keys** with `strict=False` — same as
  the official Space; output audio is fine.
- VRAM observed: could not isolate per-process (host `nvidia-smi` is broken pre-reboot);
  with this engine plus one sibling Chatterbox container resident the box showed 17.2/23.5 GiB
  used. Vendor-expected is 2–3 GB per model; both models resident ≈ 5–6 GB. Container peak
  RSS was ~6.1 GB.
- Sentence splitting is a plain `[.!?…]` regex — decimal numbers/abbreviations may split
  oddly in `/tts/stream`; `/tts` synthesizes the whole text in one shot and is unaffected.
