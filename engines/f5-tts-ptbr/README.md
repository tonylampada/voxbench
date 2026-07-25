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
docker build -f engines/f5-tts-ptbr/Dockerfile -t tts-f5-tts-ptbr:gpu .   # repo root
```

Base image is the official `ghcr.io/swivid/f5-tts:main` (21.9 GB, CUDA +
f5-tts 1.1.22 preinstalled).

**Wart — the official image is broken as shipped.** It carries torch
2.13.0+cu130 next to torchaudio 2.4.0 and torchvision 0.19.0, both built
against torch 2.4. `import f5_tts` dies immediately (`libtorchaudio.so:
undefined symbol _ZNK5torch8autograd4Node4nameEv`, and once past that,
`operator torchvision::nms does not exist` via transformers). The Dockerfile
fixes it by pinning the newest consistent cu130 pair — `torch==2.11.0` +
`torchaudio==2.11.0` (torchaudio caps at 2.11 on cu130) — and uninstalling
torchvision, which nothing here uses and transformers treats as optional.
Only other addition: `num2words`; fastapi/uvicorn already ship in the base.

## Run

```bash
../tts up f5-tts-ptbr        # port 8884
```

(GPU device/driver flags and the `voices/` mount live in `engines/tts`; see
`engines/README.md`.) Weights come from the mounted host HF cache — first
run downloads ~1.4 GB (finetune) + base v1 model + vocos vocoder into it.

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

**~1.9 GiB** with both models (en + pt) loaded, measured on the RTX 4090 as
the delta in `torch.cuda.mem_get_info()` before/after stopping the container.

## Smoke-test timings (RTX 4090)

- `/tts` en, 7.9 s of audio: **1.6 s** wall
- `/tts` pt, 8.1 s of audio: **0.6 s** wall
- output: 24 kHz mono 16-bit WAV, peak-normalized
