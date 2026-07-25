# Voxtral TTS engine

Wraps `mistralai/Voxtral-4B-TTS-2603` (4B autoregressive decoder, 9 languages,
CC-BY-NC-4.0 — the weights inherit the preset voices' non-commercial licence,
released 2026-03) behind the shared bench contract. Mistral reports 70–90 ms
to first audio on their serving stack; the point of this engine is to check
whether that responsiveness survives contact with this box.

Served by **vLLM-Omni** — the stack Mistral's model card documents for this
model. `server.py` launches `vllm serve --omni` (OpenAI-compatible
`/v1/audio/speech`) as a subprocess and adapts the contract on 8877.

## Build (from the repo root)

```
docker build -f engines/voxtral-tts/Dockerfile -t tts-voxtral-tts:gpu .
```

Base image is `vllm/vllm-omni:v0.22.0` (21 GB pull; the layer pull, not the
build, is what takes the time).

## Run

```
engines/tts up voxtral-tts
```

The registry entry uses extra-mount `voices`: the repo `voices/` clips are
mounted at `/app/voices` so the library appears in `/v1/voices`.

## What it loads

`mistralai/Voxtral-4B-TTS-2603` from the mounted host HF cache (~9 GB,
Mistral-format consolidated safetensors, public, ungated). Nothing is baked
into the image. Output is 24 kHz mono; vLLM-Omni runs it as a two-stage
pipeline (token generator + audio tokenizer) on one GPU.

## Contract notes

- **Languages**: the vendor claims en, fr, es, de, it, pt, nl, ar, hi; the
  adapter accepts exactly those and returns 400 for anything else. The model
  has no language parameter — it reads the language off the text — so `lang`
  is a validation gate only. en and pt are the two verified on this box.
- `/v1/voices`: the 20 preset speakers baked into the checkpoint are the
  builtins (gender is in the preset name); every library clip appears as a
  `reference` voice built from the manifest.
- **Voice cloning does not work on the public checkpoint** — see Warts. The
  adapter treats this as a non-cloning engine: its own 20 presets are
  rendered into the voice library (`bench/mkvoicelib.py`), reference ids with
  origin `voxtral-tts` resolve back to the preset, and foreign clips answer
  a 400 explaining the situation.
- `/v1/params`: `temperature`, `top_p`, `top_k` (forwarded via vLLM-Omni's
  `extra_params`, applied to the token stage's sampling), plus `seed` and
  `max_new_tokens` (top-level upstream fields). Defaults mirror the upstream
  deploy config: temperature 0, top_p 1, top_k -1, seed 42 — i.e. the stock
  engine is deterministic-leaning, unlike higgs.
- `stream: true` is **native streaming**: upstream emits raw 16-bit mono PCM
  as the codec stage produces it (`stream_format: "audio"`), passed through
  untouched.

## Pins and why

- Base `vllm/vllm-omni:v0.22.0` (vllm 0.22.0, vllm-omni 0.22.0,
  mistral-common 1.11.2) — a release tag, not `latest`; vLLM-Omni moves fast
  (0.20 → 0.25rc in four months) and the Voxtral pipeline config
  (`vllm_omni/deploy/voxtral_tts.yaml`) is auto-loaded from inside the wheel,
  so image tag == the whole serving behaviour. Nothing is pip-installed on
  top: vllm, vllm-omni, mistral-common, fastapi and uvicorn all come from the
  image, so they cannot drift apart.
- **Not v0.24.0, deliberately.** That release answers 200 with fluent speech
  of *some other text* — "The quick brown fox…" came back as "We're back.
  We're back." (vllm-omni #4949: #4527 moved the audio feedback frames under
  `codes.audio` and Voxtral's preprocess kept reading the old key, so the
  AR stage generated unconditioned). The fix (#4954) merged after the
  release; v0.22.0 predates the breakage and is a release the vendor recipe
  explicitly verified for this model — also without v0.24.0's measured
  time-to-first-packet regression (vllm-omni #5023).
- The upstream launch is exactly the model card's documented command
  (`vllm serve <model> --omni --trust-remote-code`); the deploy config
  auto-loads by model type. No flags are second-guessed here — the vendor's
  recipe on a 24 GB 4090 peaked at 20.5 GiB, which fits this card.

## Measurements

Observed on this box (RTX 4090 24 GB, nvdrv workaround, weights pre-cached):

- **Boot to healthy: ~76–81 s** from container start (model load; no downloads).
- **VRAM: 20.8 / 24.0 GiB** once warm (deploy config: stage 0 at 0.8 GPU
  fraction, stage 1 at 0.1 — matches the vendor recipe's 20.5 GiB on the same
  card). Nothing else fits next to it; `engines/tts down` it when done.
- **Warm synthesis: RTF ≈ 0.20** — 16.4 s of audio in 3.2 s wall; ~10 s
  renders take ~1.8 s.
- **Streaming is genuinely incremental: first PCM at 148 ms**, against 5.1 s
  total for 26 s of audio — first byte at 3% of the wall clock. The vendor's
  70–90 ms claim is nearly true even through a docker bridge and an adapter
  hop; this is the fastest first-audio on the bench after VoxCPM2, and unlike
  VoxCPM2 it is also one of the fastest generators.
- **Duration stability**: the same two-sentence English paragraph 3× →
  16.40 / 17.52 / 16.48 s. No gepard-style silent truncation; whisper-base
  transcribes the output faithfully (en, pt, fr checked).

## Warts

- **The public checkpoint cannot clone.** `ref_audio` requires audio-encoder
  weights Mistral did not ship; upstream raises `encode_waveforms requires
  encoder weights which are not available in the open-source checkpoint` —
  and it raises **inside the EngineCore, which kills the whole upstream
  engine** (every later request 500s until the container restarts). The
  adapter therefore never forwards `ref_audio`: reference voices whose
  origin is `voxtral-tts` resolve back to their builtin preset (like kokoro
  and piper), everything else is a 400 up front. Cloning-from-3-seconds is
  real only through Mistral's hosted API / gated weights.
- The preset voices are language-flavoured (`de_male`, `pt_female`, …) but
  the model speaks whatever language the input text is in, with the preset's
  timbre and accent. Cross-language use of a preset is legal and sounds like
  a foreigner speaking it.
- CC-BY-NC-4.0: the preset reference voices come from non-commercial corpora
  (EARS, CML-TTS, IndicVoices-R, …) and the weights inherit that — benchmark
  material, cannot ship commercially.
