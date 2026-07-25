# Kokoro-82M engine

Kokoro-82M wrapped behind the shared benchmark contract (`/health`, `/voices`,
`/tts`, `/tts/stream`), via the maintained community container
[remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (Apache-2.0,
code and weights).

## What runs

One container, two processes (see `start.sh`):

1. The vendor's Kokoro-FastAPI server (`api.src.main:app`) on `127.0.0.1:8880`
   inside the container — loads the model, serves the OpenAI-compatible
   `/v1/audio/speech`.
2. `server.py` — a thin translation shim on `8877` (the only published port)
   that maps our contract onto the vendor API.

## Model / checkpoint

`kokoro-v1_0` (Kokoro-82M), **baked into the vendor image** at
`/app/api/src/models/v1_0` — the base image downloads it at its build time, so
nothing is fetched from HuggingFace at container start and the HF cache mount
is not actually needed for this engine (kept in the run command only for
consistency with the other engines).

## Build

```
docker build -f engines/kokoro/Dockerfile -t tts-kokoro:gpu .   # repo root
```

## Run

```
../tts up kokoro        # port 8889
```

(GPU device/driver flags and mounts live in `engines/tts`; see `engines/README.md`.
The HF cache mount is not actually needed here — the model is baked into the
vendor image — but the shared run path keeps it for consistency.)

Health is 503 until the model server is up (typically well under a minute),
then `GET /health` → `{"status": "ok", ...}`.

## Pins and why

- Base image pinned to `v0.2.4-master` (pushed 2026-07-21) — the tag named in
  the survey; `latest` currently points at v0.6.0 with a different CUDA/layout.
- Nothing else is installed: the shim uses only the vendor venv
  (fastapi/uvicorn) plus Python stdlib (`urllib`) for upstream calls. No
  setuptools/`pkg_resources` trap here — `resemble-perth` is not in this stack.

## Voices / languages

- `voice` is a built-in Kokoro voice name (no cloning). Full list from
  `GET /voices` (~65 voices, all languages).
- Defaults: `en` → `af_heart`, `pt` → `pf_dora`.
- Portuguese: only `pf_dora`, `pm_alex`, `pm_santa`. Kokoro self-grades its
  pt as "D" — neutral pan-lusophone accent, not Brazilian. It goes through
  espeak-ng G2P, which the vendor image ships preinstalled.
- Kokoro derives the G2P language from the **voice-name prefix** (`a`/`b` =
  English, `p` = Portuguese). The shim enforces lang↔voice-prefix consistency
  and returns 400 on mismatch, so Portuguese text can never be silently run
  through the English phonemizer.

## Streaming

`/tts/stream` is **not** native streaming: the shim splits text into
sentences, groups them `chunk_size` at a time, makes one non-streaming vendor
request per group, and emits each result as a length-prefixed WAV frame. First
frame arrives after the first chunk synthesizes (fast — Kokoro is ~100x
realtime on GPU).

## Observed

- **NOT YET SMOKE-TESTED.** The base-image pull deadlocked in the local docker
  daemon for 2+ hours (all layers "Download complete", never registered) behind
  concurrent sibling pulls of shared layers on a saturated link — a known moby
  shared-layer lock issue, not an engine problem. Every command above is
  written from the vendor's v0.2.4 Dockerfile/entrypoint (verified against the
  tag's source) but has not been executed end-to-end on this box.
- VRAM: not measured (expected <1 GB per the survey).
- Wart: `docker pull` of the base image needs ghcr auth refresh if the stored
  token is stale (`gh auth token | docker login ghcr.io -u <user> --password-stdin`);
  anonymous HEAD returns an unhelpful `denied: denied`.
