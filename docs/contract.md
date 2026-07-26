# TTS bench contract v2

Every engine adapter in `engines/` serves this HTTP surface on port 8877 inside its
container. This document is the single source of truth; `bench/conform.py <engine>`
checks every clause below against a live server and fails on any deviation.

The shape is OpenAI's `/v1/audio/speech`, minus what makes no sense here and plus
what the bench needs. There is **no backwards compatibility**: the v1 surface
(`/tts`, `/tts/stream`, `GET /voices`, the `[4-byte length][WAV]` stream framing,
`chunk_size`, voice-as-filepath) is deleted. A migrated adapter answers 404 on all
of it. `model` does not exist as a request field — a model here is a container,
you pick it by picking which engine to run.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/audio/speech` | synthesize |
| GET | `/v1/voices` | the voice catalogue |
| GET | `/v1/params` | the engine's tunable knobs |
| GET | `/health` | readiness |
| GET | `/` | human test page (HTML) |

Anything else is 404.

## Errors

Every client error is **400** with a JSON body `{"detail": "<human sentence>"}`
that says what was wrong and, where possible, what would have been right.
No 422s — adapters map framework validation errors to 400. Upstream/engine
failures are 5xx.

## POST /v1/audio/speech

Request body (JSON):

```json
{"input": "Hello there.", "voice": "af_bella", "lang": "en",
 "response_format": "wav", "speed": 1.0, "stream": false,
 "params": {"speed": 1.3}}
```

| Field | Type | Default | Rules |
|---|---|---|---|
| `input` | string | — required | Text to speak. Missing, empty, or whitespace-only → 400. |
| `voice` | string | engine default for `lang` | A voice **id** from `/v1/voices`. Unknown id → 400. Ids are opaque tokens — never a file path. |
| `lang` | string | derived from `voice`; else engine default | ISO 639-1 code. A lang the engine can't speak → 400. If both `voice` and `lang` are given they must agree (the voice's `langs` must contain it) → else 400. |
| `response_format` | string | `"wav"` | Non-streaming output container. Every engine must support `"wav"`; anything it doesn't support → 400. When `stream` is true this field must be omitted or `"pcm"` (streaming is raw PCM by definition) → else 400. |
| `speed` | number | `1.0` | OpenAI-compat alias for `params.speed` — exactly the same knob. Present in both places → 400. On an engine whose `/v1/params` has no `speed`, any value other than 1.0 → 400 (1.0 is a no-op and accepted everywhere, so stock OpenAI clients work). |
| `stream` | bool | `false` | See streaming below. |
| `params` | object | `{}` | Engine-specific knobs. Every key must appear in `/v1/params` → unknown key 400. Values must match the declared type and `min`/`max` → else 400. |

### Non-streaming response

`200` with the audio bytes, `Content-Type` matching `response_format`
(`audio/wav` for wav). No framing, no envelope — the body is the file.

### Streaming response (`stream: true`)

`200` chunked. The body is **raw signed 16-bit little-endian mono PCM**, sent
progressively as the engine produces it — first bytes must arrive well before
synthesis of the whole input finishes. Headers:

- `x-sample-rate`: sample rate in Hz (e.g. `24000`). Required.
- `Content-Type: application/octet-stream`.

No per-chunk framing of any kind. The client concatenates chunks and has PCM.
An engine that cannot stream does not fake it by splitting sentences; it
returns 400 for `stream: true` with a sentence saying so.

## GET /v1/voices

```json
{"voices": [
  {"id": "af_bella", "name": "Bella", "kind": "builtin",
   "langs": ["en"], "gender": "female"},
  {"id": "xtts-v2-ana-florence-pt", "name": "Ana Florence", "kind": "reference",
   "origin": "xtts-v2", "langs": ["pt"], "gender": "female", "seconds": 10.2}
]}
```

Every entry:

| Field | Rules |
|---|---|
| `id` | Opaque, unique within the engine, **no extension, no directory separators**. Resolving id → file/upstream-name is the adapter's private business. |
| `name` | Human display name. |
| `kind` | `"builtin"` (baked into the model) or `"reference"` (a cloning reference clip from the shared voice library). |
| `langs` | Non-empty list of ISO 639-1 codes the voice can speak. |
| `gender` | `"female"`, `"male"`, or `"unknown"`. |
| `origin` | reference only: which engine rendered the clip (`voices/library/manifest.json` `engine` field). |
| `seconds` | reference only: clip duration. |

The catalogue is **built from `voices/library/manifest.json`**, not from a
directory walk — every manifest entry with a non-null `file` appears as a
`reference` voice whose id is the filename minus `.wav`
(`xtts-v2-ana-florence-pt.wav` → `xtts-v2-ana-florence-pt`). This makes the
catalogue recursive by construction and keeps the manifest's language, gender
and duration data authoritative. Manifest entries with `"file": null` (failed
renders) are omitted.

Every adapter lists the full library, including clips another engine rendered.
Whether it can *synthesize* with a given reference is a separate question: a
cloning engine can use any clip; a non-cloning engine (kokoro, piper) can only
use reference ids whose `origin` is itself — those resolve back to the
equivalent builtin — and answers 400 with an explanation for the rest. The
console filters on `origin`/`kind` as it sees fit; the catalogue never lies by
omission.

## GET /v1/params

Describes every engine-specific knob that `params` accepts — the console
renders per-model controls from this and nothing else.

```json
{"params": {"speed": {"type": "number", "min": 0.25, "max": 4.0,
                      "default": 1.0, "description": "playback speed multiplier (1.0 = normal)"}}}
```

Each knob: `type` (`"number"`, `"integer"`, `"boolean"`, `"string"`), `default`,
and a one-sentence human `description` are required; numeric knobs also carry
`min` and `max`. An engine with no knobs returns `{"params": {}}` — a real
answer, not an omission.

## GET /health

`200 {"status": "ok", "engine": "<name>"}` once the model is loaded and the
engine can synthesize; `503 {"status": "loading"}` before that. Extra fields
are allowed; those two are required.

An engine that has hit an unrecoverable fault — a poisoned CUDA context is the
one we have met — answers `503 {"status": "unusable", "detail": "<what died and
that only a container restart cures it>"}` from then on. A process that cannot
synthesize must never answer `ok`: `engines/tts ls` reads that state as `dead`
and the console paints it red instead of waiting forever for a green light.

## GET /

`200`, `text/html` — the shared test page, for humans.

## Cross-origin (CORS)

An engine must be callable from a browser page it does not serve — the console
runs on its own port and talks to the engine directly. Every response carries:

- `Access-Control-Allow-Origin` — the request's own `Origin`, echoed back, plus
  `Vary: Origin`. Never a hardcoded value.
- `Access-Control-Expose-Headers: x-sample-rate` — **load-bearing**. Without it
  the browser hands the page the streaming body but hides the header, and raw
  PCM without its sample rate is unplayable.

`OPTIONS /v1/audio/speech` answers the preflight with
`Access-Control-Allow-Methods` (including `POST`) and
`Access-Control-Allow-Headers: content-type` — the board posts JSON, which is
not a simple request, so the browser preflights it before every call.

The allowlist is the env var **`TTS_CORS_ORIGINS`**: comma-separated origins,
`*` allowed as a wildcard (it matches a port, or anything else). Default —
loopback only:

```
TTS_CORS_ORIGINS=http://127.0.0.1:*,http://localhost:*
```

An `Origin` outside the list gets **no CORS header at all**. The request is not
allowed and the browser's refusal is the correct answer, not a bug.

`Access-Control-Allow-Origin: *` is **not** an option. These engines have no
authentication of any kind and one of them drives a 24 GB GPU; with `*`, any
page in any tab of any browser that can route to the port could drive it. The
allowlist is the only thing standing there.

## The 400 catalogue

The cases a conforming adapter must reject, each with a helpful `detail`:

1. `input` missing, empty, or whitespace-only.
2. `voice` id not in the catalogue.
3. `lang` the engine cannot speak.
4. `voice` and `lang` both given but incompatible.
5. A reference voice this engine cannot synthesize with (non-cloning engine, foreign `origin`).
6. A `params` key not present in `/v1/params`.
7. A `params` value of the wrong type or outside `min`/`max`.
8. `response_format` the engine doesn't support; or a non-`pcm` `response_format` combined with `stream: true`.
9. `speed` given both top-level and in `params`; or `speed != 1.0` on an engine without a `speed` knob.
10. `stream: true` on an engine that cannot stream.
