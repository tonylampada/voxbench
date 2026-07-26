# Speaking to a voxbench TTS engine

You are talking to a text-to-speech engine over HTTP. This file is everything you need —
you do not have to read any other file in that repo.

Every engine answers the **same** API, so this works whatever is running on the port you
were given. What differs between engines is which voices exist, which knobs exist, and
whether it can stream — and you ask the engine itself for all three.

Base URL: `http://127.0.0.1:<port>`. If nobody told you the port, ask; do not guess.

## Ask before you speak

```
GET /health      → 200 {"status": "ok", "engine": "kokoro"}   (503 while the model loads)
GET /v1/voices   → the voice catalogue
GET /v1/params   → the knobs this engine accepts
```

Call `/v1/voices` and `/v1/params` once and work from the answers. Never hardcode a
voice id or a parameter name — they are per-engine, and an engine will reject what it
does not have.

`/v1/voices`:

```json
{"voices": [
  {"id": "af_bella", "name": "Bella", "kind": "builtin",
   "langs": ["en"], "gender": "female"},
  {"id": "xtts-v2-ana-florence-pt", "name": "Ana Florence", "kind": "reference",
   "origin": "xtts-v2", "langs": ["pt"], "gender": "female", "seconds": 10.2}
]}
```

- `id` is an **opaque token** — never a file path, never with an extension. Send it back
  verbatim.
- `kind: "builtin"` is a voice baked into the model. `kind: "reference"` is a cloning
  clip from the shared library.
- A cloning engine can use any reference voice. A non-cloning engine (kokoro, piper)
  lists them all anyway, but only accepts the ones whose `origin` is itself; anything
  else comes back 400 with an explanation. If you want a voice that always works, pick
  `kind: "builtin"`, or check `origin`.
- `langs` says what that voice can speak. Filter by it.

`/v1/params` describes the engine's own knobs, and there are engines with none
(`{"params": {}}` — a real answer, not an omission):

```json
{"params": {"speed": {"type": "number", "min": 0.25, "max": 4.0,
                      "default": 1.0,
                      "description": "playback speed multiplier (1.0 = normal)"}}}
```

Render or reason from this; every key you send in `params` must appear here, with a
value of the declared type inside `min`/`max`.

## Synthesize

```
POST /v1/audio/speech
content-type: application/json
```

```json
{"input": "Olá, capitão.", "voice": "af_bella", "lang": "pt",
 "response_format": "wav", "speed": 1.0, "stream": false, "params": {}}
```

| Field | Default | Rules |
|---|---|---|
| `input` | required | The text. Empty or whitespace-only → 400. |
| `voice` | engine default for `lang` | An id from `/v1/voices`. Unknown → 400. |
| `lang` | from `voice`, else engine default | ISO 639-1. If you send both `voice` and `lang` they must agree. |
| `response_format` | `"wav"` | Every engine supports `wav`. With `stream: true` it must be absent or `"pcm"`. |
| `speed` | `1.0` | Alias for `params.speed` — the same knob. Sending both → 400. On an engine with no `speed` knob, anything but 1.0 → 400. |
| `stream` | `false` | See below. |
| `params` | `{}` | Engine-specific knobs, keys from `/v1/params`. |

The minimum that works anywhere:

```bash
curl -s localhost:8888/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input": "Olá, capitão.", "lang": "pt"}' -o out.wav
```

**Non-streaming response:** `200`, and the body **is the audio file**. No envelope, no
base64, no JSON. Write the bytes to disk and you have a playable WAV.

## Streaming

`stream: true` returns `200` chunked, and the body is **raw signed 16-bit
little-endian mono PCM** — no header, no framing of any kind. The sample rate is in the
`x-sample-rate` response header. Concatenate the chunks and you have PCM; wrap it in a
44-byte WAV header if you need a file.

Not every engine streams. One that cannot returns **400 with a sentence saying so** —
that is a correct answer, not a failure. If you need to know in advance, send a
one-word `input` with `stream: true` once and remember the outcome.

```python
import urllib.request, json, wave

req = urllib.request.Request(
    "http://127.0.0.1:8888/v1/audio/speech",
    data=json.dumps({"input": "Olá, capitão.", "lang": "pt", "stream": True}).encode(),
    headers={"content-type": "application/json"})

with urllib.request.urlopen(req) as r:          # HTTPError 400 = this engine cannot stream
    rate = int(r.headers["x-sample-rate"])
    pcm = b""
    while chunk := r.read(4096):
        pcm += chunk                            # play it here instead, that is the point

with wave.open("out.wav", "wb") as w:           # only if you want a file
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(pcm)
```

In a browser, feed each chunk to an `AudioContext` as `Int16Array` → `Float32Array`
(divide by 32768) and schedule the buffers back to back. The whole reason to stream is
that sound starts before synthesis ends — if you buffer to the end, use `stream: false`
and save yourself the work.

## When something is wrong

Every client error is **400** with `{"detail": "<a human sentence>"}`. There are no
422s. The sentence says what was wrong and, where it can, what would have been right —
read it and act on it rather than retrying blind. A `5xx` is the engine breaking, not
you.

The ten cases you will actually hit:

1. `input` missing, empty or whitespace-only.
2. `voice` not in the catalogue.
3. `lang` this engine cannot speak.
4. `voice` and `lang` given but incompatible.
5. A reference voice this engine cannot clone with.
6. A `params` key not in `/v1/params`.
7. A `params` value of the wrong type or out of range.
8. An unsupported `response_format`, or a non-`pcm` one with `stream: true`.
9. `speed` sent twice, or `speed != 1.0` on an engine without that knob.
10. `stream: true` on an engine that cannot stream.

## Things that are not here

There is no `model` field — a model is a container, and you chose it when you chose the
port. There is no API key and no auth of any kind: these engines are meant to be
reachable only from localhost or a private tailnet. Do not put one on a public address.
