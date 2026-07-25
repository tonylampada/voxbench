# Piper (CPU) engine

Piper TTS wrapped behind the fleet HTTP contract. CPU-only — no GPU, no torch, no voice
cloning. ONNX inference via `piper-tts` 1.6.0.

## Build

```
docker build -f engines/piper/Dockerfile -t tts-piper:gpu .   # repo root
```

(Image name keeps the fleet's `:gpu` pattern for uniformity; it uses no GPU.)

## Run

```
docker run -d --name tts-piper -p 127.0.0.1:8888:8877 tts-piper:gpu
```

No GPU flags, no volume mounts. Voices are baked into the image at build time — they are
small ONNX files (~20–65 MB each, ~250 MB total), and the fleet's "no weights in the image"
rule targets multi-GB HF checkpoints; baking these keeps the run command mount-free.

## Models / voices

All loaded at startup from `rhasspy/piper-voices` (downloaded at image build):

- `pt_BR-faber-medium` — **default for `lang: "pt"`**
- `pt_BR-cadu-medium`
- `pt_BR-edresson-low`
- `pt_BR-jeff-medium`
- `en_US-lessac-medium` — **default for `lang: "en"`**

The `voice` field in `/tts` means a **built-in voice name** from the list above (Piper
cannot clone from a reference clip — that is the trade for CPU speed). Unknown voice → 400.
Unknown lang → 400.

`/tts/stream` is **not native streaming**: Piper has no incremental API here, so the adapter
splits text by sentence, synthesizes `chunk_size` (default 3) sentences per chunk, and emits
one length-prefixed WAV per chunk. Still honest for first-audio latency — a chunk arrives as
soon as it is synthesized.

Optional knob: `length_scale` (float body field) — >1.0 slower speech, <1.0 faster.

Sample rate is per-voice (medium = 22050 Hz, low = 16000 Hz); the WAV header carries it.

## Pins and why

- `piper-tts==1.6.0` — the live project is [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl);
  `rhasspy/piper` is **archived**. Pinned so the licensing note below stays true for what ships.
- fastapi/uvicorn unpinned — nothing in this adapter is version-sensitive.

## Licensing — read before shipping

- The Piper **engine** (piper1-gpl) is **GPL-3.0** — they vendored espeak-ng. If we embed or
  distribute this engine, GPL obligations apply. Running it as a separate network service
  (as here) does not virally infect callers, but shipping it in a product does.
- The **voices** (`rhasspy/piper-voices`) remain **MIT**.

## VRAM

Zero. CPU-only; ~500 MB RAM with all five voices loaded.

## Warts

- The container must be started fresh to pick up new voices — the voice list is a constant
  in `server.py` and voices are baked at build time.
- `/health` returns 200 only after all five voices are loaded (uvicorn also doesn't accept
  connections until startup completes, so a connectable port already implies ready).
