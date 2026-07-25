#!/usr/bin/env python3
"""
Kokoro TTS adapter — translates the shared benchmark contract
(/health, /voices, /tts, /tts/stream) to the Kokoro-FastAPI
OpenAI-compatible API running on localhost:8880 in the same container.
"""
import io
import json
import re
import urllib.error
import urllib.request
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

UPSTREAM = "http://127.0.0.1:8880"
DEFAULT_VOICE = {"en": "af_heart", "pt": "pf_dora"}
# Kokoro picks the G2P language from the voice name prefix:
# a/b = English (US/UK), p = Brazilian Portuguese (via espeak-ng).
LANG_PREFIXES = {"en": ("a", "b"), "pt": ("p",)}

app = FastAPI(title="Kokoro TTS Adapter")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3
    speed: Optional[float] = None  # optional knob, 1.0 = normal


def upstream_get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{UPSTREAM}{path}", timeout=timeout) as r:
        return json.load(r)


def resolve_voice(lang: str, voice: Optional[str]) -> str:
    if lang not in LANG_PREFIXES:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")
    if not voice:
        return DEFAULT_VOICE[lang]
    if not voice.startswith(LANG_PREFIXES[lang]):
        raise HTTPException(
            status_code=400,
            detail=f"voice '{voice}' does not match lang '{lang}': Kokoro derives "
                   f"G2P language from the voice prefix ({LANG_PREFIXES[lang]}). "
                   f"Refusing to synthesize with the wrong phonemizer.",
        )
    return voice


def synthesize(text: str, voice: str, speed: Optional[float]) -> bytes:
    body = {
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "speed": speed if speed is not None else 1.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"kokoro upstream error: {e.read().decode(errors='replace')[:500]}")


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


@app.get("/health")
async def health():
    # 200 only once the upstream kokoro server answers — it loads the model
    # during startup, so reachable == ready to synthesize.
    try:
        upstream_get("/v1/audio/voices")
    except Exception:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "models_loaded": ["kokoro-v1_0"], "queue_size": 0}


@app.get("/voices")
async def voices():
    return {"voices": sorted(upstream_get("/v1/audio/voices")["voices"])}


@app.post("/tts")
async def tts(request: TTSRequest):
    voice = resolve_voice(request.lang, request.voice)
    wav = synthesize(request.text, voice, request.speed)
    return Response(content=wav, media_type="audio/wav")


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    voice = resolve_voice(request.lang, request.voice)

    def gen() -> Generator[bytes, None, None]:
        sentences = split_into_sentences(request.text)
        size = max(1, request.chunk_size)
        for i in range(0, len(sentences), size):
            chunk = " ".join(sentences[i:i + size])
            wav = synthesize(chunk, voice, request.speed)
            yield len(wav).to_bytes(4, "big") + wav

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav"},
    )
