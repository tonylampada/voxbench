#!/usr/bin/env python3
"""
Higgs TTS 3 adapter — translates the shared benchmark contract
(/health, /voices, /tts, /tts/stream) to the SGLang-Omni
OpenAI-compatible /v1/audio/speech API running on localhost:8000
in the same container.

The upstream server (sgl-omni serve) is launched as a subprocess at
startup; /health answers 200 only once it accepts requests, which
happens only after the 4B model is loaded.
"""
import io
import json
import os
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

UPSTREAM = "http://127.0.0.1:8000"
MODEL = os.environ.get("HIGGS_MODEL", "bosonai/higgs-tts-3-4b")
VOICES_DIR = Path("/app/voices")
# Higgs is a single multilingual model: the language comes from the text
# itself, there is no lang parameter upstream. Both en and pt sit in its
# sub-5% WER tier, so both are accepted; anything else is refused loudly.
SUPPORTED_LANGS = {"en", "pt"}

app = FastAPI(title="Higgs TTS 3 Adapter")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3  # accepted for contract parity; native streaming ignores it
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None


def start_upstream() -> None:
    args = [
        "sgl-omni", "serve",
        "--model-path", MODEL,
        "--port", "8000",
        "--host", "127.0.0.1",
        # 24 GB card: the model declares 8192 ctx, but bench sentences are
        # tiny — capping the declared context bounds KV/CUDA-graph allocation
        # (sibling engine Fish S2 Pro OOMed exactly here).
        "--context-length", os.environ.get("HIGGS_CONTEXT_LEN", "4096"),
    ]
    # Reference clips are mounted at /app/voices; upstream refuses to read
    # local files outside --allowed-local-media-path.
    if VOICES_DIR.is_dir():
        args += ["--allowed-local-media-path", str(VOICES_DIR)]
    subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)


def resolve_references(voice: Optional[str]) -> list:
    """Map a voice name to upstream `references`. None/'default' → builtin
    voice (no references). Otherwise the name must be a clip in /app/voices;
    a sibling .txt transcript is passed along when present (improves cloning)."""
    if not voice or voice == "default":
        return []
    candidates = [VOICES_DIR / voice] + [
        VOICES_DIR / f"{voice}{ext}" for ext in (".flac", ".wav", ".mp3")
    ]
    for clip in candidates:
        if clip.is_file():
            ref = {"audio_path": str(clip)}
            txt = clip.with_suffix(".txt")
            if txt.is_file():
                ref["text"] = txt.read_text().strip()
            return [ref]
    raise HTTPException(status_code=400, detail=f"voice '{voice}' not found in voices/")


def upstream_body(req: TTSRequest) -> dict:
    if req.lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    body = {"model": MODEL, "voice": "default", "input": req.text}
    refs = resolve_references(req.voice)
    if refs:
        body["references"] = refs
    for knob in ("temperature", "top_k", "max_new_tokens"):
        v = getattr(req, knob)
        if v is not None:
            body[knob] = v
    return body


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
async def health():
    # sgl-omni's HTTP server only answers once the model is loaded, so
    # upstream reachable == ready to synthesize.
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=3):
            pass
    except Exception:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "higgs-tts-3", "models_loaded": [MODEL], "queue_size": 0}


@app.get("/voices")
async def voices():
    # "default" is the model's builtin voice; the rest are reference clips
    # mounted at /app/voices, used for zero-shot cloning.
    names = ["default"]
    if VOICES_DIR.is_dir():
        names += sorted(f.name for f in VOICES_DIR.iterdir()
                        if f.suffix.lower() in (".flac", ".wav", ".mp3"))
    return {"voices": names}


@app.post("/tts")
async def tts(request: TTSRequest):
    body = upstream_body(request)
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return Response(content=r.read(), media_type="audio/wav")
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"higgs upstream error: {e.read().decode(errors='replace')[:500]}")


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    # Native streaming: upstream emits raw 16-bit mono PCM chunks as the
    # vocoder produces them (sub-second first audio). Each chunk is wrapped
    # in its own WAV and framed [4-byte BE length][WAV]. chunk_size (a
    # sentences-per-chunk knob for engines that fake streaming) is ignored.
    body = upstream_body(request) | {"stream": True, "response_format": "pcm"}
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"higgs upstream error: {e.read().decode(errors='replace')[:500]}")
    sample_rate = int(resp.headers.get("x-sample-rate", 24000))

    def gen() -> Generator[bytes, None, None]:
        # Coalesce vocoder chunks to ≥0.2 s per frame so the bench isn't
        # flooded with tiny WAVs; the first frame flushes as soon as it fills.
        min_bytes = int(sample_rate * 2 * 0.2)
        pending = b""
        with resp:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                pending += chunk
                if len(pending) >= min_bytes:
                    wav = pcm_to_wav(pending, sample_rate)
                    yield struct.pack(">I", len(wav)) + wav
                    pending = b""
        if pending:
            wav = pcm_to_wav(pending, sample_rate)
            yield struct.pack(">I", len(wav)) + wav

    return StreamingResponse(gen(), media_type="application/octet-stream")


start_upstream()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
