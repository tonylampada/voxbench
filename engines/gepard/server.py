#!/usr/bin/env python3
"""Gepard 1.0 TTS adapter — same HTTP contract as the repo-root Chatterbox server.

Thin wrapper around the vendor's GepardSession (gepard-inference package):
text -> FSQ tokens (Qwen3.5 backbone) -> waveform (NVIDIA NanoCodec, 22.05 kHz).
"""
import io
import re
import wave
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from gepard_inference import GepardSession
from gepard_inference.session import SessionConfig

GEPARD_ROOT = Path("/opt/gepard")
VOICES_DIR = GEPARD_ROOT / "ref_audio"
DEFAULT_VOICE = "audio_en.wav"

app = FastAPI(title="Gepard TTS Server")
session: Optional[GepardSession] = None


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3  # sentences per chunk (/tts/stream)
    # optional engine knobs (vendor defaults otherwise)
    temperature: Optional[float] = None
    cfg_scale: Optional[float] = None


def split_into_sentences(text: str) -> list[str]:
    """Same splitter as the repo-root reference server, so chunking is comparable."""
    text = text.strip()
    if not text:
        return []

    abbreviations = [
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.',
        'etc.', 'e.g.', 'i.e.', 'St.', 'Mt.', 'Lt.', 'Gen.', 'Col.',
        'Fig.', 'Vol.', 'No.', 'pp.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.',
        'a.m.', 'p.m.', 'A.M.', 'P.M.', 'U.S.', 'U.K.', 'E.U.',
    ]

    protected = text
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        if abbr in protected:
            placeholders[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)

    protected = re.sub(r'(\d+)\.(\d+)', r'\1__DEC__\2', protected)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', protected)

    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s = s.replace('__DEC__', '.')
        for placeholder, abbr in placeholders.items():
            s = s.replace(placeholder, abbr)
        result.append(s)
    return result


def chunk_sentences(sentences: list[str], chunk_size: int = 3) -> list[str]:
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def resolve_voice(voice: Optional[str]) -> Optional[str]:
    """Voice name -> reference clip path. None -> cached default voice."""
    if not voice:
        return None
    for candidate in (VOICES_DIR / voice, VOICES_DIR / f"{voice}.wav"):
        if candidate.exists():
            return str(candidate)
    available = sorted(p.stem for p in VOICES_DIR.glob("*.wav"))
    raise HTTPException(status_code=400, detail=f"unknown voice {voice!r}; available: {available}")


def synthesize(text: str, voice_path: Optional[str], req: TTSRequest) -> np.ndarray:
    _, audio = session.synthesize(
        text, reference=voice_path,
        temperature=req.temperature, cfg_scale=req.cfg_scale,
    )
    # peak normalize like the reference server
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def validate(request: TTSRequest):
    if request.lang not in ["en", "pt"]:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")


@app.on_event("startup")
async def startup():
    global session
    print("Loading Gepard 1.0 (backbone + NanoCodec)...")
    cfg = SessionConfig(
        checkpoint="nineninesix/gepard-1.0",
        defaults=dict(
            temperature=0.3, top_k=0, cfg_scale=1.0, cfg_frames=0,
            stop_threshold=0.5, max_frames=2000,
            repetition_penalty=1.0, repetition_window=32,
        ),
        reference_audio=DEFAULT_VOICE,
        root=VOICES_DIR,
    )
    session = GepardSession(cfg).load()
    print("Gepard TTS Server Ready!")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": ["nineninesix/gepard-1.0"], "queue_size": 0}


@app.get("/voices")
async def list_voices():
    return {"voices": sorted(p.stem for p in VOICES_DIR.glob("*.wav"))}


@app.post("/tts")
async def tts(request: TTSRequest):
    validate(request)
    voice_path = resolve_voice(request.voice)
    audio = synthesize(request.text, voice_path, request)
    return Response(
        content=numpy_to_wav_bytes(audio, session.player.sample_rate),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """Sentence-chunked pseudo-stream: [4-byte BE length][WAV]... per chunk."""
    validate(request)
    voice_path = resolve_voice(request.voice)
    chunks = chunk_sentences(split_into_sentences(request.text), request.chunk_size)

    def generate() -> Generator[bytes, None, None]:
        for chunk in chunks:
            audio = synthesize(chunk, voice_path, request)
            wav_bytes = numpy_to_wav_bytes(audio, session.player.sample_rate)
            yield len(wav_bytes).to_bytes(4, 'big') + wav_bytes

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(session.player.sample_rate)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
