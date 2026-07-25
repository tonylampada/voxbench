#!/usr/bin/env python3
"""Piper TTS adapter — CPU-only ONNX engine, built-in voices, no cloning."""
import io
import re
import wave
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from piper import PiperVoice, SynthesisConfig

VOICES_DIR = "/voices"
VOICE_NAMES = [
    "pt_BR-faber-medium",
    "pt_BR-cadu-medium",
    "pt_BR-edresson-low",
    "pt_BR-jeff-medium",
    "en_US-lessac-medium",
]
DEFAULT_VOICE = {"pt": "pt_BR-faber-medium", "en": "en_US-lessac-medium"}

voices = {}

app = FastAPI(title="Piper TTS Server")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3  # sentences per chunk (stream only)
    length_scale: Optional[float] = None  # optional speed knob: >1.0 slower, <1.0 faster


@app.on_event("startup")
def load_voices():
    for name in VOICE_NAMES:
        print(f"Loading {name}...")
        voices[name] = PiperVoice.load(f"{VOICES_DIR}/{name}.onnx")
    print("All voices loaded.")


def pick_voice(lang: str, voice: Optional[str]) -> PiperVoice:
    if lang not in DEFAULT_VOICE:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")
    name = voice or DEFAULT_VOICE[lang]
    if name not in voices:
        raise HTTPException(
            status_code=400,
            detail=f"unknown voice {name!r}; available: {sorted(voices)}",
        )
    return voices[name]


def synth_wav(voice: PiperVoice, text: str, length_scale: Optional[float]) -> bytes:
    syn_config = SynthesisConfig(length_scale=length_scale) if length_scale else None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=syn_config)
    return buf.getvalue()


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@app.post("/tts")
def tts(req: TTSRequest):
    voice = pick_voice(req.lang, req.voice)
    wav_bytes = synth_wav(voice, req.text, req.length_scale)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    voice = pick_voice(req.lang, req.voice)
    sentences = split_into_sentences(req.text) or [req.text]
    chunks = [
        " ".join(sentences[i : i + req.chunk_size])
        for i in range(0, len(sentences), req.chunk_size)
    ]

    def gen():
        for chunk in chunks:
            wav_bytes = synth_wav(voice, chunk, req.length_scale)
            yield len(wav_bytes).to_bytes(4, "big") + wav_bytes

    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.get("/voices")
def list_voices():
    return {"voices": sorted(voices)}


@app.get("/health")
def health():
    if len(voices) < len(VOICE_NAMES):
        raise HTTPException(status_code=503, detail="voices still loading")
    return {"status": "ok", "models_loaded": sorted(voices), "queue_size": 0}
