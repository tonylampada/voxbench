#!/usr/bin/env python3
"""
Chatterbox Nano TTS engine — bench contract adapter.

ResembleAI/chatterbox-nano (110M, English only), loaded through
ChatterboxTurboTTS with nano=True. Device via NANO_DEVICE=cpu|cuda.
"""
import io
import os
import re
import wave
from pathlib import Path
from typing import Optional, Generator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = os.getenv("NANO_DEVICE", "cuda")
VOICES_DIR = Path("voices")

app = FastAPI(title="Chatterbox Nano TTS Engine")
models = {}


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def get_voice_path(voice: Optional[str]) -> Optional[str]:
    if not voice:
        return None
    for candidate in [VOICES_DIR / voice] + [VOICES_DIR / f"{voice}{ext}" for ext in (".flac", ".wav", ".mp3")]:
        if candidate.exists():
            return str(candidate)
    raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")


def check_lang(lang: str):
    if lang != "en":
        raise HTTPException(status_code=400, detail="Chatterbox Nano is English only; lang must be 'en'")


def generate_speech(text, voice_path, temperature=None, top_p=None, repetition_penalty=None) -> np.ndarray:
    print(f"[en] {text}")
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    wav = models["en"].generate(text, audio_prompt_path=voice_path, **kwargs)

    audio = wav.cpu().numpy().squeeze()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def numpy_to_wav_bytes(audio: np.ndarray) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(models["en"].sr)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


@app.on_event("startup")
def startup():
    print(f"Device: {DEVICE}")
    print("Loading Chatterbox Nano...")
    models["en"] = ChatterboxTurboTTS.from_pretrained(device=DEVICE, nano=True)
    print("Chatterbox Nano engine ready")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys()), "queue_size": 0}


@app.get("/voices")
def list_voices():
    voices = sorted(f.name for f in VOICES_DIR.iterdir() if f.suffix in (".flac", ".wav", ".mp3")) if VOICES_DIR.exists() else []
    return {"voices": voices}


@app.post("/tts")
def tts(request: TTSRequest):
    check_lang(request.lang)
    audio = generate_speech(
        request.text, get_voice_path(request.voice),
        request.temperature, request.top_p, request.repetition_penalty,
    )
    return Response(content=numpy_to_wav_bytes(audio), media_type="audio/wav")


def stream_chunks(request: TTSRequest, voice_path: Optional[str]) -> Generator[bytes, None, None]:
    chunks = chunk_sentences(split_into_sentences(request.text), request.chunk_size)
    for chunk in chunks:
        audio = generate_speech(
            chunk, voice_path,
            request.temperature, request.top_p, request.repetition_penalty,
        )
        wav_bytes = numpy_to_wav_bytes(audio)
        yield len(wav_bytes).to_bytes(4, 'big') + wav_bytes


@app.post("/tts/stream")
def tts_stream(request: TTSRequest):
    check_lang(request.lang)
    voice_path = get_voice_path(request.voice)
    return StreamingResponse(
        stream_chunks(request, voice_path),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(models["en"].sr)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
