#!/usr/bin/env python3
"""
Chatterbox Flash TTS engine — bench contract adapter.

en -> ResembleAI/chatterbox-flash (block-diffusion, parallel unmasking, English-only)
pt -> 400 (the model is English-only)
"""
import io
import re
import wave
from pathlib import Path
from typing import Optional, Generator

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from chatterbox_flash import ChatterboxFlashTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000
VOICES_DIR = Path("voices")
# Chatterbox Flash has no builtin voice (no conds.pt in the release) — every
# generation clones from a reference clip, so a default is mandatory.
DEFAULT_VOICE = "en_female.flac"
DEFAULT_EXAGGERATION = 0.5

app = FastAPI(title="Chatterbox Flash TTS Engine")
models = {}
# prepare_conditionals costs a librosa load + embed_ref per call; cache per
# (voice, exaggeration) so repeated requests don't pay it again.
conds_cache = {}


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3
    exaggeration: Optional[float] = None
    temperature: Optional[float] = None
    num_steps: Optional[int] = None


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def get_voice_path(voice: Optional[str]) -> str:
    voice = voice or DEFAULT_VOICE
    for candidate in [VOICES_DIR / voice] + [VOICES_DIR / f"{voice}{ext}" for ext in (".flac", ".wav", ".mp3")]:
        if candidate.exists():
            return str(candidate)
    raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")


def get_conds(voice_path: str, exaggeration: float):
    key = (voice_path, exaggeration)
    if key not in conds_cache:
        conds_cache[key] = models["en"].prepare_conditionals(voice_path, exaggeration=exaggeration)
    return conds_cache[key]


def generate_speech(text, voice_path, exaggeration=None, temperature=None, num_steps=None) -> np.ndarray:
    print(f"[en] {text}")
    tts = models["en"]
    tts.conds = get_conds(voice_path, exaggeration if exaggeration is not None else DEFAULT_EXAGGERATION)
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if num_steps is not None:
        kwargs["num_steps"] = num_steps
    wav = tts.generate(text, **kwargs)

    audio = wav.cpu().numpy().squeeze()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def check_lang(lang: str):
    if lang == "pt":
        raise HTTPException(
            status_code=400,
            detail="Chatterbox Flash is English-only; 'pt' is not supported by this engine.",
        )
    if lang != "en":
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")


@app.on_event("startup")
def startup():
    print(f"Device: {DEVICE}")
    print("Loading Chatterbox Flash...")
    models["en"] = ChatterboxFlashTTS.from_pretrained(device=DEVICE)
    # Warm the default voice so the first /tts doesn't pay conditioning prep.
    get_conds(get_voice_path(None), DEFAULT_EXAGGERATION)
    print("Chatterbox Flash engine ready")


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
        request.exaggeration, request.temperature, request.num_steps,
    )
    return Response(content=numpy_to_wav_bytes(audio), media_type="audio/wav")


def stream_chunks(request: TTSRequest, voice_path: str) -> Generator[bytes, None, None]:
    chunks = chunk_sentences(split_into_sentences(request.text), request.chunk_size)
    for chunk in chunks:
        audio = generate_speech(
            chunk, voice_path,
            request.exaggeration, request.temperature, request.num_steps,
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
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(SAMPLE_RATE)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
