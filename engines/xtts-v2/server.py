#!/usr/bin/env python3
"""XTTS-v2 adapter — Coqui's multilingual voice-cloning model (idiap fork), GPU."""
import io
import os
import re
import wave
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICES_DIR = "/voices"  # optional mount of reference clips for cloning
DEFAULT_SPEAKER = "Ana Florence"  # XTTS built-in studio speaker
LANGS = {"en", "pt"}

model = None
SAMPLE_RATE = 24000

app = FastAPI(title="XTTS-v2 TTS Server")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3  # sentences per chunk (stream only)


@app.on_event("startup")
def load_model():
    global model, SAMPLE_RATE
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = TTS(MODEL_NAME).to(device)
    SAMPLE_RATE = model.synthesizer.output_sample_rate
    print(f"Model loaded. {len(model.speakers)} built-in speakers, sr={SAMPLE_RATE}")


def resolve_voice(voice: Optional[str]) -> dict:
    """Map the contract's voice field to XTTS kwargs: built-in speaker name,
    or a reference clip under /voices for cloning."""
    if not voice:
        return {"speaker": DEFAULT_SPEAKER}
    if voice in model.speakers:
        return {"speaker": voice}
    for cand in (voice, f"{voice}.flac", f"{voice}.wav", f"{voice}.mp3"):
        path = os.path.join(VOICES_DIR, cand)
        if os.path.isfile(path):
            return {"speaker_wav": path}
    raise HTTPException(
        status_code=400,
        detail=f"unknown voice {voice!r}; use a built-in speaker from /voices "
        f"or mount a reference clip at {VOICES_DIR}",
    )


def synth_wav(text: str, lang: str, voice_kwargs: dict) -> bytes:
    audio = np.array(model.tts(text=text, language=lang, **voice_kwargs))
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def check_lang(lang: str):
    if lang not in LANGS:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")


@app.post("/tts")
def tts(req: TTSRequest):
    check_lang(req.lang)
    wav_bytes = synth_wav(req.text, req.lang, resolve_voice(req.voice))
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    check_lang(req.lang)
    voice_kwargs = resolve_voice(req.voice)
    sentences = split_into_sentences(req.text) or [req.text]
    chunks = [
        " ".join(sentences[i : i + req.chunk_size])
        for i in range(0, len(sentences), req.chunk_size)
    ]

    def gen():
        for chunk in chunks:
            wav_bytes = synth_wav(chunk, req.lang, voice_kwargs)
            yield len(wav_bytes).to_bytes(4, "big") + wav_bytes

    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.get("/voices")
def list_voices():
    return {"voices": sorted(model.speakers) if model else []}


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return {"status": "ok", "engine": "xtts-v2", "models_loaded": [MODEL_NAME], "queue_size": 0}
