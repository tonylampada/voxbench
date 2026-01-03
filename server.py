#!/usr/bin/env python3
"""
Chatterbox TTS Server - FastAPI server with preloaded models.
Supports English (Turbo) and Portuguese (Multilingual) with voice cloning.
"""
import io
import os
import re
import wave
import queue
import threading
from pathlib import Path
from typing import Optional, Generator

from dotenv import load_dotenv
load_dotenv()

import torch
import sounddevice as sd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# Patch torch.load permanently for CPU loading (workaround for chatterbox bug)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI(title="Chatterbox TTS Server")

# Global state
models = {}
audio_queue = queue.Queue()
SAMPLE_RATE = 24000
VOICES_DIR = Path("voices")


class SpeakRequest(BaseModel):
    text: str
    lang: str = "en"  # "en" or "pt"
    voice: Optional[str] = None  # voice file name (e.g., "pt_male.flac")
    split_sentences: bool = True


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def audio_player_worker():
    """Worker thread that plays audio from the queue."""
    while True:
        item = audio_queue.get()
        if item is None:
            break
        wav, sr = item
        sd.play(wav, sr)
        sd.wait()
        audio_queue.task_done()


def get_voice_path(voice: Optional[str]) -> Optional[str]:
    """Resolve voice name to full path."""
    if not voice:
        return None

    voice_path = VOICES_DIR / voice
    if voice_path.exists():
        return str(voice_path)

    for ext in [".flac", ".wav", ".mp3"]:
        p = VOICES_DIR / f"{voice}{ext}"
        if p.exists():
            return str(p)

    if Path(voice).exists():
        return voice

    return None


def generate_speech(text: str, lang: str, voice: Optional[str] = None) -> np.ndarray:
    """Generate speech for given text."""
    voice_path = get_voice_path(voice)

    if lang == "en":
        model = models["en"]
        wav = model.generate(text, audio_prompt_path=voice_path)
    elif lang == "pt":
        model = models["pt"]
        wav = model.generate(text, audio_prompt_path=voice_path, language_id="pt")
    else:
        raise ValueError(f"Unsupported language: {lang}")

    return wav.cpu().numpy().squeeze()


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    audio_int16 = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()


def generate_audio_stream(text: str, lang: str, voice: Optional[str] = None) -> Generator[bytes, None, None]:
    """Generate audio chunks for each sentence, yielding WAV bytes."""
    sentences = split_into_sentences(text)

    for i, sentence in enumerate(sentences):
        if not sentence:
            continue

        # Generate audio for this sentence
        audio = generate_speech(sentence, lang, voice)
        wav_bytes = numpy_to_wav_bytes(audio)

        # Yield chunk with length prefix (4 bytes) + WAV data
        # This allows client to know where each chunk ends
        chunk_len = len(wav_bytes).to_bytes(4, 'big')
        yield chunk_len + wav_bytes


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    print("Loading English (Turbo) model...")
    models["en"] = ChatterboxTurboTTS.from_pretrained(device="cpu")
    print("English model loaded!")

    print("Loading Portuguese (Multilingual) model...")
    models["pt"] = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    print("Portuguese model loaded!")

    # Start audio player thread
    player_thread = threading.Thread(target=audio_player_worker, daemon=True)
    player_thread.start()
    print("Audio player started!")

    print("\n" + "="*50)
    print("Chatterbox TTS Server Ready!")
    print("="*50 + "\n")


# =============================================================================
# /speak - Play audio locally (original endpoint)
# =============================================================================

@app.post("/speak")
async def speak(request: SpeakRequest):
    """Generate and play speech locally."""
    if request.lang not in ["en", "pt"]:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")

    if request.split_sentences:
        sentences = split_into_sentences(request.text)
    else:
        sentences = [request.text]

    for sentence in sentences:
        if not sentence:
            continue
        wav = generate_speech(sentence, request.lang, request.voice)
        audio_queue.put((wav, SAMPLE_RATE))

    return {
        "status": "queued",
        "sentences": len(sentences),
        "lang": request.lang,
        "voice": request.voice
    }


# =============================================================================
# /tts - Return binary WAV audio
# =============================================================================

@app.post("/tts")
async def tts(request: TTSRequest):
    """Generate speech and return as WAV binary."""
    if request.lang not in ["en", "pt"]:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")

    audio = generate_speech(request.text, request.lang, request.voice)
    wav_bytes = numpy_to_wav_bytes(audio)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )


@app.get("/tts")
async def tts_get(
    text: str = Query(..., description="Text to synthesize"),
    lang: str = Query("en", description="Language: 'en' or 'pt'"),
    voice: Optional[str] = Query(None, description="Voice file name")
):
    """Generate speech and return as WAV binary (GET version)."""
    if lang not in ["en", "pt"]:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")

    audio = generate_speech(text, lang, voice)
    wav_bytes = numpy_to_wav_bytes(audio)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )


# =============================================================================
# /tts/stream - Stream audio chunks (one per sentence)
# =============================================================================

@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """
    Generate speech and stream WAV chunks (one per sentence).

    Each chunk is prefixed with 4 bytes (big-endian) indicating chunk length.
    Format: [4 bytes length][WAV data][4 bytes length][WAV data]...
    """
    if request.lang not in ["en", "pt"]:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")

    return StreamingResponse(
        generate_audio_stream(request.text, request.lang, request.voice),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(SAMPLE_RATE)}
    )


@app.get("/tts/stream")
async def tts_stream_get(
    text: str = Query(..., description="Text to synthesize"),
    lang: str = Query("en", description="Language: 'en' or 'pt'"),
    voice: Optional[str] = Query(None, description="Voice file name")
):
    """
    Generate speech and stream WAV chunks (one per sentence) - GET version.

    Each chunk is prefixed with 4 bytes (big-endian) indicating chunk length.
    """
    if lang not in ["en", "pt"]:
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")

    return StreamingResponse(
        generate_audio_stream(text, lang, voice),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(SAMPLE_RATE)}
    )


# =============================================================================
# Utility endpoints
# =============================================================================

@app.get("/voices")
async def list_voices():
    """List available voices."""
    voices = []
    if VOICES_DIR.exists():
        for f in VOICES_DIR.iterdir():
            if f.suffix in [".flac", ".wav", ".mp3"]:
                voices.append(f.name)
    return {"voices": sorted(voices)}


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "queue_size": audio_queue.qsize()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
