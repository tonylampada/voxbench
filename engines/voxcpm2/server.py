#!/usr/bin/env python3
"""
VoxCPM2 TTS engine adapter — same HTTP contract as the root Chatterbox server.
Loads openbmb/VoxCPM2 (48 kHz output) and clones voices from reference clips
in voices/ via the model's basic cloning mode (reference_wav_path).
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
from voxcpm import VoxCPM

app = FastAPI(title="VoxCPM2 TTS Server")

model = None
SAMPLE_RATE = 48000  # overwritten from the loaded model at startup
VOICES_DIR = Path("voices")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3
    cfg_value: Optional[float] = None  # guidance scale (voxcpm default: 2.0)
    inference_timesteps: Optional[int] = None  # diffusion steps (voxcpm default: 10)


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), max(1, chunk_size))]


def resolve_voice(voice: Optional[str]) -> Optional[str]:
    if not voice:
        return None
    candidates = [VOICES_DIR / voice] + [VOICES_DIR / f"{voice}{ext}" for ext in (".flac", ".wav", ".mp3")]
    for p in candidates:
        if p.exists():
            return str(p)
    raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")


def gen_kwargs(req: TTSRequest, voice_path: Optional[str]) -> dict:
    kwargs = {"reference_wav_path": voice_path}
    if req.cfg_value is not None:
        kwargs["cfg_value"] = req.cfg_value
    if req.inference_timesteps is not None:
        kwargs["inference_timesteps"] = req.inference_timesteps
    return kwargs


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def check_lang(lang: str):
    # VoxCPM2 is multilingual (30 languages incl. en and pt) and infers the
    # language from the text itself; the field is only validated here.
    if lang not in ("en", "pt"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")


@app.on_event("startup")
async def startup():
    global model, SAMPLE_RATE
    print("Loading openbmb/VoxCPM2...")
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,  # denoiser needs modelscope + an extra model; reference clips are clean
        optimize=os.environ.get("VOXCPM_OPTIMIZE", "1") == "1",
    )
    SAMPLE_RATE = model.tts_model.sample_rate
    print(f"VoxCPM2 ready, sample rate {SAMPLE_RATE}")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": ["openbmb/VoxCPM2"], "queue_size": 0}


@app.get("/voices")
async def list_voices():
    voices = [f.name for f in VOICES_DIR.iterdir() if f.suffix in (".flac", ".wav", ".mp3")] if VOICES_DIR.exists() else []
    return {"voices": sorted(voices)}


@app.post("/tts")
async def tts(request: TTSRequest):
    check_lang(request.lang)
    voice_path = resolve_voice(request.voice)
    print(f"[{request.lang}] voice={request.voice} {request.text}")
    audio = model.generate(text=request.text, **gen_kwargs(request, voice_path))
    return Response(
        content=numpy_to_wav_bytes(audio, SAMPLE_RATE),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


def stream_chunks(request: TTSRequest, voice_path: Optional[str]) -> Generator[bytes, None, None]:
    # Text is split into chunk_size-sentence pieces for robust generation; each
    # piece runs through native generate_streaming(), and every audio chunk it
    # yields is framed as [4-byte BE length][complete WAV].
    for text_chunk in chunk_sentences(split_into_sentences(request.text), request.chunk_size):
        for audio in model.generate_streaming(text=text_chunk, **gen_kwargs(request, voice_path)):
            wav_bytes = numpy_to_wav_bytes(audio, SAMPLE_RATE)
            yield len(wav_bytes).to_bytes(4, "big") + wav_bytes


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    check_lang(request.lang)
    voice_path = resolve_voice(request.voice)
    print(f"[stream] [{request.lang}] voice={request.voice} {request.text}")
    return StreamingResponse(
        stream_chunks(request, voice_path),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(SAMPLE_RATE)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)  # bound to localhost via docker -p 127.0.0.1:...
