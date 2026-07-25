#!/usr/bin/env python3
"""Qwen3-TTS adapter for the TTS bench contract (see repo root server.py)."""
import io
import os
import re
import threading
import wave
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
LANG_MAP = {"en": "English", "pt": "Portuguese"}
DEFAULT_SPEAKER = "ryan"

app = FastAPI(title="Qwen3-TTS Server")
state = {}
# ponytail: single global lock — one GPU, bench is serial; queue if throughput matters
gpu_lock = threading.Lock()


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    instruct: Optional[str] = None  # engine-specific: tone/emotion instruction, e.g. "Very happy."
    chunk_size: int = 3


@app.on_event("startup")
def load_model():
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_ID} ({dtype}) ...")
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=dtype)
    state["model"] = model
    state["speakers"] = model.get_supported_speakers() or []
    print(f"Model ready. Speakers: {state['speakers']}")


def synthesize(text: str, lang: str, voice: Optional[str], instruct: Optional[str]):
    if lang not in LANG_MAP:
        raise HTTPException(400, f"Unsupported lang {lang!r}; this engine serves {list(LANG_MAP)}")
    speaker = (voice or DEFAULT_SPEAKER).lower()
    if state["speakers"] and speaker not in state["speakers"]:
        raise HTTPException(400, f"Unknown voice {voice!r}; available: {state['speakers']}")
    with gpu_lock:
        wavs, sr = state["model"].generate_custom_voice(
            text=text, language=LANG_MAP[lang], speaker=speaker, instruct=instruct or ""
        )
    return wavs[0], sr


def to_wav_bytes(audio, sr: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok", "models_loaded": [MODEL_ID], "queue_size": 0}


@app.get("/voices")
def voices():
    return {"voices": state.get("speakers", [])}


@app.post("/tts")
def tts(req: TTSRequest):
    audio, sr = synthesize(req.text, req.lang, req.voice, req.instruct)
    return Response(content=to_wav_bytes(audio, sr), media_type="audio/wav")


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    if req.lang not in LANG_MAP:
        raise HTTPException(400, f"Unsupported lang {req.lang!r}; this engine serves {list(LANG_MAP)}")

    # qwen-tts 0.1.1 exposes no streaming *output* API (non_streaming_mode only
    # simulates streaming text input), so we chunk by sentence — see README.
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", req.text.strip()) if s]
    chunks = [" ".join(sentences[i : i + req.chunk_size]) for i in range(0, len(sentences), req.chunk_size)]

    def gen():
        for chunk in chunks:
            audio, sr = synthesize(chunk, req.lang, req.voice, req.instruct)
            wav = to_wav_bytes(audio, sr)
            yield len(wav).to_bytes(4, "big") + wav

    return StreamingResponse(gen(), media_type="application/octet-stream", headers={"X-Audio-Format": "wav"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8877)
