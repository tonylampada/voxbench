#!/usr/bin/env python3
"""OmniVoice pt-BR adapter — edwixx/omnivoice-brpt-v15 finetune of k2-fsa/OmniVoice."""
import io
import re
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from omnivoice import OmniVoice
from pydantic import BaseModel

MODEL_ID = "edwixx/omnivoice-brpt-v15"
VOICES_DIR = Path("/app/voices")
# The refine data was trained with this instruction (per the model card); used
# for pt requests with no voice so auto-mode output matches the finetune's setup.
PT_INSTRUCT = "portuguese accent"

model = None
prompts = {}  # voice file name -> VoiceClonePrompt (first use auto-loads Whisper to transcribe the clip)
gen_lock = threading.Lock()  # endpoints are sync (threadpool); serialize GPU work

app = FastAPI(title="OmniVoice pt-BR TTS Server")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3  # sentences per chunk (stream only)
    speed: Optional[float] = None  # >1.0 faster, <1.0 slower
    num_step: Optional[int] = None  # diffusion steps (default 32; 16 is faster)
    instruct: Optional[str] = None  # voice-design attributes, only used when no voice given


@app.on_event("startup")
def load_model():
    global model
    print(f"Loading {MODEL_ID}...")
    # asr_device="cpu": Whisper only transcribes a ~5s reference clip once per
    # voice; on GPU it OOMs when sibling engines fill the card.
    model = OmniVoice.from_pretrained(
        MODEL_ID, device_map="cuda:0", dtype=torch.float16, asr_device="cpu"
    )
    print("Model loaded.")


def list_voice_files() -> list[str]:
    if not VOICES_DIR.exists():
        return []
    return sorted(f.name for f in VOICES_DIR.iterdir() if f.suffix in (".flac", ".wav", ".mp3"))


def resolve_voice(voice: str) -> Path:
    for candidate in (voice, f"{voice}.flac", f"{voice}.wav", f"{voice}.mp3"):
        p = VOICES_DIR / candidate
        if p.is_file():
            return p
    raise HTTPException(status_code=400, detail=f"unknown voice {voice!r}; available: {list_voice_files()}")


def synth(req: TTSRequest, text: str) -> np.ndarray:
    kwargs = {"language": req.lang}
    if req.speed is not None:
        kwargs["speed"] = req.speed
    if req.num_step is not None:
        kwargs["num_step"] = req.num_step
    if req.voice:
        path = resolve_voice(req.voice)
        if req.voice not in prompts:
            prompts[req.voice] = model.create_voice_clone_prompt(ref_audio=str(path))
        kwargs["voice_clone_prompt"] = prompts[req.voice]
    elif req.instruct:
        kwargs["instruct"] = req.instruct
    elif req.lang == "pt":
        kwargs["instruct"] = PT_INSTRUCT
    return model.generate(text=text, **kwargs)[0]


def to_wav_bytes(audio: np.ndarray) -> bytes:
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(model.sampling_rate))
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def check_lang(lang: str):
    if lang not in ("en", "pt"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@app.post("/tts")
def tts(req: TTSRequest):
    check_lang(req.lang)
    with gen_lock:
        wav_bytes = to_wav_bytes(synth(req, req.text))
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    """No native streaming upstream — sentence-chunk fallback, one framed WAV per chunk."""
    check_lang(req.lang)
    sentences = split_into_sentences(req.text) or [req.text]
    chunks = [
        " ".join(sentences[i : i + req.chunk_size])
        for i in range(0, len(sentences), req.chunk_size)
    ]

    def gen():
        for chunk in chunks:
            with gen_lock:
                wav_bytes = to_wav_bytes(synth(req, chunk))
            yield len(wav_bytes).to_bytes(4, "big") + wav_bytes

    return StreamingResponse(gen(), media_type="application/octet-stream")


@app.get("/voices")
def voices():
    return {"voices": list_voice_files()}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return {"status": "ok", "models_loaded": [MODEL_ID], "queue_size": 0}
