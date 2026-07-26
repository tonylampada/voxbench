#!/usr/bin/env python3
"""Gepard 1.0 TTS adapter — serves the bench contract (docs/contract.md).

Thin wrapper around the vendor's GepardSession (gepard-inference package):
text -> FSQ tokens (Qwen3.5 backbone) -> waveform (NVIDIA NanoCodec, 22.05 kHz).
Language is inferred from the text (model locales: en-US, en-UK, es-MX, pt-BR);
a reference clip only sets the timbre, so this is a cloning engine — any
library clip works via GepardSession's reference tokenizer (no transcript needed).
"""
import io
import json
import os
import re
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from gepard_inference import GepardSession
from gepard_inference.session import SessionConfig

GEPARD_ROOT = Path("/opt/gepard")
REF_DIR = GEPARD_ROOT / "ref_audio"
DEFAULT_VOICE = "audio_en.wav"
LANGS = ["en", "es", "pt"]  # the model card's locales, ISO-ified
# Vendor reference clips baked in the image; language comes from the text, the
# clip is timbre only, so every builtin can speak all supported langs.
BUILTIN_GENDER = {"audio_en": "unknown", "audio_ru": "unknown",
                  "nurisa_en": "female", "ulan_emo": "male"}

PARAMS = {
    "temperature": {"type": "number", "min": 0.05, "max": 2.0, "default": 0.3,
                    "description": "sampling temperature; higher = more varied, less stable delivery"},
    "cfg_scale": {"type": "number", "min": 1.0, "max": 3.0, "default": 1.0,
                  "description": "classifier-free guidance scale (1.0 = single pass, no CFG)"},
}

# Library manifest + clips, mounted live at /app/voices.
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Gepard TTS Server")

# CORS: the console calls this engine from a page it does not serve
# (docs/contract.md, "Cross-origin"). No auth here, so never "*".
CORS_ORIGINS = os.environ.get("TTS_CORS_ORIGINS", "http://127.0.0.1:*,http://localhost:*")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="|".join(re.escape(o.strip()).replace(r"\*", ".*")
                                for o in CORS_ORIGINS.split(",") if o.strip()),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type"],
    expose_headers=["x-sample-rate"],  # raw PCM is unplayable without its rate
)
session: Optional[GepardSession] = None


@app.exception_handler(RequestValidationError)
async def validation_to_400(request, exc):  # the contract has no 422s
    e = exc.errors()[0]
    where = ".".join(str(p) for p in e["loc"] if p != "body")
    return JSONResponse(status_code=400, content={"detail": f"{where or 'body'}: {e['msg']}"})


class SpeechRequest(BaseModel):
    input: str = ""
    voice: Optional[str] = None
    lang: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None  # OpenAI-compat alias; this engine has no speed knob
    stream: bool = False
    params: dict = {}


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
        root=REF_DIR,
    )
    session = GepardSession(cfg).load()
    print("Gepard TTS Server Ready!")


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> Optional[str]:
    """-> reference clip path (None = cached default voice), enforcing the 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {LANGS}")
    if req.voice is None:
        return None
    if req.voice in BUILTIN_GENDER:
        return str(REF_DIR / f"{req.voice}.wav")  # builtins speak every supported lang
    if req.voice in REFERENCES:
        entry = REFERENCES[req.voice]
        if lang is not None and lang != entry["lang"]:
            bad(f"voice '{req.voice}' speaks '{entry['lang']}', not '{lang}'")
        if entry["lang"] not in LANGS:
            bad(f"voice '{req.voice}' speaks '{entry['lang']}', which this engine cannot — see /v1/voices")
        return str(LIBRARY_DIR / entry["file"])
    bad(f"unknown voice '{req.voice}' — see /v1/voices")


def effective_params(req: SpeechRequest) -> dict:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {}
    for k, v in req.params.items():
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        p = PARAMS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad(f"params.{k} must be a number")
        if not p["min"] <= v <= p["max"]:
            bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = float(v)
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    return knobs


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
async def health():
    if session is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "gepard"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": v, "name": v.replace("_", " ").title(), "kind": "builtin",
            "langs": LANGS, "gender": g} for v, g in sorted(BUILTIN_GENDER.items())]
    out += [{"id": vid, "name": e["voice"], "kind": "reference", "origin": e["engine"],
             "langs": [e["lang"]], "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
            for vid, e in sorted(REFERENCES.items())]
    return {"voices": out}


@app.get("/v1/params")
async def params():
    return {"params": PARAMS}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice_path = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        bad("this engine cannot stream — Gepard's native streaming lives in the vLLM "
            "serving path, not the eager one used here; use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    sample_rate, audio = session.synthesize(req.input, reference=voice_path, **knobs)
    return Response(content=numpy_to_wav_bytes(audio, sample_rate),
                    media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
