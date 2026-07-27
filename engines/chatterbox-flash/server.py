#!/usr/bin/env python3
"""
Chatterbox Flash TTS engine — bench contract adapter (docs/contract.md).

ResembleAI/chatterbox-flash (block-diffusion, parallel unmasking, English-only).
No builtin voice (no conds.pt in the release) — every generation clones from a
reference clip, so the default voice is a library reference.
"""
import io
import json
import os
import re
import threading
import wave
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from chatterbox_flash import ChatterboxFlashTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000
DEFAULT_VOICE = "kokoro-af-heart-en"
DEFAULT_EXAGGERATION = 0.5

PARAMS = {
    "exaggeration": {"type": "number", "min": 0.0, "max": 2.0, "default": 0.5,
                     "description": "emotion intensity of the delivery (0 = flat, 0.5 = neutral, 2 = theatrical)"},
    "temperature": {"type": "number", "min": 0.05, "max": 2.0, "default": 0.6,
                    "description": "sampling temperature; higher = more varied, less stable delivery"},
    "num_steps": {"type": "integer", "min": 1, "max": 50, "default": 10,
                  "description": "diffusion unmasking steps per block; more = better quality, slower"},
}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

# One generation at a time: the endpoint is sync, so FastAPI runs each request
# in its own threadpool thread, all on the same model — concurrent decodes
# corrupt each other's state on the GPU. Latecomers wait.
gen_lock = threading.Lock()

app = FastAPI(title="Chatterbox Flash TTS Engine")

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
models = {}
# prepare_conditionals costs a librosa load + embed_ref per call; cache per
# (voice, exaggeration) so repeated requests don't pay it again.
conds_cache = {}


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


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> str:
    """-> audio prompt path (this engine always clones)."""
    lang = req.lang
    if lang is not None and lang != "en":
        bad(f"lang '{lang}' not supported — Chatterbox Flash is English-only")
    if req.voice is None:
        return str(LIBRARY_DIR / REFERENCES[DEFAULT_VOICE]["file"])
    if req.voice not in REFERENCES:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    e = REFERENCES[req.voice]
    if lang is not None and lang != e["lang"]:
        bad(f"voice '{req.voice}' speaks '{e['lang']}', not '{lang}'")
    if e["lang"] != "en":
        bad(f"voice '{req.voice}' speaks '{e['lang']}' — Chatterbox Flash is English-only")
    return str(LIBRARY_DIR / e["file"])


def effective_params(req: SpeechRequest) -> dict:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {}
    for k, v in req.params.items():
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        p = PARAMS[k]
        want = int if p["type"] == "integer" else (int, float)
        if isinstance(v, bool) or not isinstance(v, want):
            bad(f"params.{k} must be a {p['type']}")
        if not p["min"] <= v <= p["max"]:
            bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = v
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    return knobs


def get_conds(voice_path: str, exaggeration: float):
    key = (voice_path, exaggeration)
    if key not in conds_cache:
        conds_cache[key] = models["en"].prepare_conditionals(voice_path, exaggeration=exaggeration)
    return conds_cache[key]


def generate_speech(text: str, voice_path: str, knobs: dict) -> np.ndarray:
    print(f"[en] {text}")
    tts = models["en"]
    tts.conds = get_conds(voice_path, knobs.pop("exaggeration", DEFAULT_EXAGGERATION))
    wav = tts.generate(text, **knobs)

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


@app.on_event("startup")
def startup():
    print(f"Device: {DEVICE}")
    print("Loading Chatterbox Flash...")
    models["en"] = ChatterboxFlashTTS.from_pretrained(device=DEVICE)
    # Warm the default voice so the first request doesn't pay conditioning prep.
    get_conds(str(LIBRARY_DIR / REFERENCES[DEFAULT_VOICE]["file"]), DEFAULT_EXAGGERATION)
    print("Chatterbox Flash engine ready")


@app.get("/health")
def health():
    if "en" not in models:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "chatterbox-flash"}


@app.get("/v1/voices")
def voices():
    out = [{"id": vid, "name": e["voice"], "kind": "reference", "origin": e["engine"],
            "langs": [e["lang"]], "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
           for vid, e in sorted(REFERENCES.items())]
    return {"voices": out}


@app.get("/v1/params")
def params():
    return {"params": PARAMS}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice_path = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        bad("this engine cannot stream — chatterbox-flash synthesizes the whole utterance at once; use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    with gen_lock:
        audio = generate_speech(req.input, voice_path, knobs)
    return Response(content=numpy_to_wav_bytes(audio), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
