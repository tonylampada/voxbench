#!/usr/bin/env python3
"""
Chatterbox Nano TTS engine — bench contract adapter (docs/contract.md).

ResembleAI/chatterbox-nano (110M, English only), loaded through
ChatterboxTurboTTS with nano=True. Device via NANO_DEVICE=cpu|cuda.
"""
import io
import json
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = os.getenv("NANO_DEVICE", "cuda")

BUILTINS = {
    "en-default": {"name": "Nano default", "kind": "builtin", "langs": ["en"], "gender": "unknown"},
}

PARAMS = {
    "temperature": {"type": "number", "min": 0.05, "max": 5.0, "default": 0.8,
                    "description": "sampling temperature; higher = more varied, less stable delivery"},
    "top_p": {"type": "number", "min": 0.1, "max": 1.0, "default": 0.95,
              "description": "nucleus sampling cutoff; lower = safer, more monotone speech"},
    "repetition_penalty": {"type": "number", "min": 1.0, "max": 3.0, "default": 1.2,
                           "description": "penalty against repeating speech tokens; raise it if output stutters or loops"},
}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Chatterbox Nano TTS Engine")
models = {}


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


def resolve(req: SpeechRequest) -> Optional[str]:
    """-> audio prompt path, or None for the builtin default voice."""
    lang = req.lang
    if lang is not None and lang != "en":
        bad(f"lang '{lang}' not supported — Chatterbox Nano is English-only")
    if req.voice is None or req.voice in BUILTINS:
        return None
    if req.voice not in REFERENCES:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    e = REFERENCES[req.voice]
    if lang is not None and lang != e["lang"]:
        bad(f"voice '{req.voice}' speaks '{e['lang']}', not '{lang}'")
    if e["lang"] != "en":
        bad(f"voice '{req.voice}' speaks '{e['lang']}' — Chatterbox Nano is English-only")
    return str(LIBRARY_DIR / e["file"])


def effective_params(req: SpeechRequest) -> dict:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {}
    for k, v in req.params.items():
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        p = PARAMS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad(f"params.{k} must be a {p['type']}")
        if not p["min"] <= v <= p["max"]:
            bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = v
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    return knobs


def generate_speech(text: str, voice_path: Optional[str], knobs: dict) -> np.ndarray:
    print(f"[en] {text}")
    wav = models["en"].generate(text, audio_prompt_path=voice_path, **knobs)

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


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
def health():
    if "en" not in models:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "chatterbox-nano"}


@app.get("/v1/voices")
def voices():
    out = [{"id": vid, **meta} for vid, meta in BUILTINS.items()]
    out += [{"id": vid, "name": e["voice"], "kind": "reference", "origin": e["engine"],
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
        bad("this engine cannot stream — chatterbox-nano synthesizes the whole utterance at once; use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    audio = generate_speech(req.input, voice_path, knobs)
    return Response(content=numpy_to_wav_bytes(audio), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
