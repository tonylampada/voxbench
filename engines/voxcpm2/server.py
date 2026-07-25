#!/usr/bin/env python3
"""
VoxCPM2 TTS engine adapter — serves the bench contract (docs/contract.md).
Loads openbmb/VoxCPM2 (48 kHz output) and clones voices from library reference
clips via the model's basic cloning mode (reference_wav_path).
"""
import io
import json
import os
import re
import wave
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from voxcpm import VoxCPM

# VoxCPM2's 30 supported languages (model card), as ISO 639-1. The model
# infers the language from the text itself; `lang` is validated, not passed.
LANGS = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "id", "it",
         "ja", "km", "ko", "lo", "ms", "my", "nl", "no", "pl", "pt", "ru", "sv",
         "sw", "th", "tl", "tr", "vi", "zh"}

PARAMS = {
    "cfg_value": {"type": "number", "min": 1.0, "max": 4.0, "default": 2.0,
                  "description": "guidance scale on the diffusion head; higher = closer to the reference, may degrade quality"},
    "inference_timesteps": {"type": "integer", "min": 4, "max": 50, "default": 10,
                            "description": "diffusion sampling steps; more = cleaner audio, fewer = faster"},
}

# No builtins: VoxCPM2 has no fixed factory voices — with no reference it
# invents an arbitrary voice per request. The catalogue is the library.
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="VoxCPM2 TTS Server")
model = None
SAMPLE_RATE = 48000  # overwritten from the loaded model at startup


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
    """-> reference clip path, or None for the model's own (arbitrary) voice."""
    if req.lang is not None and req.lang not in LANGS:
        bad(f"lang '{req.lang}' not supported — this engine speaks {sorted(LANGS)}")
    if req.voice is None:
        return None
    v = req.voice
    if v in REFERENCES:  # cloning engine: any library clip works
        e = REFERENCES[v]
        if req.lang is not None and req.lang != e["lang"]:
            bad(f"voice '{v}' speaks '{e['lang']}', not '{req.lang}'")
        return str(LIBRARY_DIR / e["file"])
    bad(f"unknown voice '{v}' — see /v1/voices")


def effective_params(req: SpeechRequest) -> dict:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {}
    for k, v in req.params.items():
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        p = PARAMS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or \
                (p["type"] == "integer" and not isinstance(v, int)):
            bad(f"params.{k} must be a {p['type']}")
        if not p["min"] <= v <= p["max"]:
            bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = v
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    return knobs


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()] if text else []


def chunk_sentences(sentences: list[str], size: int = 3) -> list[str]:
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), size)]


def to_int16_bytes(audio: np.ndarray) -> bytes:
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak
    return (audio * 32767).astype(np.int16).tobytes()


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(to_int16_bytes(audio))
    return buffer.getvalue()


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


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
async def health():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "voxcpm2"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": vid, "name": e["voice"], "kind": "reference", "origin": e["engine"],
            "langs": [e["lang"]], "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
           for vid, e in sorted(REFERENCES.items())]
    return {"voices": out}


@app.get("/v1/params")
async def params():
    return {"params": PARAMS}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice_path = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")
        # Native incremental streaming: text is split into 3-sentence pieces
        # for robust generation; each piece runs through generate_streaming(),
        # and every chunk the model natively yields goes out as raw int16 PCM.
        pieces = chunk_sentences(split_into_sentences(req.input)) or [req.input]

        def gen() -> Generator[bytes, None, None]:
            for piece in pieces:
                for audio in model.generate_streaming(
                        text=piece, reference_wav_path=voice_path, **knobs):
                    yield to_int16_bytes(audio)

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"x-sample-rate": str(SAMPLE_RATE)})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    audio = model.generate(text=req.input, reference_wav_path=voice_path, **knobs)
    return Response(content=numpy_to_wav_bytes(audio, SAMPLE_RATE), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)  # bound to localhost via docker -p 127.0.0.1:...
