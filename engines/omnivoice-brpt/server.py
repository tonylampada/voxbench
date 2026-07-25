#!/usr/bin/env python3
"""OmniVoice pt-BR adapter — edwixx/omnivoice-brpt-v15 finetune of k2-fsa/OmniVoice.
Cloning engine; serves the bench contract (docs/contract.md)."""
import io
import json
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

MODEL_ID = "edwixx/omnivoice-brpt-v15"
LANGS = ["en", "pt"]
DEFAULT_LANG = "en"
# The refine data was trained with this instruction (per the model card); used
# for pt requests with no voice so auto-mode output matches the finetune's setup.
PT_INSTRUCT = "portuguese accent"

PARAMS = {
    "speed": {"type": "number", "min": 0.5, "max": 2.0, "default": 1.0,
              "description": "playback speed multiplier (1.0 = normal)"},
    "num_step": {"type": "integer", "min": 4, "max": 64, "default": 32,
                 "description": "diffusion steps; fewer is faster at some quality cost"},
    "instruct": {"type": "string", "default": "",
                 "description": "voice-design attributes (e.g. 'female, low pitch'); only valid without a voice"},
}

# Library manifest + clips, mounted live at /app/voices (baked copy as fallback).
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

model = None
prompts = {}  # voice id -> VoiceClonePrompt (first use auto-loads Whisper to transcribe the clip)
gen_lock = threading.Lock()  # endpoints are sync (threadpool); serialize GPU work

app = FastAPI(title="OmniVoice pt-BR TTS Server")


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
    speed: Optional[float] = None  # OpenAI-compat alias for params.speed
    stream: bool = False
    params: dict = {}


@app.on_event("startup")
def load_model():
    global model
    from omnivoice import OmniVoice
    print(f"Loading {MODEL_ID}...")
    # asr_device="cpu": Whisper only transcribes a ~5s reference clip once per
    # voice; on GPU it OOMs when sibling engines fill the card.
    model = OmniVoice.from_pretrained(
        MODEL_ID, device_map="cuda:0", dtype=torch.float16, asr_device="cpu"
    )
    print("Model loaded.")


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> tuple[Optional[Path], str]:
    """-> (reference clip path or None for auto-voice, lang), enforcing the 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {LANGS}")
    if req.voice is None:
        return None, lang or DEFAULT_LANG
    if req.voice not in REFERENCES:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    entry = REFERENCES[req.voice]
    if lang is not None and lang != entry["lang"]:
        bad(f"voice '{req.voice}' speaks '{entry['lang']}', not '{lang}'")
    if entry["lang"] not in LANGS:
        bad(f"voice '{req.voice}' speaks '{entry['lang']}', which this engine cannot — see /v1/voices")
    return LIBRARY_DIR / entry["file"], entry["lang"]


def effective_params(req: SpeechRequest) -> dict:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {}
    for k, v in req.params.items():
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        p = PARAMS[k]
        if p["type"] == "string":
            if not isinstance(v, str):
                bad(f"params.{k} must be a string")
        elif p["type"] == "integer":
            if isinstance(v, bool) or not isinstance(v, int):
                bad(f"params.{k} must be an integer")
            if not p["min"] <= v <= p["max"]:
                bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                bad(f"params.{k} must be a number")
            if not p["min"] <= v <= p["max"]:
                bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = v
    if req.speed is not None:
        if "speed" in knobs:
            bad("speed given both top-level and in params — pick one place")
        p = PARAMS["speed"]
        if not p["min"] <= req.speed <= p["max"]:
            bad(f"speed {req.speed} out of range [{p['min']}, {p['max']}]")
        knobs["speed"] = float(req.speed)
    return knobs


def to_wav_bytes(audio: np.ndarray) -> bytes:
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(model.sampling_rate))
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
def health():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "omnivoice-brpt"}


@app.get("/v1/voices")
def voices():
    # No builtins: with no voice the model invents one (auto-voice mode) — the
    # catalogue is the cloning library.
    return {"voices": [
        {"id": vid, "name": e["voice"], "kind": "reference", "origin": e["engine"],
         "langs": [e["lang"]], "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
        for vid, e in sorted(REFERENCES.items())]}


@app.get("/v1/params")
def params():
    return {"params": PARAMS}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        bad("input is required and must be non-empty")
    clip, lang = resolve(req)
    knobs = effective_params(req)
    instruct = knobs.pop("instruct", "")
    if instruct and clip is not None:
        bad("params.instruct only applies in auto-voice mode — omit it when a voice is given")
    if req.stream:
        bad("this engine cannot stream — OmniVoice has no incremental synthesis API; "
            "use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    kwargs = {"language": lang, **knobs}
    if instruct:
        kwargs["instruct"] = instruct
    elif clip is None and lang == "pt":
        kwargs["instruct"] = PT_INSTRUCT
    with gen_lock:
        if clip is not None:
            if req.voice not in prompts:
                prompts[req.voice] = model.create_voice_clone_prompt(ref_audio=str(clip))
            kwargs["voice_clone_prompt"] = prompts[req.voice]
        audio = model.generate(text=req.input, **kwargs)[0]
    return Response(content=to_wav_bytes(audio), media_type="audio/wav")
