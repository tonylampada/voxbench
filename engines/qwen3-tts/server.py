#!/usr/bin/env python3
"""Qwen3-TTS adapter — 9 built-in speakers, speaker-id synthesis only (the
CustomVoice checkpoint cannot clone). Serves the bench contract (docs/contract.md)."""
import io
import json
import os
import re
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
LANG_MAP = {"en": "English", "pt": "Portuguese"}
DEFAULT_LANG = "en"
DEFAULT_SPEAKER = "ryan"
# Gender per the Qwen3-TTS voice descriptions; every speaker is multilingual
# (one model, language picked per request), so each lists both served langs.
GENDER = {"aiden": "male", "dylan": "male", "eric": "male", "ono_anna": "female",
          "ryan": "male", "serena": "female", "sohee": "female",
          "uncle_fu": "male", "vivian": "female"}

PARAMS = {
    "instruct": {"type": "string", "default": "",
                 "description": "tone/emotion instruction passed to the model, e.g. 'Very happy.'"},
}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
_manifest_file = next((p for p in (Path("/app/voices/library/manifest.json"),
                                   Path("/app/voices_manifest.json")) if p.exists()), None)
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Qwen3-TTS Adapter")

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
state = {}
# ponytail: single global lock — one GPU, bench is serial; queue if throughput matters
gpu_lock = threading.Lock()


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
def load_model():
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_ID} ({dtype}) ...")
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=dtype)
    state["model"] = model
    state["speakers"] = sorted(model.get_supported_speakers() or GENDER)
    print(f"Model ready. Speakers: {state['speakers']}")


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> tuple[str, str]:
    """-> (speaker, lang), enforcing the contract's 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANG_MAP:
        bad(f"lang '{lang}' not supported — this engine speaks {sorted(LANG_MAP)}")
    if req.voice is None:
        return DEFAULT_SPEAKER, lang or DEFAULT_LANG
    voice = req.voice
    if voice in REFERENCES:
        entry = REFERENCES[voice]
        if entry["engine"] != "qwen3-tts":
            bad(f"'{voice}' is a cloning reference rendered by {entry['engine']}; "
                f"this CustomVoice checkpoint cannot clone — use a builtin or qwen3-tts-origin voice id")
        if lang is not None and lang != entry["lang"]:
            bad(f"voice '{voice}' speaks '{entry['lang']}', not '{lang}'")
        return entry["voice"], entry["lang"]
    if voice not in state["speakers"]:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    return voice, lang or DEFAULT_LANG  # every builtin speaker is multilingual


def effective_instruct(req: SpeechRequest) -> str:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    for k in req.params:
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
    instruct = req.params.get("instruct", "")
    if not isinstance(instruct, str):
        bad("params.instruct must be a string")
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    return instruct


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


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
def health():
    if "model" not in state:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "qwen3-tts"}


@app.get("/v1/voices")
def voices():
    langs = sorted(LANG_MAP)
    out = [{"id": s, "name": s.replace("_", " ").title(), "kind": "builtin",
            "langs": langs, "gender": GENDER.get(s, "unknown")}
           for s in state.get("speakers", [])]
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
    speaker, lang = resolve(req)
    instruct = effective_instruct(req)
    if req.stream:
        bad("this engine cannot stream — qwen-tts 0.1.1 exposes no streaming output API; "
            "use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    with gpu_lock:
        wavs, sr = state["model"].generate_custom_voice(
            text=req.input, language=LANG_MAP[lang], speaker=speaker, instruct=instruct
        )
    return Response(content=to_wav_bytes(wavs[0], sr), media_type="audio/wav")
