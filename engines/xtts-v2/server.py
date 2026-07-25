#!/usr/bin/env python3
"""XTTS-v2 adapter — Coqui's multilingual voice-cloning model (idiap fork), GPU.
Serves the bench contract (docs/contract.md)."""
import io
import json
import re
import unicodedata
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_SPEAKER = "Ana Florence"  # XTTS built-in studio speaker
SAMPLE_RATE = 24000
DEFAULT_LANG = "en"

# XTTS names Chinese "zh-cn"; the contract wants ISO 639-1.
ISO_TO_XTTS = {"zh": "zh-cn"}
XTTS_TO_ISO = {v: k for k, v in ISO_TO_XTTS.items()}

PARAMS = {
    "speed": {"type": "number", "min": 0.5, "max": 2.0, "default": 1.0,
              "description": "playback speed multiplier (1.0 = normal)"},
    "temperature": {"type": "number", "min": 0.05, "max": 2.0, "default": 0.75,
                    "description": "GPT sampling temperature; higher = more varied, less stable delivery"},
}

# Library manifest, mounted live at /voices (this engine's voicesroot mount).
LIBRARY_DIR = Path("/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

model = None
SPEAKERS: dict[str, str] = {}  # slug id -> XTTS speaker name
LANGS: set[str] = set()        # ISO 639-1 codes the model speaks

app = FastAPI(title="XTTS-v2 TTS Server")


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


def slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


@app.on_event("startup")
def load_model():
    global model, SAMPLE_RATE, SPEAKERS, LANGS
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    m = TTS(MODEL_NAME).to(device)
    SAMPLE_RATE = m.synthesizer.output_sample_rate
    SPEAKERS = {slug(s): s for s in m.speakers}
    # The model's own language list, not a guess (XTTS-v2: 17 languages).
    LANGS = {XTTS_TO_ISO.get(l, l) for l in m.synthesizer.tts_model.language_manager.language_names}
    model = m
    print(f"Model loaded. {len(SPEAKERS)} built-in speakers, langs={sorted(LANGS)}, sr={SAMPLE_RATE}")


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> tuple[dict, str]:
    """-> (XTTS voice kwargs, xtts language code), enforcing the 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {sorted(LANGS)}")
    if req.voice is None:
        return {"speaker": DEFAULT_SPEAKER}, ISO_TO_XTTS.get(lang or DEFAULT_LANG, lang or DEFAULT_LANG)
    v = req.voice
    if v in SPEAKERS:  # builtin studio speakers are multilingual — any supported lang
        return {"speaker": SPEAKERS[v]}, ISO_TO_XTTS.get(lang or DEFAULT_LANG, lang or DEFAULT_LANG)
    if v in REFERENCES:  # cloning engine: any library clip works, in the clip's language
        e = REFERENCES[v]
        if lang is not None and lang != e["lang"]:
            bad(f"voice '{v}' speaks '{e['lang']}', not '{lang}'")
        if e["lang"] not in LANGS:
            bad(f"voice '{v}' speaks '{e['lang']}', which this engine cannot — see /v1/voices")
        return {"speaker_wav": str(LIBRARY_DIR / e["file"])}, ISO_TO_XTTS.get(e["lang"], e["lang"])
    bad(f"unknown voice '{v}' — see /v1/voices")


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
        knobs[k] = float(v)
    if req.speed is not None:
        if "speed" in knobs:
            bad("speed given both top-level and in params — pick one place")
        p = PARAMS["speed"]
        if not p["min"] <= req.speed <= p["max"]:
            bad(f"speed {req.speed} out of range [{p['min']}, {p['max']}]")
        knobs["speed"] = float(req.speed)
    return knobs


def synth_wav(text: str, xtts_lang: str, voice_kwargs: dict, knobs: dict) -> bytes:
    audio = np.array(model.tts(text=text, language=xtts_lang, **voice_kwargs, **knobs))
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
def health():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "xtts-v2"}


@app.get("/v1/voices")
def voices():
    langs = sorted(LANGS)
    out = [{"id": sid, "name": name, "kind": "builtin", "langs": langs, "gender": "unknown"}
           for sid, name in sorted(SPEAKERS.items())]
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
    voice_kwargs, xtts_lang = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        bad("this engine cannot stream — XTTS synthesizes whole utterances through the "
            "coqui TTS api; use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    return Response(content=synth_wav(req.input, xtts_lang, voice_kwargs, knobs),
                    media_type="audio/wav")
