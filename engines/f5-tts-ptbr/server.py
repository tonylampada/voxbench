#!/usr/bin/env python3
"""
F5-TTS pt-BR adapter — serves the bench contract (docs/contract.md).

English  -> base F5-TTS v1 checkpoint (SWivid/F5-TTS).
Portuguese -> community finetune firstpixel/F5-TTS-pt-br (old F5TTS_Base arch,
             stock 2545-token vocab bundled with the f5-tts package).
"""
import io
import json
import re
import wave
from importlib.resources import files
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from huggingface_hub import hf_hub_download
from num2words import num2words
from pydantic import BaseModel

LANGS = ("en", "pt")
DEFAULT_LANG = "en"

# The one builtin: the reference clip shipped inside the f5-tts package.
# F5 clones whatever clip it is given, so this voice speaks both models' langs.
DEFAULT_REF = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."
BUILTINS = {"default": {"name": "F5 bundled reference", "kind": "builtin",
                        "langs": list(LANGS), "gender": "unknown"}}

# F5-TTS cloning needs the reference clip's transcript. Every library clip
# reads the bench seed sentence in its own language (bench/mkvoicelib.py), so
# the transcript of any library reference is known by construction.
SEED_TEXT = {
    "en": "The harbour was quiet that morning, and the light came in sideways across the water. Nothing much happened, which was exactly what everyone had hoped for.",
    "pt": "O porto estava quieto naquela manhã, e a luz entrava de lado sobre a água. Não aconteceu grande coisa, que era exatamente o que todo mundo esperava.",
}

PARAMS = {
    "speed": {"type": "number", "min": 0.3, "max": 2.0, "default": 1.0,
              "description": "speech rate multiplier passed to the flow-matching sampler (1.0 = normal)"},
    "nfe_step": {"type": "integer", "min": 16, "max": 64, "default": 32,
                 "description": "diffusion (NFE) steps; fewer = faster, more = cleaner audio"},
}

# Library manifest: mounted live at /app/voices.
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="F5-TTS pt-BR TTS Server")
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
    speed: Optional[float] = None  # OpenAI-compat alias for params.speed
    stream: bool = False
    params: dict = {}


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> Tuple[str, str, str]:
    """-> (ref_audio_path, ref_text, lang). Never falls back silently: an
    unusable voice is a 400 with the reason, not a swap to the default."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {list(LANGS)}")
    if req.voice is None or req.voice in BUILTINS:
        return DEFAULT_REF, DEFAULT_REF_TEXT, lang or DEFAULT_LANG
    v = req.voice
    if v in REFERENCES:
        e = REFERENCES[v]
        if lang is not None and lang != e["lang"]:
            bad(f"voice '{v}' speaks '{e['lang']}', not '{lang}'")
        if e["lang"] not in LANGS:
            bad(f"voice '{v}' speaks '{e['lang']}' — this engine only speaks en and pt")
        return str(LIBRARY_DIR / e["file"]), SEED_TEXT[e["lang"]], e["lang"]
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
    if req.speed is not None:
        if "speed" in knobs:
            bad("speed given both top-level and in params — pick one place")
        p = PARAMS["speed"]
        if not p["min"] <= req.speed <= p["max"]:
            bad(f"speed {req.speed} out of range [{p['min']}, {p['max']}]")
        knobs["speed"] = float(req.speed)
    return knobs


def normalize_pt(text: str) -> str:
    """The pt-BR finetune has no text normalization: lowercase + expand numbers."""
    def repl(m):
        return num2words(int(m.group()), lang="pt_BR")
    return re.sub(r"\d+", repl, text).lower()


def generate_speech(text: str, lang: str, ref_file: str, ref_text: str,
                    knobs: dict) -> Tuple[np.ndarray, int]:
    if lang == "pt":
        text = normalize_pt(text)
    print(f"[{lang}] {text}")
    wav, sr, _ = models[lang].infer(
        ref_file=ref_file, ref_text=ref_text, gen_text=text, **knobs
    )
    audio = np.asarray(wav).squeeze()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio, sr


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


@app.on_event("startup")
async def startup():
    print("Loading English (base F5-TTS v1) model...")
    from f5_tts.api import F5TTS
    models["en"] = F5TTS()
    print("Loading Portuguese (firstpixel/F5-TTS-pt-br) model...")
    ckpt = hf_hub_download("firstpixel/F5-TTS-pt-br", "pt-br/model_last.safetensors")
    # Finetune predates the v1 arch: must load with the old F5TTS_Base config.
    models["pt"] = F5TTS(model="F5TTS_Base", ckpt_file=ckpt)
    print("F5-TTS pt-BR server ready!")


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
async def health():
    if len(models) < 2:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "f5-tts-ptbr"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": vid, **meta} for vid, meta in BUILTINS.items()]
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
    ref_file, ref_text, lang = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        bad("this engine cannot stream — F5-TTS has no incremental synthesis; use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    audio, sr = generate_speech(req.input, lang, ref_file, ref_text, knobs)
    return Response(content=numpy_to_wav_bytes(audio, sr), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
