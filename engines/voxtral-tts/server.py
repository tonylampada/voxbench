#!/usr/bin/env python3
"""
Voxtral TTS adapter — serves the bench contract (docs/contract.md) by
translating it to the vLLM-Omni OpenAI-compatible /v1/audio/speech API
running on localhost:8000 in the same container. Upstream already speaks
the contract's shape, so this adapter mostly validates and proxies —
including native PCM streaming, passed through untouched.

The upstream server (vllm serve --omni) is launched as a subprocess at
startup; /health answers 200 only once it accepts requests, which happens
only after the 4B model is loaded and both pipeline stages are up.

The PUBLIC checkpoint cannot clone: ref_audio needs audio-encoder weights
Mistral did not ship, and worse, a forwarded ref_audio request raises inside
the EngineCore and kills the whole upstream engine (every later request 500s
until the container restarts). So this adapter never forwards ref_audio —
reference voices resolve back to the builtin preset when their origin is
this engine, and are refused with a 400 otherwise.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

UPSTREAM = "http://127.0.0.1:8000"
MODEL = os.environ.get("VOXTRAL_MODEL", "mistralai/Voxtral-4B-TTS-2603")
VOICES_DIR = Path("/app/voices")
SAMPLE_RATE = 24000

# The 9 languages Mistral claims for this model. The model has no lang
# parameter upstream — language comes from the text itself — so this list is
# a validation gate, not a routing knob. en and pt verified on this box;
# the rest are accepted on the vendor's word.
LANGS = ["ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"]

# The 20 preset speakers baked into the checkpoint (vllm-omni recipe,
# confirmed against upstream /v1/audio/voices at runtime). Gender is in the
# name; every preset speaks whatever language the input text is in, so each
# lists all of LANGS.
PRESETS = [
    "ar_male", "casual_female", "casual_male", "cheerful_female",
    "de_female", "de_male", "es_female", "es_male", "fr_female", "fr_male",
    "hi_female", "hi_male", "it_female", "it_male",
    "neutral_female", "neutral_male", "nl_female", "nl_male",
    "pt_female", "pt_male",
]
PRESET_LANG_NAMES = {"ar": "Arabic", "de": "German", "es": "Spanish",
                     "fr": "French", "hi": "Hindi", "it": "Italian",
                     "nl": "Dutch", "pt": "Portuguese"}


def preset_name(pid: str) -> str:
    prefix, _, gender = pid.partition("_")
    return f"{PRESET_LANG_NAMES.get(prefix, prefix.capitalize())} {gender.capitalize()}"


def default_voice(lang: str) -> str:
    if lang == "ar":
        return "ar_male"  # the one preset language with no female voice
    return {"en": "neutral_female"}.get(lang, f"{lang}_female")


# Knob → where it goes upstream: temperature/top_p/top_k ride extra_params
# (vllm-omni applies them to the token stage's SamplingParams); seed and
# max_new_tokens are top-level request fields. Defaults mirror the deploy
# config vllm_omni/deploy/voxtral_tts.yaml.
PARAMS = {
    "temperature": {"type": "number", "min": 0.0, "max": 2.0, "default": 0.0,
                    "description": "sampling temperature of the audio-token stage; 0 = the stable default take, higher = more varied deliveries"},
    "top_p": {"type": "number", "min": 0.0, "max": 1.0, "default": 1.0,
              "description": "nucleus sampling cutoff (only matters when temperature > 0)"},
    "top_k": {"type": "integer", "min": -1, "max": 200, "default": -1,
              "description": "top-k sampling cutoff (-1 = disabled)"},
    "seed": {"type": "integer", "min": 0, "max": 2147483647, "default": 42,
             "description": "random seed; same text + same knobs + same seed reproduces the same take"},
    "max_new_tokens": {"type": "integer", "min": 1, "max": 2048, "default": 2048,
                       "description": "cap on generated audio tokens (~12.5 per second of audio) — a length ceiling, not a target"},
}

# Library manifest + clips, mounted live at /app/voices.
LIBRARY_DIR = VOICES_DIR / "library"
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Voxtral TTS Adapter")

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


def start_upstream() -> None:
    subprocess.Popen([
        "vllm", "serve", MODEL,
        "--omni",
        "--port", "8000",
        "--host", "127.0.0.1",
        "--trust-remote-code",
    ], stdout=sys.stdout, stderr=sys.stderr)


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve_voice(req: SpeechRequest) -> dict:
    """-> upstream fields for the chosen voice, enforcing the 400 catalogue.
    Always a preset: this engine cannot clone (see module docstring)."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {LANGS}")
    if req.voice is None:
        return {"voice": default_voice(lang or "en")}
    if req.voice in PRESETS:
        return {"voice": req.voice}  # any preset speaks any supported lang
    if req.voice not in REFERENCES:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    entry = REFERENCES[req.voice]
    if lang is not None and lang != entry["lang"]:
        bad(f"voice '{req.voice}' speaks '{entry['lang']}', not '{lang}'")
    if entry["engine"] != "voxtral-tts":
        bad(f"voice '{req.voice}' is a cloning reference from '{entry['engine']}', and the "
            "public Voxtral checkpoint cannot clone (no audio-encoder weights) — use a "
            "builtin preset or a reference whose origin is voxtral-tts")
    return {"voice": entry["voice"]}  # own clip — resolve back to the preset it was rendered from


def effective_params(req: SpeechRequest) -> dict:
    """-> upstream request fields for the declared knobs."""
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {}
    for k, v in req.params.items():
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        p = PARAMS[k]
        if p["type"] == "integer":
            if isinstance(v, bool) or not isinstance(v, int):
                bad(f"params.{k} must be an integer")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            bad(f"params.{k} must be a number")
        if not p["min"] <= v <= p["max"]:
            bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = v
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    out = {}
    extra = {k: knobs[k] for k in ("temperature", "top_p", "top_k") if k in knobs}
    if extra:
        out["extra_params"] = extra
    for k in ("seed", "max_new_tokens"):
        if k in knobs:
            out[k] = knobs[k]
    return out


def upstream_speech(body: dict, timeout: float = 600):
    r = urllib.request.Request(f"{UPSTREAM}/v1/audio/speech",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(r, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise HTTPException(status_code=502, detail=f"voxtral upstream error: {detail}")


@app.get("/health")
async def health():
    # vllm's HTTP server only answers /health once the pipeline is loaded, so
    # upstream reachable == ready to synthesize.
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=3):
            pass
    except Exception:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "voxtral-tts"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": p, "name": preset_name(p), "kind": "builtin",
            "langs": LANGS, "gender": p.rsplit("_", 1)[1]} for p in PRESETS]
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
    body = {"model": MODEL, "input": req.input,
            **resolve_voice(req), **effective_params(req)}
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")
        # Native streaming: stream_format=audio makes upstream emit raw
        # 16-bit mono PCM as the codec stage produces it — passed through
        # untouched. Small reads matter: read(N) blocks until N bytes arrive,
        # so a big N would add artificial first-audio latency. 4 KB = 85 ms
        # of audio at 24 kHz.
        resp = upstream_speech(body | {"stream": True, "stream_format": "audio",
                                       "response_format": "pcm"})

        def gen() -> Generator[bytes, None, None]:
            with resp:
                while chunk := resp.read(4096):
                    yield chunk

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"x-sample-rate": str(SAMPLE_RATE)})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    with upstream_speech(body | {"response_format": "wav"}) as r:
        return Response(content=r.read(), media_type="audio/wav")


start_upstream()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
