#!/usr/bin/env python3
"""
Higgs TTS 3 adapter — serves the bench contract (docs/contract.md) by
translating it to the SGLang-Omni OpenAI-compatible /v1/audio/speech API
running on localhost:8000 in the same container. Upstream already speaks the
contract's shape, so this adapter mostly validates and proxies — including
native PCM streaming, passed through untouched.

The upstream server (sgl-omni serve) is launched as a subprocess at
startup; /health answers 200 only once it accepts requests, which
happens only after the 4B model is loaded.
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
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

UPSTREAM = "http://127.0.0.1:8000"
MODEL = os.environ.get("HIGGS_MODEL", "bosonai/higgs-tts-3-4b")
VOICES_DIR = Path("/app/voices")
# Higgs is a single multilingual model: the language comes from the text
# itself, there is no lang parameter upstream. Both en and pt sit in its
# sub-5% WER tier, so both are accepted; anything else is refused loudly.
LANGS = ["en", "pt"]

PARAMS = {
    "temperature": {"type": "number", "min": 0.0, "max": 2.0, "default": 1.0,
                    "description": "sampling temperature; higher = more varied takes, lower = more stable"},
    "top_k": {"type": "integer", "min": 0, "max": 200, "default": 50,
              "description": "top-k sampling cutoff (0 = disabled)"},
}

# Library manifest + clips, mounted live at /app/voices.
LIBRARY_DIR = VOICES_DIR / "library"
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Higgs TTS 3 Adapter")

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
    args = [
        "/opt/omni/bin/sgl-omni", "serve",
        "--model-path", MODEL,
        "--port", "8000",
        "--host", "127.0.0.1",
    ]
    # 24 GB card, 9.3 GB weights: sgl-omni serve has NO --context-length
    # flag; the OOM headroom knob it does expose is --mem-fraction-static.
    # Unset → SGLang picks automatically.
    mem_frac = os.environ.get("HIGGS_MEM_FRACTION")
    if mem_frac:
        args += ["--mem-fraction-static", mem_frac]
    # Reference clips are mounted at /app/voices; upstream refuses to read
    # local files outside --allowed-local-media-path.
    if VOICES_DIR.is_dir():
        args += ["--allowed-local-media-path", str(VOICES_DIR)]
    subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve_references(req: SpeechRequest) -> list:
    """-> upstream `references` (empty = the builtin voice), enforcing the 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {LANGS}")
    if req.voice is None or req.voice == "default":
        return []
    if req.voice not in REFERENCES:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    entry = REFERENCES[req.voice]
    if lang is not None and lang != entry["lang"]:
        bad(f"voice '{req.voice}' speaks '{entry['lang']}', not '{lang}'")
    if entry["lang"] not in LANGS:
        bad(f"voice '{req.voice}' speaks '{entry['lang']}', which this engine cannot — see /v1/voices")
    clip = LIBRARY_DIR / entry["file"]
    ref = {"audio_path": str(clip)}
    txt = clip.with_suffix(".txt")  # sibling transcript improves cloning when present
    if txt.is_file():
        ref["text"] = txt.read_text().strip()
    return [ref]


def effective_params(req: SpeechRequest) -> dict:
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
    return knobs


def upstream_speech(body: dict, timeout: float = 600):
    r = urllib.request.Request(f"{UPSTREAM}/v1/audio/speech",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(r, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"higgs upstream error: {e.read().decode(errors='replace')[:500]}")


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
async def health():
    # sgl-omni's HTTP server only answers once the model is loaded, so
    # upstream reachable == ready to synthesize.
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=3):
            pass
    except Exception:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "higgs-tts-3"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": "default", "name": "Default", "kind": "builtin",
            "langs": LANGS, "gender": "unknown"}]
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
    refs = resolve_references(req)
    knobs = effective_params(req)
    body = {"model": MODEL, "voice": "default", "input": req.input, **knobs}
    if refs:
        body["references"] = refs
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")
        # Native streaming: upstream emits raw 16-bit mono PCM as the vocoder
        # produces it — passed through untouched. Small reads matter: read(N)
        # blocks until N bytes arrive, so a big N would add artificial
        # first-audio latency. 4 KB = 85 ms of audio at 24 kHz.
        resp = upstream_speech(body | {"stream": True, "response_format": "pcm"})
        sample_rate = resp.headers.get("x-sample-rate", "24000")

        def gen() -> Generator[bytes, None, None]:
            with resp:
                while chunk := resp.read(4096):
                    yield chunk

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"x-sample-rate": sample_rate})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    with upstream_speech(body) as r:
        return Response(content=r.read(), media_type="audio/wav")


start_upstream()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
