#!/usr/bin/env python3
"""Piper TTS adapter — CPU-only ONNX engine, built-in voices, no cloning.
Serves the bench contract (docs/contract.md)."""
import io
import json
import os
import queue
import re
import threading
import wave
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from piper import PiperVoice, SynthesisConfig
from pydantic import BaseModel

VOICES_DIR = "/voices"
# name -> (lang, gender); gender per the rhasspy/piper-voices dataset cards
BUILTIN = {
    "pt_BR-faber-medium": ("pt", "male"),
    "pt_BR-cadu-medium": ("pt", "male"),
    "pt_BR-edresson-low": ("pt", "male"),
    "pt_BR-jeff-medium": ("pt", "male"),
    "en_US-lessac-medium": ("en", "female"),
}
DEFAULT_VOICE = {"pt": "pt_BR-faber-medium", "en": "en_US-lessac-medium"}
DEFAULT_LANG = "en"
LANGS = sorted(set(l for l, _ in BUILTIN.values()))

PARAMS = {
    "speed": {"type": "number", "min": 0.25, "max": 4.0, "default": 1.0,
              "description": "playback speed multiplier (1.0 = normal; maps to Piper's inverse length_scale)"},
    "noise_scale": {"type": "number", "min": 0.0, "max": 2.0, "default": 0.667,
                    "description": "phoneme noise: higher = more expressive variation, lower = flatter delivery"},
    "noise_w_scale": {"type": "number", "min": 0.0, "max": 2.0, "default": 0.8,
                      "description": "phoneme duration noise: higher = more varied speech rhythm"},
}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
_manifest_file = next((p for p in (Path("/app/voices/library/manifest.json"),
                                   Path("/app/voices_manifest.json")) if p.exists()), None)
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

voices = {}
# One synthesis at a time: the endpoint is sync, so FastAPI runs each request in
# its own threadpool thread, all sharing one PiperVoice per id. Latecomers wait.
gen_lock = threading.Lock()

app = FastAPI(title="Piper TTS Adapter")

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
    speed: Optional[float] = None  # OpenAI-compat alias for params.speed
    stream: bool = False
    params: dict = {}


@app.on_event("startup")
def load_voices():
    for name in BUILTIN:
        print(f"Loading {name}...")
        voices[name] = PiperVoice.load(f"{VOICES_DIR}/{name}.onnx")
    print("All voices loaded.")


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def pretty(name: str) -> str:
    return name.split("-")[1].title()  # pt_BR-faber-medium -> Faber


def resolve(req: SpeechRequest) -> str:
    """-> builtin voice name, enforcing the contract's 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {LANGS}")
    if req.voice is None:
        return DEFAULT_VOICE[lang or DEFAULT_LANG]
    voice = req.voice
    if voice in REFERENCES:
        entry = REFERENCES[voice]
        if entry["engine"] != "piper":
            bad(f"'{voice}' is a cloning reference rendered by {entry['engine']}; "
                f"piper cannot clone — use a builtin or piper-origin voice id")
        if lang is not None and lang != entry["lang"]:
            bad(f"voice '{voice}' speaks '{entry['lang']}', not '{lang}'")
        voice = entry["voice"]
    if voice not in BUILTIN:
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    if lang is not None and lang != BUILTIN[voice][0]:
        bad(f"voice '{voice}' speaks '{BUILTIN[voice][0]}', not '{lang}'")
    return voice


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


def syn_config(knobs: dict) -> Optional[SynthesisConfig]:
    if not knobs:
        return None
    return SynthesisConfig(
        length_scale=1.0 / knobs["speed"] if "speed" in knobs else None,
        noise_scale=knobs.get("noise_scale"),
        noise_w_scale=knobs.get("noise_w_scale"),
    )


@app.get("/health")
def health():
    if len(voices) < len(BUILTIN):
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "piper"}


@app.get("/v1/voices")
def list_voices():
    out = [{"id": v, "name": pretty(v), "kind": "builtin", "langs": [lang], "gender": gender}
           for v, (lang, gender) in sorted(BUILTIN.items())]
    out += [{"id": vid, "name": pretty(e["voice"]) if e["engine"] == "piper" else e["voice"],
             "kind": "reference", "origin": e["engine"], "langs": [e["lang"]],
             "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
            for vid, e in sorted(REFERENCES.items())]
    return {"voices": out}


@app.get("/v1/params")
def params():
    return {"params": PARAMS}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice = voices[resolve(req)]
    cfg = syn_config(effective_params(req))
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")

        # Native incremental synthesis: PiperVoice.synthesize yields audio
        # chunks as the ONNX model produces them, one per sentence.
        #
        # Synthesis runs on a thread WE own, not on the response generator, for
        # two reasons the lock cares about. It happens while the body is being
        # consumed, so holding the lock only inside speech() would serialize
        # nothing; and when a client hangs up mid-stream (the board aborts its
        # fetch on every new message) Starlette abandons the response generator
        # where it stands, without finalizing it — a `with gen_lock` in there is
        # simply never exited and the engine wedges for good, which is a worse
        # bug than the race. A thread always reaches its finally.
        def gen() -> Generator[bytes, None, None]:
            # `None` ends it, an exception re-raises here. NO maxsize, on
            # purpose: a bounded queue would park the producer inside put()
            # once an abandoned client stopped draining it — still holding
            # gen_lock — which is the exact deadlock this thread exists to
            # avoid. The price is one utterance buffered in memory, which the
            # non-streaming path pays anyway.
            chunks = queue.Queue()

            def produce():
                try:
                    with gen_lock:
                        for chunk in voice.synthesize(req.input, syn_config=cfg):
                            chunks.put(chunk.audio_int16_bytes)
                    chunks.put(None)
                except BaseException as exc:
                    chunks.put(exc)

            threading.Thread(target=produce, daemon=True).start()
            while (item := chunks.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"x-sample-rate": str(voice.config.sample_rate)})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    buf = io.BytesIO()
    with gen_lock, wave.open(buf, "wb") as wf:
        voice.synthesize_wav(req.input, wf, syn_config=cfg)
    return Response(content=buf.getvalue(), media_type="audio/wav")
