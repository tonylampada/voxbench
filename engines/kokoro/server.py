#!/usr/bin/env python3
"""
Kokoro TTS adapter — serves the bench contract (docs/contract.md) by
translating it to the Kokoro-FastAPI OpenAI-compatible API running on
localhost:8880 in the same container. Reference implementation for v2.
"""
import json
import os
import queue
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

UPSTREAM = "http://127.0.0.1:8880"
SAMPLE_RATE = 24000
DEFAULT_LANG = "en"
DEFAULT_VOICE = {"en": "af_heart", "pt": "pf_dora"}
# Kokoro picks the G2P language from the voice name prefix:
# a/b = English (US/UK), p = Brazilian Portuguese, e = Spanish, f = French,
# h = Hindi, i = Italian, j = Japanese, z = Mandarin.
PREFIX_LANG = {"a": "en", "b": "en", "p": "pt", "e": "es", "f": "fr",
               "h": "hi", "i": "it", "j": "ja", "z": "zh"}
GENDER = {"f": "female", "m": "male"}

PARAMS = {"speed": {"type": "number", "min": 0.25, "max": 4.0, "default": 1.0,
                    "description": "playback speed multiplier (1.0 = normal)"}}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
_manifest_file = next((p for p in (Path("/app/voices/library/manifest.json"),
                                   Path("/app/voices_manifest.json")) if p.exists()), None)
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Kokoro TTS Adapter")

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


def upstream_get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"{UPSTREAM}{path}", timeout=timeout) as r:
        return json.load(r)


_builtins: Optional[list[str]] = None


def builtins() -> list[str]:
    global _builtins
    if _builtins is None:
        _builtins = sorted(upstream_get("/v1/audio/voices")["voices"])
    return _builtins


def pretty(voice: str) -> str:
    return voice.split("_", 1)[-1].replace("_", " ").title()


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> tuple[str, str]:
    """-> (upstream kokoro voice, lang), enforcing the contract's 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in PREFIX_LANG.values():
        bad(f"lang '{lang}' not supported — one of {sorted(set(PREFIX_LANG.values()))}")
    if req.voice is None:
        lang = lang or DEFAULT_LANG
        if lang not in DEFAULT_VOICE:
            bad(f"no default voice for lang '{lang}' — pick one from /v1/voices")
        return DEFAULT_VOICE[lang], lang
    voice = req.voice
    if voice in REFERENCES:
        entry = REFERENCES[voice]
        if entry["engine"] != "kokoro":
            bad(f"'{voice}' is a cloning reference rendered by {entry['engine']}; "
                f"kokoro cannot clone — use a builtin or kokoro-origin voice id")
        if lang is not None and lang != entry["lang"]:
            bad(f"voice '{voice}' speaks '{entry['lang']}', not '{lang}'")
        voice, lang = entry["voice"], entry["lang"]
    if voice not in builtins():
        bad(f"unknown voice '{req.voice}' — see /v1/voices")
    voice_lang = PREFIX_LANG.get(voice[0])
    if lang is not None and lang != voice_lang:
        bad(f"voice '{voice}' speaks '{voice_lang}', not '{lang}': Kokoro derives "
            f"G2P language from the voice prefix. Refusing the wrong phonemizer.")
    return voice, voice_lang


def effective_speed(req: SpeechRequest) -> float:
    knobs = req.params if isinstance(req.params, dict) else bad("params must be an object")
    for k in knobs:
        if k not in PARAMS:
            bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
    speed = knobs.get("speed")
    if req.speed is not None and speed is not None:
        bad("speed given both top-level and in params — pick one place")
    if speed is None:
        speed = req.speed
    if speed is None:
        return 1.0
    p = PARAMS["speed"]
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        bad("params.speed must be a number")
    if not p["min"] <= speed <= p["max"]:
        bad(f"speed {speed} out of range [{p['min']}, {p['max']}]")
    return float(speed)


def upstream_speech(body: dict, timeout: float = 300):
    req = urllib.request.Request(f"{UPSTREAM}/v1/audio/speech",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502,
                            detail=f"kokoro upstream error: {e.read().decode(errors='replace')[:500]}")


@app.get("/health")
async def health():
    # 200 only once the upstream kokoro server answers — it loads the model
    # during startup, so reachable == ready to synthesize.
    try:
        builtins()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "kokoro"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": v, "name": pretty(v), "kind": "builtin",
            "langs": [PREFIX_LANG.get(v[0], "en")],
            "gender": GENDER.get(v[1:2], "unknown")} for v in builtins()]
    out += [{"id": vid, "name": pretty(e["voice"]) if e["engine"] == "kokoro" else e["voice"],
             "kind": "reference", "origin": e["engine"], "langs": [e["lang"]],
             "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
            for vid, e in sorted(REFERENCES.items())]
    return {"voices": out}


@app.get("/v1/params")
async def params():
    return {"params": PARAMS}


# One synthesis at a time (docs/contract.md, "Concurrency"). The generation
# happens in the upstream process, which could take both at once — this engine
# serializes anyway so every adapter answers the console the same way.
gen_lock = threading.Lock()


def locked_stream(open_upstream, read_size: int):
    """Take gen_lock, open the upstream response, and relay it -> (resp, body).

    The lock is held from before upstream starts generating until its last byte,
    and it is released by the relay thread, which always reaches its finally.
    That thread is the whole point: when a client hangs up mid-stream Starlette
    abandons the response generator where it stands without finalizing it, so a
    `with gen_lock` inside that generator would never be exited and the engine
    would wedge for good. Draining upstream also leaves it clean for whoever is
    next in line.
    """
    gen_lock.acquire()
    try:
        resp = open_upstream()
    except BaseException:  # nothing to relay, so nothing would ever release it
        gen_lock.release()
        raise
    # No maxsize, on purpose: a bounded put() would park the relay thread while
    # it still holds gen_lock as soon as an abandoned client stopped draining —
    # the exact deadlock this design avoids. The price is one utterance in memory.
    chunks = queue.Queue()

    def relay():
        try:
            with resp:
                while chunk := resp.read(read_size):
                    chunks.put(chunk)
            chunks.put(None)
        except BaseException as exc:
            chunks.put(exc)
        finally:
            gen_lock.release()

    threading.Thread(target=relay, daemon=True).start()

    def body() -> Generator[bytes, None, None]:
        while (item := chunks.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item

    return resp, body()


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice, _ = resolve(req)
    speed = effective_speed(req)
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")
        _, body = locked_stream(
            lambda: upstream_speech({"model": "kokoro", "input": req.input, "voice": voice,
                                     "response_format": "pcm", "speed": speed, "stream": True}),
            8192)
        return StreamingResponse(body, media_type="application/octet-stream",
                                 headers={"x-sample-rate": str(SAMPLE_RATE)})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    with gen_lock, upstream_speech({"model": "kokoro", "input": req.input, "voice": voice,
                                    "response_format": "wav", "speed": speed,
                                    "stream": False}) as r:
        return Response(content=r.read(), media_type="audio/wav")
