#!/usr/bin/env python3
"""
VoxCPM2 TTS engine adapter — serves the bench contract (docs/contract.md).
Loads openbmb/VoxCPM2 (48 kHz output) and clones voices from library reference
clips via the model's basic cloning mode (reference_wav_path).
"""
import asyncio
import io
import json
import os
import queue
import re
import threading
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
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
model = None
SAMPLE_RATE = 48000  # overwritten from the loaded model at startup
# One generation at a time. The endpoint is sync, so FastAPI runs each request in
# its own threadpool thread, all on the same `model` — two concurrent decodes
# share its kv_cache, corrupt each other, and at worst trip a device-side assert
# that poisons the CUDA context for the life of the process (see latch()).
# Latecomers wait; the contract has no "busy" answer.
gen_lock = threading.Lock()
FATAL = None  # set once the CUDA context is gone; see latch() — only a restart clears it


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


# How much text goes into ONE decode. Measured on this model: a decode stays
# well-behaved to ~800 characters and degenerates past it — 995 characters of
# run-on text produced 113 s of dragging audio (8.8 chars/s against a healthy
# 17), and a long enough decode walks off the model's 8192-entry audio codebook
# and takes the CUDA context with it. 400 keeps a 2x margin.
MAX_CHUNK_CHARS = 400


def split_into_sentences(text: str) -> list[str]:
    text = text.strip()
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()] if text else []


def cut_to_size(piece: str, limit: int) -> list[str]:
    """Cut one over-long sentence on the best seam under the limit: a comma-ish
    pause if there is one, else a space. Punctuation stays with the head, so a
    chunk still ends on the pause the writer wrote."""
    out = []
    while len(piece) > limit:
        head = piece[:limit]
        cut = max(head.rfind(", "), head.rfind("; "), head.rfind(": "))
        if cut < limit // 2:  # too early to be a fair seam — fall back to a word break
            cut = head.rfind(" ")
        if cut <= 0:  # a single word longer than the limit
            cut = limit - 1
        out.append(piece[:cut + 1].strip())
        piece = piece[cut + 1:].strip()
    return out + ([piece] if piece else [])


def chunk_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Text -> pieces no longer than `limit`, each one decode's worth. Sentences
    are the seam when they exist; a sentence longer than the limit is cut anyway,
    because 'no full stop in it' is not a reason to hand the model a runaway."""
    chunks, cur = [], ""
    for sentence in split_into_sentences(text):
        for piece in cut_to_size(sentence, limit):
            if cur and len(cur) + 1 + len(piece) > limit:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


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


@app.get("/health")
async def health():
    if FATAL:
        return JSONResponse(status_code=503,
                            content={"status": "unusable", "engine": "voxcpm2", "detail": FATAL})
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


def latch(exc: BaseException) -> None:
    """A device-side assert poisons the CUDA context for the life of the process:
    every later call fails, forever. There is no in-process recovery, so record it
    and let /health say so. An OOM is not that — the context survives one."""
    global FATAL
    if "CUDA" in str(exc) and "out of memory" not in str(exc).lower():
        FATAL = (f"CUDA context is dead ({type(exc).__name__}: {str(exc).splitlines()[0][:160]}). "
                 "Every synthesis from here on fails; restart the container "
                 "(engines/tts down voxcpm2 && engines/tts up voxcpm2).")
        print(f"FATAL: {FATAL}", flush=True)


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if FATAL:
        raise HTTPException(status_code=503, detail=FATAL)
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice_path = resolve(req)
    knobs = effective_params(req)
    # ONE decode gets at most MAX_CHUNK_CHARS characters, on both paths. Streaming
    # bounds how the audio comes out; this bounds how much text goes in, and only
    # the second one keeps the model from running away.
    pieces = chunk_text(req.input) or [req.input]
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")
        # Each piece runs through generate_streaming(), and every chunk the model
        # natively yields goes out as raw int16 PCM.
        #
        # Decoding runs on a thread WE own, not on the response generator, for
        # two reasons the lock cares about. It happens while the body is being
        # consumed, so holding the lock only inside speech() would serialize
        # nothing; and when a client hangs up mid-stream (the board aborts its
        # fetch on every new message) Starlette abandons the response generator
        # where it stands, without finalizing it — a `with gen_lock` in there is
        # simply never exited and the engine wedges for good, which is a worse
        # bug than the race. A thread always reaches its finally.
        #
        # The generator is async so the hangup is *heard*: an abandoned async
        # generator has its task cancelled, so its finally DOES run (that is
        # what the sync one, driven from a threadpool, never gets). It sets
        # `gone`, produce() sees it between chunks and stops generating, and
        # gen_lock is free within a chunk instead of a whole utterance.
        async def gen() -> AsyncGenerator[bytes, None]:
            # `None` ends it, an exception re-raises here. NO maxsize, on
            # purpose: a bounded queue would park the producer inside put()
            # once an abandoned client stopped draining it — still holding
            # gen_lock — which is the exact deadlock this thread exists to
            # avoid. The price is one utterance buffered in memory, which the
            # non-streaming path pays anyway.
            chunks = queue.Queue()
            gone = threading.Event()  # the client hung up; stop decoding for nobody

            def produce():
                try:
                    with gen_lock:
                        for piece in pieces:
                            if gone.is_set():
                                break
                            for audio in model.generate_streaming(
                                    text=piece, reference_wav_path=voice_path, **knobs):
                                chunks.put(to_int16_bytes(audio))
                                if gone.is_set():
                                    break
                    chunks.put(None)
                except BaseException as exc:  # the response already started — latch, then die
                    latch(exc)
                    chunks.put(exc)

            threading.Thread(target=produce, daemon=True).start()
            try:
                # to_thread, not chunks.get(): a blocking get would stall the
                # whole event loop, disconnects included.
                while (item := await asyncio.to_thread(chunks.get)) is not None:
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                gone.set()  # every exit — done, error, or hangup

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"x-sample-rate": str(SAMPLE_RATE)})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    try:
        with gen_lock:
            audio = np.concatenate([model.generate(text=p, reference_wav_path=voice_path, **knobs)
                                    for p in pieces])
    except Exception as exc:
        latch(exc)
        raise
    return Response(content=numpy_to_wav_bytes(audio, SAMPLE_RATE), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)  # bound to localhost via docker -p 127.0.0.1:...
