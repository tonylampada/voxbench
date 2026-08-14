#!/usr/bin/env python3
"""
VoxCPM2 TTS engine adapter — serves the bench contract (docs/contract.md).
Loads openbmb/VoxCPM2 (48 kHz output) and gets its voice one of two ways: cloned
from a library reference clip via the model's basic cloning mode
(reference_wav_path), or invented from a written description via its voice-design
mode (params.voice_prompt).
"""
import asyncio
import io
import json
import os
import queue
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import torch
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
    "seed": {"type": "integer", "min": 0, "max": 2147483647, "default": 42,
             "description": "random seed; same text + same knobs + same seed reproduces the same take"},
    "voice_prompt": {"type": "string", "default": "",
                     "description": "describe a voice in words and the model invents one to match ('a tired old man, speaking slowly'); empty means clone the library voice instead, and it cannot be combined with one"},
    "continuity": {"type": "boolean", "default": True,
                   "description": "carry the voice across the pieces of one message instead of restarting it at every seam; off reproduces the old take, which drifts into a different person over a long paragraph"},
}

# The model's voice-design mode is a text convention, not an argument: voxcpm's
# own CLI (`voxcpm design --control "..."`) just writes the description into a
# leading parenthesis of the text it decodes, and the model reads it as an
# instruction instead of speaking it. See build_final_text() in voxcpm/cli.py.
MAX_VOICE_PROMPT = 300

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


def resolve(req: SpeechRequest) -> tuple[Optional[str], Optional[str]]:
    """-> (reference clip path, what that clip says), or (None, None) for the
    model's own (arbitrary) voice. The transcript comes from the manifest
    (bench/transcribelib.py put it there) and is what makes the model's
    ref-audio-plus-transcript mode reachable — a clip alone says who is
    speaking, a clip with its words also says how they speak them."""
    if req.lang is not None and req.lang not in LANGS:
        bad(f"lang '{req.lang}' not supported — this engine speaks {sorted(LANGS)}")
    if req.voice is None:
        return None, None
    v = req.voice
    if v in REFERENCES:  # cloning engine: any library clip works
        e = REFERENCES[v]
        if req.lang is not None and req.lang != e["lang"]:
            bad(f"voice '{v}' speaks '{e['lang']}', not '{req.lang}'")
        return str(LIBRARY_DIR / e["file"]), (e.get("text") or "").strip() or None
    bad(f"unknown voice '{v}' — see /v1/voices")


def effective_params(req: SpeechRequest) -> tuple[dict, int, str, bool]:
    """-> (kwargs for the decoder, seed, voice description, continuity). Only the
    first is a generate kwarg: voxcpm has no seed argument, so the seed is spent
    on torch's global RNG instead (see seed_rng), which is what its samplers draw
    from; the description is prefixed onto the text (see designed); and
    continuity is about how the pieces are conditioned, not about one decode."""
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
            knobs[k] = v
            continue
        if p["type"] == "boolean":
            if not isinstance(v, bool):
                bad(f"params.{k} must be a boolean")
            knobs[k] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or \
                (p["type"] == "integer" and not isinstance(v, int)):
            bad(f"params.{k} must be a {p['type']}")
        if not p["min"] <= v <= p["max"]:
            bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
        knobs[k] = v
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    # The other two knobs let voxcpm apply its own defaults; the seed cannot,
    # because "no seed" means fresh randomness and that is the bug. Defaulted
    # here, every request draws the same sampling noise, so the chunks of one
    # message stop being independent takes.
    seed = knobs.pop("seed", PARAMS["seed"]["default"])
    continuity = knobs.pop("continuity", PARAMS["continuity"]["default"])
    voice_prompt = knobs.pop("voice_prompt", "").strip()
    if voice_prompt:
        if req.voice is not None:
            bad(f"voice '{req.voice}' and params.voice_prompt are two voice sources for one "
                "request — a library clip clones a voice, a description invents one; pass one")
        # The description rides in a parenthesis of the text, so a parenthesis
        # of its own would end it early and the rest would be spoken aloud.
        if "(" in voice_prompt or ")" in voice_prompt:
            bad("params.voice_prompt must not contain parentheses")
        if len(voice_prompt) > MAX_VOICE_PROMPT:
            bad(f"params.voice_prompt is {len(voice_prompt)} characters — "
                f"at most {MAX_VOICE_PROMPT}; it describes a voice, it is not the text")
    return knobs, seed, voice_prompt, continuity


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


def designed(text: str, voice_prompt: str) -> str:
    """One piece of text, with the voice description the model reads as an
    instruction instead of speaking. Unchanged when there is no description."""
    return f"({voice_prompt}){text}" if voice_prompt else text


# ── keeping one message one person ───────────────────────────────────────────
# A piece is its own decode with no memory of the last one, so what each piece
# is conditioned on IS the voice. Measured over a seven-piece paragraph
# (engines/voxcpm2/continuity.py, the artifact on MNC-102), on a scale where the
# same voice twice scores 0.977 and two different library voices score 0.492:
#
#                                        worst   vs its own   speaking-rate
#                                         seam      opening         spread
#   reference clip alone, every piece     0.948        0.946            15%
#   clip + its transcript, every piece    0.982        0.985            11%
#   each piece continuing the last one    0.970        0.916            28%
#   both                                  0.962        0.818            16%
#
# The transcript is the fix and continuation is a trap. Feeding a piece the
# previous piece's audio does smooth the seam — and the take accelerates from
# 14.8 to 20.0 characters a second while it does, because each piece inherits
# the last one's drift and adds its own. Without the clip to hold it, 26.1.
# An anchor works precisely because it is the SAME anchor every time.


def clean(text: str) -> str:
    """What VoxCPM.generate() does to text before decoding it. We call the model
    one level down from generate() — it cannot be told to reuse a prompt cache —
    so its text handling comes with us."""
    return re.sub(r"\s+", " ", text.replace("\n", " "))


def anchor(voice_path: str, voice_text: Optional[str]):
    """The conditioning every piece of this request gets: the reference clip,
    and its own words when the manifest knows them (bench/transcribelib.py put
    them there). A clip alone says who is speaking; a clip with its transcript
    also says how that person speaks — which is the half that was drifting."""
    return model.tts_model.build_prompt_cache(
        reference_wav_path=voice_path,
        prompt_wav_path=voice_path if voice_text else None,
        prompt_text=voice_text)


def anchor_from(audio: np.ndarray):
    """A described voice has no clip to be anchored to, which is why it drifted
    furthest of all — 0.651 at its worst seam, most of the way to being somebody
    else. So the first piece becomes the clip: it is a real recording of the
    voice the description invented, and every piece after it clones that."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(numpy_to_wav_bytes(audio, SAMPLE_RATE))
    try:
        return model.tts_model.build_prompt_cache(reference_wav_path=f.name)
    finally:
        os.unlink(f.name)


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


def seed_rng(seed: int) -> None:
    """Point the samplers at a known RNG state. voxcpm takes no seed and no
    generator: the flow-matching head draws its initial noise with a bare
    torch.randn (modules/locdit/unified_cfm.py) and the vocoder's noise blocks
    do the same, so the global generator IS the sampler's RNG. Safe because
    gen_lock already serializes generation — nobody else is drawing.

    Called once per PIECE, not once per request, so a paragraph the engine
    split itself sounds exactly like the same paragraph the client split and
    sent chunk by chunk. Either way, every chunk starts from the same noise."""
    torch.manual_seed(seed)


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
    voice_path, voice_text = resolve(req)
    knobs, seed, voice_prompt, continuity = effective_params(req)
    # ONE decode gets at most MAX_CHUNK_CHARS characters, on both paths. Streaming
    # bounds how the audio comes out; this bounds how much text goes in, and only
    # the second one keeps the model from running away. The description counts
    # against the same budget — the model decodes it too — and goes on EVERY
    # piece, because each piece is its own decode with no memory of the last: a
    # described voice the first chunk alone carried would be an arbitrary voice
    # from the second chunk on — unless continuity is on, and then the first
    # piece's own audio is the anchor from there, so the description has done
    # its job once and would only eat decode budget after that.
    limit = MAX_CHUNK_CHARS - len(designed("", voice_prompt))
    plain = [clean(p) for p in chunk_text(req.input, limit) or [req.input]]
    pieces = [designed(plain[0], voice_prompt)] + (
        plain[1:] if continuity else [designed(p, voice_prompt) for p in plain[1:]])
    # Encoding the clip is GPU work like any other, so it happens under
    # gen_lock, not here. The transcript half of the anchor is what continuity
    # buys a cloned voice; off, the clip goes in on its own, as it used to.
    def build_base():
        return None if voice_path is None else anchor(
            voice_path, voice_text if continuity else None)
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")
        # Each piece runs through the model's streaming decode, and every chunk it
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
                        cache = build_base()
                        for piece in pieces:
                            if gone.is_set():
                                break
                            seed_rng(seed)
                            # A described voice keeps its first piece, to become
                            # the clip the rest of the message clones.
                            keep = [] if cache is None and continuity else None
                            for audio, _, _ in model.tts_model.generate_with_prompt_cache_streaming(
                                    target_text=piece, prompt_cache=cache, **knobs):
                                a = audio.squeeze(0).cpu().numpy()
                                if keep is not None:
                                    keep.append(a)
                                chunks.put(to_int16_bytes(a))
                                if gone.is_set():
                                    break
                            if keep:
                                cache = anchor_from(np.concatenate(keep))
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
            cache, parts = build_base(), []
            for piece in pieces:
                seed_rng(seed)
                wav, _, _ = model.tts_model.generate_with_prompt_cache(
                    target_text=piece, prompt_cache=cache,
                    retry_badcase=True,  # what VoxCPM.generate() does; streaming cannot
                    **knobs)
                parts.append(wav.squeeze(0).cpu().numpy())
                if cache is None and continuity:  # described: the first take is the clip now
                    cache = anchor_from(parts[0])
            audio = np.concatenate(parts)
    except Exception as exc:
        latch(exc)
        raise
    return Response(content=numpy_to_wav_bytes(audio, SAMPLE_RATE), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)  # bound to localhost via docker -p 127.0.0.1:...
