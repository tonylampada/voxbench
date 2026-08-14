#!/usr/bin/env python3
"""
VibeVoice-1.5B TTS engine adapter — serves the bench contract (docs/contract.md).
Loads microsoft/VibeVoice-1.5B (24 kHz) and CLONES: every voice, preset or
library reference alike, is a reference wav the model conditions on.

That is the whole difference from the sibling engine on 8895. The realtime 0.5B
has the acoustic ENCODER zeroed out, so it can only replay prefilled caches
Microsoft shipped; this checkpoint carries the encoder (552 encoder tensors in
its safetensors index), so any clip in voices/library/ is a usable voice.
"""
import asyncio
import io
import json
import os
import re
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

from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL_ID = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-1.5B")
PRESET_DIR = Path("/app/presets")
SAMPLE_RATE = 24000  # the acoustic tokenizer's rate; fixed, not read from config

# Preset filenames carry the vendor's own language tag; "in" is the
# Indian-accented ENGLISH voice, not a language.
PREFIX_LANG = {"en": "en", "in": "en", "zh": "zh"}

PARAMS = {
    "cfg_scale": {"type": "number", "min": 1.0, "max": 4.0, "default": 1.3,
                  "description": "classifier-free guidance on the diffusion head; 1.3 is the vendor default, above ~2 the voice gets tense"},
    "inference_steps": {"type": "integer", "min": 1, "max": 50, "default": 10,
                        "description": "DDPM sampling steps per acoustic latent; more = cleaner audio, fewer = faster"},
    "seed": {"type": "integer", "min": 0, "max": 2147483647, "default": 42,
             "description": "random seed; same text + same knobs + same seed reproduces the same take"},
}

LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}


def discover_presets() -> dict:
    """-> {voice_id: {"file": Path, "lang": .., "gender": .., "name": ..}}.
    Ids are the vendor's filenames ("en-Alice_woman"), kept verbatim so a voice
    can be matched back to the upstream clip."""
    out = {}
    for f in sorted(PRESET_DIR.glob("*.wav")):
        prefix, _, rest = f.stem.partition("-")
        lang = PREFIX_LANG.get(prefix)
        if lang is None:
            print(f"skipping preset {f.name}: unknown language prefix '{prefix}'", flush=True)
            continue
        parts = rest.split("_")
        gender = {"man": "male", "woman": "female"}.get(parts[1] if len(parts) > 1 else "", "unknown")
        # "_bgm" in the filename means the reference clip has music under the
        # speech — and the clone reproduces the music. Say so in the name; the
        # console shows nothing else.
        name = parts[0] + (" + music bed" if rest.endswith("_bgm") else "")
        out[f.stem] = {"file": f, "lang": lang, "gender": gender,
                       "name": f"{name} ({lang})"}
    return out


PRESETS = discover_presets()
# Every language the catalogue can speak. The model itself takes no language
# argument: it reads whatever the text says in whatever the reference sounds
# like. Microsoft trained and tested en + zh only — the rest of the library is
# the model trying, and the README has the measured Portuguese.
LANGS = {e["lang"] for e in PRESETS.values()} | {e["lang"] for e in REFERENCES.values()}
DEFAULT_LANG = "en"
# One default voice per language, for a request that names a lang and no voice.
# Microsoft ships presets for two languages only; for the rest the default is
# the first LIBRARY clip of that language by id — arbitrary but stable, and a
# cloning engine that lists 200-odd references has no business refusing to pick
# one. The console and the bench both drive it that way.
DEFAULT_VOICE = {}
for _vid, _e in sorted(PRESETS.items()):
    DEFAULT_VOICE.setdefault(_e["lang"], _vid)
if "en-Carter_man" in PRESETS:  # the vendor demo's own default
    DEFAULT_VOICE["en"] = "en-Carter_man"
for _vid, _e in sorted(REFERENCES.items()):
    DEFAULT_VOICE.setdefault(_e["lang"], _vid)

app = FastAPI(title="VibeVoice-1.5B TTS Server")

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
processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# One generation at a time (docs/contract.md, "Concurrency"). Two decodes on one
# model share its kv_cache and corrupt each other; latecomers wait.
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


def resolve(req: SpeechRequest) -> str:
    """-> the reference clip to condition on, enforcing the contract's 400
    catalogue. Presets and library clips take the same path — both are wavs."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not in this engine's catalogue — see /v1/voices for what it can speak")
    v = req.voice
    if v is None:
        lang = lang or DEFAULT_LANG
        if lang not in DEFAULT_VOICE:
            bad(f"no default voice for lang '{lang}' — name a voice from /v1/voices")
        v = DEFAULT_VOICE[lang]
    if v in PRESETS:
        if lang is not None and lang != PRESETS[v]["lang"]:
            bad(f"voice '{v}' speaks '{PRESETS[v]['lang']}', not '{lang}'")
        return str(PRESETS[v]["file"])
    if v in REFERENCES:  # cloning engine: any library clip works, whoever rendered it
        e = REFERENCES[v]
        if lang is not None and lang != e["lang"]:
            bad(f"voice '{v}' speaks '{e['lang']}', not '{lang}'")
        path = LIBRARY_DIR / e["file"]
        if not path.exists():
            bad(f"voice '{v}' is in the manifest but its clip is missing from the mounted library")
        return str(path)
    bad(f"unknown voice '{v}' — see /v1/voices")


def effective_params(req: SpeechRequest) -> dict:
    if not isinstance(req.params, dict):
        bad("params must be an object")
    knobs = {k: p["default"] for k, p in PARAMS.items()}
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


def to_int16_bytes(audio: np.ndarray) -> bytes:
    peak = np.abs(audio).max() if audio.size else 0.0
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


def to_script(text: str) -> str:
    """The processor parses a podcast script, one 'Speaker N: ...' line at a
    time, and DROPS every line that does not match that shape (raising if none
    do). So a paragraph with newlines in it must arrive as one line, or the
    engine would silently speak only its first sentence."""
    # The vendor's demos normalize curly punctuation before tokenizing; the text
    # tokenizer has no entry for them and they come out as pauses otherwise.
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return "Speaker 1: " + " ".join(text.split())


def prepare(text: str, voice_path: str):
    """-> model inputs on device, with the reference clip encoded as the voice."""
    inputs = processor(text=[to_script(text)], voice_samples=[[voice_path]],
                       padding=True, return_tensors="pt", return_attention_mask=True)
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def latch(exc: BaseException) -> None:
    """A device-side assert poisons the CUDA context for the life of the process:
    every later call fails, forever. Record it and let /health say so. An OOM is
    not that — the context survives one."""
    global FATAL
    if "CUDA" in str(exc) and "out of memory" not in str(exc).lower():
        FATAL = (f"CUDA context is dead ({type(exc).__name__}: {str(exc).splitlines()[0][:160]}). "
                 "Every synthesis from here on fails; restart the container "
                 "(engines/tts down vibevoice-15b && engines/tts up vibevoice-15b).")
        print(f"FATAL: {FATAL}", flush=True)


@app.on_event("startup")
async def startup():
    global model, processor
    if not PRESETS:
        raise RuntimeError(f"no voice presets found under {PRESET_DIR} — the image is broken")
    print(f"Loading {MODEL_ID} on {DEVICE}...", flush=True)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_ID)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        # flash_attention_2 is what the vendor tested, but it is a compiled wheel
        # we would have to build; sdpa is their own documented fallback.
        attn_implementation="sdpa")
    model.eval()
    print(f"VibeVoice-1.5B ready: {len(PRESETS)} presets, "
          f"{len(REFERENCES)} clonable references, langs {sorted(LANGS)}", flush=True)


@app.get("/health")
async def health():
    if FATAL:
        return JSONResponse(status_code=503,
                            content={"status": "unusable", "engine": "vibevoice-15b", "detail": FATAL})
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "vibevoice-15b"}


@app.get("/v1/voices")
async def voices():
    out = [{"id": vid, "name": e["name"], "kind": "builtin",
            "langs": [e["lang"]], "gender": e["gender"]}
           for vid, e in sorted(PRESETS.items())]
    out += [{"id": vid, "name": e["voice"], "kind": "reference", "origin": e["engine"],
             "langs": [e["lang"]], "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
            for vid, e in sorted(REFERENCES.items())]
    return {"voices": out}


@app.get("/v1/params")
async def params():
    return {"params": PARAMS}


def synthesize(voice_path: str, text: str, knobs: dict, streamer=None, stop_check=None):
    """One decode, caller holding gen_lock. -> the generate() output."""
    inputs = prepare(text, voice_path)
    # VibeVoice takes no seed and no usable generator: the diffusion head draws
    # its initial noise with a bare torch.randn (modeling_vibevoice_inference.py
    # :703) and dpm_solver's randn_tensor gets generator=None, so the global
    # generator IS the sampler's RNG. The vendor's gradio demo builds a
    # torch.Generator and passes it to generate(), where it lands in **kwargs
    # and is never read — seeding it that way does nothing. Safe here because
    # gen_lock already serializes generation: nobody else is drawing.
    torch.manual_seed(knobs["seed"])
    model.set_ddpm_inference_steps(num_steps=knobs["inference_steps"])
    return model.generate(
        **inputs, max_new_tokens=None, cfg_scale=knobs["cfg_scale"],
        tokenizer=processor.tokenizer, generation_config={"do_sample": False},
        verbose=False, show_progress_bar=False,
        audio_streamer=streamer, stop_check_fn=stop_check,
        is_prefill=True)  # prefill = condition on the reference clip, i.e. clone


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if FATAL:
        raise HTTPException(status_code=503, detail=FATAL)
    if not req.input.strip():
        bad("input is required and must be non-empty")
    voice_path = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")

        # Decoding runs on a thread WE own, not on the response generator: it
        # happens while the body is being consumed, so holding gen_lock only
        # inside speech() would serialize nothing, and a client that hangs up
        # mid-stream leaves an abandoned sync generator whose `with gen_lock`
        # is never exited — the engine would wedge for good. A thread always
        # reaches its finally. (Same reasoning as engines/vibevoice/server.py.)
        #
        # An abandoned generation here is actually STOPPED rather than run to
        # completion: generate() polls stop_check_fn once per decoding step.
        async def gen() -> AsyncGenerator[bytes, None]:
            gone = threading.Event()  # the client hung up; stop decoding for nobody
            streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            err = []

            def produce():
                try:
                    with gen_lock:
                        if not gone.is_set():
                            synthesize(voice_path, req.input, knobs, streamer, gone.is_set)
                except BaseException as exc:
                    latch(exc)
                    err.append(exc)
                finally:
                    streamer.end()  # unblocks the reader whether it finished or died

            threading.Thread(target=produce, daemon=True).start()
            q = streamer.audio_queues[0]
            try:
                # to_thread, not q.get(): a blocking get would stall the whole
                # event loop, disconnects included.
                while (chunk := await asyncio.to_thread(q.get)) is not None:
                    yield to_int16_bytes(chunk.detach().cpu().to(torch.float32).numpy().reshape(-1))
                if err:
                    raise err[0]
            finally:
                gone.set()  # every exit — done, error, or hangup

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"x-sample-rate": str(SAMPLE_RATE)})
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    try:
        with gen_lock:
            out = synthesize(voice_path, req.input, knobs)
            audio = out.speech_outputs[0].detach().cpu().to(torch.float32).numpy().reshape(-1)
    except Exception as exc:
        latch(exc)
        raise
    return Response(content=numpy_to_wav_bytes(audio, SAMPLE_RATE), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)  # bound to localhost via docker -p 127.0.0.1:...
