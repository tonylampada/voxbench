#!/usr/bin/env python3
"""
VibeVoice-Realtime-0.5B TTS engine adapter — serves the bench contract
(docs/contract.md). Loads microsoft/VibeVoice-Realtime-0.5B (24 kHz) and speaks
through the voice presets Microsoft ships with the model's code repo.

It cannot clone: Microsoft removed the acoustic ENCODER weights from the public
checkpoint on purpose ("Removed acoustic tokenizer to avoid users creating
embedding on their own"), so there is no way to turn a reference clip into a
voice. A preset is a prefilled KV cache of a reference read, not an embedding
this engine could compute. Library clips are therefore listed and refused, the
same answer kokoro and piper give.
"""
import asyncio
import copy
import io
import json
import os
import pickletools
import re
import threading
import wave
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference)
from vibevoice.modular.streamer import AudioStreamer
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

MODEL_ID = os.environ.get("VIBEVOICE_MODEL", "microsoft/VibeVoice-Realtime-0.5B")
PRESET_DIR = Path("/app/presets")
SAMPLE_RATE = 24000  # the acoustic tokenizer's rate; fixed, not read from config

# Preset filenames start with the vendor's own language tag, which is not always
# ISO 639-1 ("jp", "kr", "sp"), plus "in" for the Indian-accented ENGLISH voice.
# English is what the model was trained on; the other nine are Microsoft's
# "explore and share feedback" list and are experimental by their own words.
PREFIX_LANG = {"en": "en", "in": "en", "de": "de", "fr": "fr", "it": "it",
               "jp": "ja", "kr": "ko", "nl": "nl", "pl": "pl", "pt": "pt", "sp": "es"}
LANGS = set(PREFIX_LANG.values())
DEFAULT_LANG = "en"

PARAMS = {
    "cfg_scale": {"type": "number", "min": 1.0, "max": 3.0, "default": 1.5,
                  "description": "classifier-free guidance on the diffusion head; higher = closer to the preset voice, may sound tense"},
    "inference_steps": {"type": "integer", "min": 1, "max": 50, "default": 5,
                        "description": "DDPM sampling steps per acoustic latent; more = cleaner audio, fewer = faster"},
    "seed": {"type": "integer", "min": 0, "max": 2147483647, "default": 42,
             "description": "random seed; same text + same knobs + same seed reproduces the same take"},
}

LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

# Every global a preset pickle is allowed to name. The presets are baked into
# the image from a pinned commit of microsoft/VibeVoice, but they are pickles,
# and torch cannot load THESE ones with weights_only=True: they hold
# BaseModelOutputWithPast, and torch's safe unpickler only accepts exactly
# dict/OrderedDict/Counter as a SETITEMS target, subclasses included in the
# refusal. So the file is scanned before it is trusted — anything naming a
# global outside this set never reaches pickle.load.
SAFE_GLOBALS = {
    "collections OrderedDict",
    "torch BFloat16Storage", "torch FloatStorage", "torch HalfStorage",
    "torch._utils _rebuild_tensor_v2",
    "transformers.cache_utils DynamicCache",
    "transformers.modeling_outputs BaseModelOutputWithPast",
}


def preset_globals(path: Path) -> set[str]:
    """Every GLOBAL/STACK_GLOBAL a .pt's pickle names, as 'module name'."""
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
        data = z.read(pkl)
    found, strings = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":
            found.add(arg)
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "SHORT_BINSTRING"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":  # module and name were the last two strings
            found.add(" ".join(strings[-2:]))
    return found


def load_preset(path: Path, device: str):
    unsafe = preset_globals(path) - SAFE_GLOBALS
    if unsafe:
        raise RuntimeError(f"{path.name} names unexpected pickle globals {sorted(unsafe)} — refusing to load it")
    return torch.load(path, map_location=device, weights_only=False)


def discover_presets() -> dict:
    """-> {voice_id: {"file": Path, "lang": .., "gender": .., "name": ..}}.
    Ids are the filenames Microsoft chose ("pt-Spk1_man"), kept verbatim so a
    reader can match a voice back to the upstream file."""
    out = {}
    for f in sorted(PRESET_DIR.rglob("*.pt")):
        prefix, _, rest = f.stem.partition("-")
        lang = PREFIX_LANG.get(prefix)
        if lang is None:
            print(f"skipping preset {f.name}: unknown language prefix '{prefix}'", flush=True)
            continue
        gender = {"man": "male", "woman": "female"}.get(rest.rsplit("_", 1)[-1], "unknown")
        # Microsoft named most non-English presets "Spk0"/"Spk1" — the same two
        # strings across ten languages. The language goes in every display name
        # or the console shows the user a column of identical rows.
        name = rest.rsplit("_", 1)[0].replace("_", " ")
        out[f.stem] = {"file": f, "lang": lang, "gender": gender,
                       "name": f"{name} ({lang})"}
    return out


PRESETS = discover_presets()
# One default per language; en-Carter_man is the vendor demo's own default.
DEFAULT_VOICE = {}
for _vid, _e in PRESETS.items():
    DEFAULT_VOICE.setdefault(_e["lang"], _vid)
if "en-Carter_man" in PRESETS:
    DEFAULT_VOICE["en"] = "en-Carter_man"

app = FastAPI(title="VibeVoice-Realtime TTS Server")

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
_preset_cache = {}
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
    """-> preset voice id, enforcing the contract's 400 catalogue."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {sorted(LANGS)}")
    if req.voice is None:
        lang = lang or DEFAULT_LANG
        if lang not in DEFAULT_VOICE:
            bad(f"no preset voice for lang '{lang}' — pick one from /v1/voices")
        return DEFAULT_VOICE[lang]
    v = req.voice
    if v in REFERENCES:
        e = REFERENCES[v]
        bad(f"'{v}' is a cloning reference rendered by {e['engine']}; VibeVoice cannot clone — "
            f"Microsoft withheld the acoustic encoder from the public checkpoint. Use a builtin preset from /v1/voices.")
    if v not in PRESETS:
        bad(f"unknown voice '{v}' — see /v1/voices")
    if lang is not None and lang != PRESETS[v]["lang"]:
        bad(f"voice '{v}' speaks '{PRESETS[v]['lang']}', not '{lang}'")
    return v


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


def cached_preset(vid: str):
    if vid not in _preset_cache:
        _preset_cache[vid] = load_preset(PRESETS[vid]["file"], DEVICE)
    return _preset_cache[vid]


def prepare(text: str, vid: str):
    """-> (model inputs on device, the preset it was built from)."""
    pre = cached_preset(vid)
    # The vendor's demos normalize curly punctuation before tokenizing; the text
    # tokenizer has no entry for them and they come out as pauses otherwise.
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    inputs = processor.process_input_with_cached_prompt(
        text=text.strip(), cached_prompt=pre, padding=True,
        return_tensors="pt", return_attention_mask=True)
    return ({k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}, pre)


def latch(exc: BaseException) -> None:
    """A device-side assert poisons the CUDA context for the life of the process:
    every later call fails, forever. Record it and let /health say so. An OOM is
    not that — the context survives one."""
    global FATAL
    if "CUDA" in str(exc) and "out of memory" not in str(exc).lower():
        FATAL = (f"CUDA context is dead ({type(exc).__name__}: {str(exc).splitlines()[0][:160]}). "
                 "Every synthesis from here on fails; restart the container "
                 "(engines/tts down vibevoice && engines/tts up vibevoice).")
        print(f"FATAL: {FATAL}", flush=True)


@app.on_event("startup")
async def startup():
    global model, processor
    if not PRESETS:
        raise RuntimeError(f"no voice presets found under {PRESET_DIR} — the image is broken")
    print(f"Loading {MODEL_ID} on {DEVICE}...", flush=True)
    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_ID)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        # flash_attention_2 is what the vendor tested, but it is a compiled wheel
        # we would have to build; sdpa is their own documented fallback.
        attn_implementation="sdpa")
    model.eval()
    print(f"VibeVoice ready: {len(PRESETS)} presets, {sorted(LANGS)}", flush=True)


@app.get("/health")
async def health():
    if FATAL:
        return JSONResponse(status_code=503,
                            content={"status": "unusable", "engine": "vibevoice", "detail": FATAL})
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "vibevoice"}


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


def generate_kwargs(inputs, pre, knobs, streamer=None, stop_check=None) -> dict:
    return dict(
        **inputs, max_new_tokens=None, cfg_scale=knobs["cfg_scale"],
        tokenizer=processor.tokenizer, generation_config={"do_sample": False},
        verbose=False, audio_streamer=streamer, stop_check_fn=stop_check,
        # generate() consumes the cache it is handed, so each request gets its
        # own copy and the cached preset stays reusable.
        all_prefilled_outputs=copy.deepcopy(pre))


def synthesize(vid: str, text: str, knobs: dict, streamer=None, stop_check=None):
    """One decode, caller holding gen_lock. -> the generate() output."""
    inputs, pre = prepare(text, vid)
    # VibeVoice takes no seed and no generator: the diffusion head draws its
    # initial noise with a bare torch.randn and dpm_solver's randn_tensor gets
    # generator=None, so the global generator IS the sampler's RNG. Safe because
    # gen_lock already serializes generation — nobody else is drawing.
    torch.manual_seed(knobs["seed"])
    model.set_ddpm_inference_steps(num_steps=knobs["inference_steps"])
    return model.generate(**generate_kwargs(inputs, pre, knobs, streamer, stop_check))


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if FATAL:
        raise HTTPException(status_code=503, detail=FATAL)
    if not req.input.strip():
        bad("input is required and must be non-empty")
    vid = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        if req.response_format not in (None, "pcm"):
            bad(f"response_format '{req.response_format}' with stream: streaming is raw PCM — omit it or say 'pcm'")

        # Decoding runs on a thread WE own, not on the response generator: it
        # happens while the body is being consumed, so holding gen_lock only
        # inside speech() would serialize nothing, and a client that hangs up
        # mid-stream leaves an abandoned sync generator whose `with gen_lock`
        # is never exited — the engine would wedge for good. A thread always
        # reaches its finally. (Same reasoning as engines/voxcpm2/server.py.)
        #
        # Unlike voxcpm2, an abandoned generation here is actually STOPPED
        # rather than run to completion: VibeVoice's generate() polls
        # stop_check_fn between steps, so the GPU is free within a step.
        async def gen() -> AsyncGenerator[bytes, None]:
            gone = threading.Event()  # the client hung up; stop decoding for nobody
            streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            err = []

            def produce():
                try:
                    with gen_lock:
                        if not gone.is_set():
                            synthesize(vid, req.input, knobs, streamer, gone.is_set)
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
            out = synthesize(vid, req.input, knobs)
            audio = out.speech_outputs[0].detach().cpu().to(torch.float32).numpy().reshape(-1)
    except Exception as exc:
        latch(exc)
        raise
    return Response(content=numpy_to_wav_bytes(audio, SAMPLE_RATE), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)  # bound to localhost via docker -p 127.0.0.1:...
