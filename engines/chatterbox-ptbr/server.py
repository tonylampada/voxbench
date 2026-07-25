#!/usr/bin/env python3
"""
Chatterbox pt-BR TTS engine — bench contract adapter (docs/contract.md).

en -> ResembleAI/chatterbox-turbo (English Turbo)
pt -> ResembleAI/Chatterbox-Multilingual-pt-br (dedicated pt-BR finetune)
"""
import io
import json
import os
import wave
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import load_file as load_safetensors

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

PTBR_REPO = "ResembleAI/Chatterbox-Multilingual-pt-br"
BASE_REPO = "ResembleAI/chatterbox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000
LANGS = ("en", "pt")

# Each model carries a baked default voice (Turbo's conds; the base
# multilingual conds.pt for pt) — those are the two builtins.
BUILTINS = {
    "en-default": {"name": "Turbo default", "kind": "builtin", "langs": ["en"], "gender": "unknown"},
    "pt-default": {"name": "Multilingual default", "kind": "builtin", "langs": ["pt"], "gender": "unknown"},
}
DEFAULT_VOICE = {"en": "en-default", "pt": "pt-default"}

PARAMS = {
    "exaggeration": {"type": "number", "min": 0.0, "max": 2.0, "default": 0.5,
                     "description": "emotion intensity of the delivery (0 = flat, 0.5 = neutral, 2 = theatrical)"},
    "cfg_weight": {"type": "number", "min": 0.0, "max": 1.0, "default": 0.5,
                   "description": "classifier-free guidance weight; lower = slower pace, looser adherence to the reference voice"},
    "temperature": {"type": "number", "min": 0.05, "max": 5.0, "default": 0.8,
                    "description": "sampling temperature; higher = more varied, less stable delivery"},
}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
LIBRARY_DIR = Path("/app/voices/library")
_manifest_file = LIBRARY_DIR / "manifest.json"
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file.exists() else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

app = FastAPI(title="Chatterbox pt-BR TTS Engine")
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
    speed: Optional[float] = None  # OpenAI-compat alias; this engine has no speed knob
    stream: bool = False
    params: dict = {}


def load_ptbr(device: str) -> ChatterboxMultilingualTTS:
    """
    Load the pt-BR finetune: t3 + s3gen weights from the finetune repo,
    voice encoder / tokenizer / builtin voice from the base multilingual repo.

    chatterbox-tts 0.1.7 from_pretrained() only knows the base checkpoint; this
    mirrors the loader in the official ResembleAI pt-BR HF Space.
    """
    token = os.getenv("HF_TOKEN") or None
    base_dir = Path(snapshot_download(
        repo_id=BASE_REPO,
        allow_patterns=["ve.pt", "conds.pt", "grapheme_mtl_merged_expanded_v1.json"],
        token=token,
    ))
    t3_path = hf_hub_download(PTBR_REPO, "t3_pt_br.safetensors", token=token)
    s3gen_path = hf_hub_download(PTBR_REPO, "s3gen_v3.pt", token=token)

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(base_dir / "ve.pt", map_location="cpu", weights_only=True))
    ve.to(device).eval()

    t3 = T3(T3Config.multilingual())
    t3_state = load_safetensors(t3_path)
    if "model" in t3_state:
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.to(device).eval()

    s3gen = S3Gen()
    # the v3 checkpoint and 0.1.7's S3Gen disagree on a few buffers; the
    # official Space also loads it with strict=False
    missing, unexpected = s3gen.load_state_dict(
        torch.load(s3gen_path, map_location="cpu", weights_only=True), strict=False
    )
    if missing or unexpected:
        print(f"s3gen_v3 load: {len(missing)} missing / {len(unexpected)} unexpected keys")
    s3gen.to(device).eval()

    tokenizer = MTLTokenizer(str(base_dir / "grapheme_mtl_merged_expanded_v1.json"))
    conds = Conditionals.load(base_dir / "conds.pt", map_location="cpu").to(device)
    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)


def bad(msg: str):
    raise HTTPException(status_code=400, detail=msg)


def resolve(req: SpeechRequest) -> tuple[str, Optional[str]]:
    """-> (lang, audio prompt path or None for the builtin voice)."""
    lang = req.lang
    if lang is not None and lang not in LANGS:
        bad(f"lang '{lang}' not supported — this engine speaks {list(LANGS)}")
    if req.voice is None:
        return lang or "en", None
    v = req.voice
    if v in BUILTINS:
        vlang = BUILTINS[v]["langs"][0]
        if lang is not None and lang != vlang:
            bad(f"voice '{v}' speaks '{vlang}', not '{lang}'")
        return vlang, None
    if v in REFERENCES:
        e = REFERENCES[v]
        if lang is not None and lang != e["lang"]:
            bad(f"voice '{v}' speaks '{e['lang']}', not '{lang}'")
        if e["lang"] not in LANGS:
            bad(f"voice '{v}' speaks '{e['lang']}' — this engine only speaks en and pt")
        return e["lang"], str(LIBRARY_DIR / e["file"])
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
        knobs[k] = v
    if req.speed is not None and req.speed != 1.0:
        bad("this engine has no speed knob — only the no-op speed 1.0 is accepted")
    return knobs


def generate_speech(text: str, lang: str, voice_path: Optional[str], knobs: dict) -> np.ndarray:
    print(f"[{lang}] {text}")
    if lang == "en":
        wav = models["en"].generate(text, audio_prompt_path=voice_path, norm_loudness=True, **knobs)
    else:
        wav = models["pt"].generate(text, language_id="pt", audio_prompt_path=voice_path, **knobs)

    audio = wav.cpu().numpy().squeeze()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


@app.on_event("startup")
def startup():
    print(f"Device: {DEVICE}")
    print("Loading English (Turbo) model...")
    models["en"] = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    print("Loading pt-BR finetune...")
    models["pt"] = load_ptbr(DEVICE)
    print("Chatterbox pt-BR engine ready")


@app.get("/")
def testpage():
    return FileResponse("/app/testpage.html")


@app.get("/health")
def health():
    if len(models) < 2:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "engine": "chatterbox-ptbr"}


@app.get("/v1/voices")
def voices():
    out = [{"id": vid, **meta} for vid, meta in BUILTINS.items()]
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
    lang, voice_path = resolve(req)
    knobs = effective_params(req)
    if req.stream:
        bad("this engine cannot stream — chatterbox synthesizes the whole utterance at once; use stream: false")
    if (req.response_format or "wav") != "wav":
        bad(f"response_format '{req.response_format}' not supported — this engine serves wav")
    audio = generate_speech(req.input, lang, voice_path, knobs)
    return Response(content=numpy_to_wav_bytes(audio), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
