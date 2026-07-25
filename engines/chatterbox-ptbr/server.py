#!/usr/bin/env python3
"""
Chatterbox pt-BR TTS engine — bench contract adapter.

en -> ResembleAI/chatterbox-turbo (English Turbo)
pt -> ResembleAI/Chatterbox-Multilingual-pt-br (dedicated pt-BR finetune)
"""
import io
import re
import os
import wave
from pathlib import Path
from typing import Optional, Generator

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
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
VOICES_DIR = Path("voices")

app = FastAPI(title="Chatterbox pt-BR TTS Engine")
models = {}


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


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]


def get_voice_path(voice: Optional[str]) -> Optional[str]:
    if not voice:
        return None
    for candidate in [VOICES_DIR / voice] + [VOICES_DIR / f"{voice}{ext}" for ext in (".flac", ".wav", ".mp3")]:
        if candidate.exists():
            return str(candidate)
    raise HTTPException(status_code=400, detail=f"Unknown voice: {voice}")


def generate_speech(text, lang, voice_path, exaggeration=None, cfg_weight=None, temperature=None) -> np.ndarray:
    print(f"[{lang}] {text}")
    kwargs = {}
    if exaggeration is not None:
        kwargs["exaggeration"] = exaggeration
    if cfg_weight is not None:
        kwargs["cfg_weight"] = cfg_weight
    if temperature is not None:
        kwargs["temperature"] = temperature

    if lang == "en":
        wav = models["en"].generate(text, audio_prompt_path=voice_path, norm_loudness=True, **kwargs)
    else:
        wav = models["pt"].generate(text, language_id="pt", audio_prompt_path=voice_path, **kwargs)

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


def check_lang(lang: str):
    if lang not in ("en", "pt"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")


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
    return {"status": "ok", "engine": "chatterbox-ptbr", "models_loaded": list(models.keys()), "queue_size": 0}


@app.get("/voices")
def list_voices():
    voices = sorted(f.name for f in VOICES_DIR.iterdir() if f.suffix in (".flac", ".wav", ".mp3")) if VOICES_DIR.exists() else []
    return {"voices": voices}


@app.post("/tts")
def tts(request: TTSRequest):
    check_lang(request.lang)
    audio = generate_speech(
        request.text, request.lang, get_voice_path(request.voice),
        request.exaggeration, request.cfg_weight, request.temperature,
    )
    return Response(content=numpy_to_wav_bytes(audio), media_type="audio/wav")


def stream_chunks(request: TTSRequest, voice_path: Optional[str]) -> Generator[bytes, None, None]:
    chunks = chunk_sentences(split_into_sentences(request.text), request.chunk_size)
    for chunk in chunks:
        audio = generate_speech(
            chunk, request.lang, voice_path,
            request.exaggeration, request.cfg_weight, request.temperature,
        )
        wav_bytes = numpy_to_wav_bytes(audio)
        yield len(wav_bytes).to_bytes(4, 'big') + wav_bytes


@app.post("/tts/stream")
def tts_stream(request: TTSRequest):
    check_lang(request.lang)
    voice_path = get_voice_path(request.voice)
    return StreamingResponse(
        stream_chunks(request, voice_path),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav", "X-Sample-Rate": str(SAMPLE_RATE)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
