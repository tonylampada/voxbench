#!/usr/bin/env python3
"""
F5-TTS pt-BR adapter — speaks the shared benchmark HTTP contract.

English  -> base F5-TTS v1 checkpoint (SWivid/F5-TTS).
Portuguese -> community finetune firstpixel/F5-TTS-pt-br (old F5TTS_Base arch,
             stock 2545-token vocab bundled with the f5-tts package).
"""
import io
import re
import wave
from importlib.resources import files
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from huggingface_hub import hf_hub_download
from num2words import num2words
from pydantic import BaseModel

app = FastAPI(title="F5-TTS pt-BR TTS Server")

models = {}
VOICES_DIR = Path("voices")

# Bundled reference clip shipped inside the f5-tts package. Used whenever the
# requested voice has no sibling .txt transcript (F5-TTS cloning requires the
# transcript of the reference clip).
DEFAULT_REF = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
DEFAULT_REF_TEXT = "Some call me nature, others call me mother nature."


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    voice: Optional[str] = None
    chunk_size: int = 3  # sentences per chunk, /tts/stream only
    speed: Optional[float] = None  # optional engine knob
    nfe_step: Optional[int] = None  # optional engine knob (diffusion steps)


def normalize_pt(text: str) -> str:
    """The pt-BR finetune has no text normalization: lowercase + expand numbers."""
    def repl(m):
        return num2words(int(m.group()), lang="pt_BR")
    return re.sub(r"\d+", repl, text).lower()


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), max(1, chunk_size))]


def resolve_voice(voice: Optional[str]) -> Tuple[str, str]:
    """Return (ref_audio_path, ref_text). Falls back to the bundled default
    voice when the clip is missing or has no sibling .txt transcript."""
    if not voice:
        return DEFAULT_REF, DEFAULT_REF_TEXT

    candidates = [VOICES_DIR / voice] + [VOICES_DIR / f"{voice}{ext}" for ext in (".flac", ".wav", ".mp3")]
    for p in candidates:
        if p.exists():
            txt = p.with_suffix(".txt")
            if txt.exists():
                return str(p), txt.read_text().strip()
            print(f"voice '{voice}' has no transcript ({txt.name} missing) — using default voice")
            return DEFAULT_REF, DEFAULT_REF_TEXT

    print(f"voice '{voice}' not found — using default voice")
    return DEFAULT_REF, DEFAULT_REF_TEXT


def generate_speech(text: str, lang: str, voice: Optional[str],
                    speed: Optional[float], nfe_step: Optional[int]) -> Tuple[np.ndarray, int]:
    ref_file, ref_text = resolve_voice(voice)
    if lang == "pt":
        text = normalize_pt(text)
    print(f"[{lang}] {text}")

    kwargs = {}
    if speed is not None:
        kwargs["speed"] = speed
    if nfe_step is not None:
        kwargs["nfe_step"] = nfe_step

    wav, sr, _ = models[lang].infer(
        ref_file=ref_file, ref_text=ref_text, gen_text=text, **kwargs
    )
    audio = np.asarray(wav).squeeze()
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio, sr


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


@app.on_event("startup")
async def startup():
    print("Loading English (base F5-TTS v1) model...")
    from f5_tts.api import F5TTS
    models["en"] = F5TTS()
    print("Loading Portuguese (firstpixel/F5-TTS-pt-br) model...")
    ckpt = hf_hub_download("firstpixel/F5-TTS-pt-br", "pt-br/model_last.safetensors")
    # Finetune predates the v1 arch: must load with the old F5TTS_Base config.
    models["pt"] = F5TTS(model="F5TTS_Base", ckpt_file=ckpt)
    print("F5-TTS pt-BR server ready!")


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys()), "queue_size": 0}


@app.get("/voices")
async def list_voices():
    # Only clips with a sibling .txt transcript are usable for cloning;
    # "default" is the f5-tts bundled reference clip.
    voices = ["default"]
    if VOICES_DIR.exists():
        for f in sorted(VOICES_DIR.iterdir()):
            if f.suffix in (".flac", ".wav", ".mp3") and f.with_suffix(".txt").exists():
                voices.append(f.name)
    return {"voices": voices}


@app.post("/tts")
async def tts(request: TTSRequest):
    if request.lang not in ("en", "pt"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")
    audio, sr = generate_speech(request.text, request.lang, request.voice,
                                request.speed, request.nfe_step)
    return Response(
        content=numpy_to_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


def stream_chunks(request: TTSRequest) -> Generator[bytes, None, None]:
    chunks = chunk_sentences(split_into_sentences(request.text), request.chunk_size)
    for chunk in chunks:
        audio, sr = generate_speech(chunk, request.lang, request.voice,
                                    request.speed, request.nfe_step)
        wav_bytes = numpy_to_wav_bytes(audio, sr)
        yield len(wav_bytes).to_bytes(4, "big") + wav_bytes


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """No native streaming in F5-TTS: text is chunked by sentence, one framed WAV
    per chunk. Format: [4 bytes big-endian length][WAV]..."""
    if request.lang not in ("en", "pt"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'pt'")
    return StreamingResponse(
        stream_chunks(request),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8877)
