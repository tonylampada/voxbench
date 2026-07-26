FROM python:3.12-slim

# sox CLI: the pysox package (qwen-tts dep) shells out to it
RUN apt-get update && apt-get install -y --no-install-recommends sox && rm -rf /var/lib/apt/lists/*

# torch from PyPI bundles CUDA 12.8 userspace libs (host driver 580.x covers it).
# torchaudio must match torch exactly — unpinned it resolves to 2.11 (CUDA 13
# build) and dies at import with "libcudart.so.13: cannot open shared object file".
# qwen-tts==0.1.1 pins its own transformers==4.57.3 / accelerate==1.12.0.
# flash-attn deliberately skipped (build-time swamp, ~10% speedup only).
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 qwen-tts==0.1.1 fastapi uvicorn

WORKDIR /app
COPY engines/qwen3-tts/server.py .
# Voice-library metadata for /v1/voices (qwen3-tts only needs the manifest, not
# the clips — CustomVoice can't clone; qwen3-tts-origin entries resolve back to
# builtin speakers).
COPY voices/library/manifest.json /app/voices_manifest.json
EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
