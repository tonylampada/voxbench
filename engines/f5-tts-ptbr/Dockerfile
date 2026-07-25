# F5-TTS pt-BR benchmark engine.
# Base: official F5-TTS image (CUDA + f5-tts 1.1.22 preinstalled).
FROM ghcr.io/swivid/f5-tts:main

# The official image is broken as shipped: torch 2.13.0+cu130 alongside
# torchaudio 2.4.0 and torchvision 0.19.0, both built against torch 2.4.
# Importing f5_tts dies (libtorchaudio.so undefined symbol; torchvision::nms
# missing operator via transformers). Fix: pin the newest consistent cu130
# pair (torchaudio caps at 2.11.0 on cu130) and drop torchvision — nothing in
# this engine uses it and transformers treats it as optional.
# num2words: pt-BR number expansion (the finetune has no text normalization).
RUN pip install --no-cache-dir torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 \
    && pip uninstall -y torchvision \
    && pip install --no-cache-dir num2words

WORKDIR /app
COPY engines/f5-tts-ptbr/server.py .
COPY engines/testpage.html /app/testpage.html

EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
