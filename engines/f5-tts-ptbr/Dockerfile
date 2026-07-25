# F5-TTS pt-BR benchmark engine.
# Base: official F5-TTS image (torch + CUDA + f5-tts already installed).
FROM ghcr.io/swivid/f5-tts:main

# num2words: pt-BR number expansion (the finetune has no text normalization).
# fastapi/uvicorn: come transitively via gradio in the base image, but install
# explicitly so the adapter does not depend on gradio staying around.
RUN pip install --no-cache-dir fastapi uvicorn num2words

WORKDIR /app
COPY server.py .

EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
