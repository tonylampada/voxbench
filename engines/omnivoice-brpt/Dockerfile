# Build from the REPO ROOT (context needs voices/):
#   docker build -f engines/omnivoice-brpt/Dockerfile -t tts-omnivoice-brpt:gpu .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# omnivoice pinned so the license/behavior notes in the README stay true for what
# ships; it needs torch>=2.4 and transformers>=5.3 (both satisfied/resolved here).
RUN pip install --no-cache-dir omnivoice==0.2.1 fastapi uvicorn

WORKDIR /app
COPY engines/omnivoice-brpt/server.py .
# Reference clips for voice cloning (~8 MB total; the "no weights in image" rule
# targets multi-GB checkpoints — model weights come from the mounted HF cache).
COPY voices/ /app/voices/
EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
