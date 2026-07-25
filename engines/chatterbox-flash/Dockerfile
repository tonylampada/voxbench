# Build from the REPO ROOT (needs voices/):
#   docker build -f engines/chatterbox-flash/Dockerfile -t tts-chatterbox-flash:gpu .
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# chatterbox-flash 0.1.0 depends on chatterbox-tts>=0.1.7, which pins
# torch==2.6.0 (cu124 wheels) itself; nothing else needs pinning.
RUN pip install --no-cache-dir --retries 10 --timeout 120 chatterbox-flash fastapi "uvicorn[standard]" && \
    # resemble-perth (pulled in via chatterbox-tts, the watermarker) imports
    # pkg_resources, removed in setuptools 81 -> import dies with
    # "'NoneType' object is not callable"
    pip install --no-cache-dir "setuptools<81"

COPY engines/chatterbox-flash/server.py ./
COPY voices/ ./voices/

EXPOSE 8877

CMD ["python", "server.py"]
