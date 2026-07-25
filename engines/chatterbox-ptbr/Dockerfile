# Build from the REPO ROOT (needs voices/):
#   docker build -f engines/chatterbox-ptbr/Dockerfile -t tts-chatterbox-ptbr:gpu .
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# chatterbox-tts 0.1.7 pins torch==2.6.0 (cu124 wheels) itself; nothing else to pin.
RUN pip install --no-cache-dir --retries 10 --timeout 120 chatterbox-tts==0.1.7 fastapi "uvicorn[standard]" && \
    # resemble-perth (the watermarker) imports pkg_resources, removed in
    # setuptools 81 -> model load dies with "'NoneType' object is not callable"
    pip install --no-cache-dir "setuptools<81"

COPY engines/chatterbox-ptbr/server.py ./
COPY engines/testpage.html /app/testpage.html
COPY voices/ ./voices/

EXPOSE 8877

CMD ["python", "server.py"]
