# syntax=docker/dockerfile:1
# Build from the REPO ROOT:
#   docker build -f engines/base/Dockerfile -t tts-base:gpu .
#
# Dependency layer ONLY. No server, no model download, no baked weights, no CMD
# — this image does nothing on its own and is not meant to be run. It exists so
# chatterbox-flash, chatterbox-nano and xtts-v2 can share one torch install:
# torch 2.6.0+cu124 and the whole chatterbox-tts 0.1.7 tree are multi-GB, and
# this box's link has never been able to pull them three times over.
#
# Replaces the old root Dockerfile / chatterbox-tts:gpu, which was the v1
# single-engine server and dragged server.py, download_models.py and a baked HF
# cache along with the pip layer the three engines actually wanted.
#
# Versions live in requirements.txt as a full transitive lock — read the header
# there before touching it.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY engines/base/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Reference voice clips. engines/tts also bind-mounts the live repo voices/ over
# this for most engines; baking them keeps the image usable without the mount.
COPY voices/ ./voices/
