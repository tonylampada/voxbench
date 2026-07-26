# Build from the REPO ROOT:
#   docker build -f engines/chatterbox-nano/Dockerfile -t tts-chatterbox-nano:gpu .
#
# Based on the repo's shared dependency layer (build it first from the repo root:
# `docker build -f engines/base/Dockerfile -t tts-base:gpu .`, or just let
# `engines/tts up chatterbox-nano` do it). Chosen over python:3.11-slim because
# this box's link was at ~50 KB/s with several engine builds pulling in
# parallel — tts-base:gpu carries torch 2.6.0+cu124, the whole chatterbox-tts
# 0.1.7 dependency tree, setuptools<81, and voices/.
FROM tts-base:gpu

# nano=True only exists on GitHub master — PyPI's chatterbox-tts 0.1.7
# (2026-03-26) predates the Nano release; pin the "Update ChatterboxNano"
# commit (2026-07-21). Installed from the tarball (no git in the base image)
# with --no-deps: every pin in master's pyproject is already satisfied by the
# base (torch 2.6.0, transformers 5.2.0, safetensors 0.5.3, diffusers 0.29.0,
# omegaconf, resemble-perth 1.0.1, ...) except gradio, which only the demo
# scripts import. --force-reinstall because master still calls itself 0.1.7,
# so pip otherwise treats the base's PyPI 0.1.7 as already satisfying the URL.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    "chatterbox-tts @ https://github.com/resemble-ai/chatterbox/archive/5de7a54aa4e5.tar.gz"

# Nano's weights are not baked into any image here, so point at the host cache
# that engines/tts mounts and allow the first download.
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_OFFLINE=0

WORKDIR /app
COPY engines/chatterbox-nano/server.py ./server.py

EXPOSE 8877

CMD ["python", "server.py"]
