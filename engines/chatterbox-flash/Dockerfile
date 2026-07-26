# Build from the REPO ROOT:
#   docker build -f engines/chatterbox-flash/Dockerfile -t tts-chatterbox-flash:gpu .
#
# Based on the repo's shared dependency layer (build it first from the repo root:
# `docker build -f engines/base/Dockerfile -t tts-base:gpu .`, or just let
# `engines/tts up chatterbox-flash` do it). Chosen over python:3.11-slim because
# this box's WiFi could not re-pull the multi-GB torch wheels three times over —
# tts-base:gpu carries torch 2.6.0+cu124, chatterbox-tts 0.1.7 (which
# chatterbox-flash depends on), setuptools<81, fastapi/uvicorn, and voices/.
# Same trick as engines/xtts-v2/Dockerfile, same reason.
FROM tts-base:gpu

# chatterbox-flash 0.1.0 is a separate package from chatterbox-tts with its own
# API. All heavy deps (torch, transformers, librosa, ...) are satisfied by the
# base image; this only adds the package itself plus small pure-Python gaps
# (inflect, Unidecode).
RUN pip install --no-cache-dir --retries 10 --timeout 120 chatterbox-flash

# Read the HOST HF cache mounted at /root/.cache/huggingface, and be allowed to
# download the flash checkpoint on first start.
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_OFFLINE=0

WORKDIR /app
COPY engines/chatterbox-flash/server.py ./

EXPOSE 8877

CMD ["python", "server.py"]
