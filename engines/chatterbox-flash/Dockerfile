# Build from the REPO ROOT:
#   docker build -f engines/chatterbox-flash/Dockerfile -t tts-chatterbox-flash:gpu .
#
# Based on the repo's own chatterbox image (build it first from the repo root:
# `docker build -t chatterbox-tts:gpu .`). Chosen over python:3.11-slim because
# this box's WiFi could not re-pull the multi-GB torch wheels — chatterbox-tts:gpu
# is already local and carries torch 2.6.0+cu124, chatterbox-tts 0.1.7 (which
# chatterbox-flash depends on), setuptools<81, fastapi/uvicorn, and voices/.
# Same trick as engines/xtts-v2/Dockerfile, same reason.
FROM chatterbox-tts:gpu

# chatterbox-flash 0.1.0 is a separate package from chatterbox-tts with its own
# API. All heavy deps (torch, transformers, librosa, ...) are satisfied by the
# base image; this only adds the package itself plus small pure-Python gaps
# (inflect, Unidecode).
RUN pip install --no-cache-dir --retries 10 --timeout 120 chatterbox-flash

# The base image bakes its own HF cache at /app/.cache and forces offline mode;
# this engine instead reads the HOST cache mounted at /root/.cache/huggingface
# and must be allowed to download the flash checkpoint on first start.
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_OFFLINE=0

WORKDIR /app
COPY engines/testpage.html /app/testpage.html
COPY engines/chatterbox-flash/server.py ./

EXPOSE 8877

CMD ["python", "server.py"]
