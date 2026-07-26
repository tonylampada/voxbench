# Voxtral TTS (mistralai/Voxtral-4B-TTS-2603) behind the shared bench contract.
# Build from the REPO ROOT:
#   docker build -f engines/voxtral-tts/Dockerfile -t tts-voxtral-tts:gpu .
#
# Base: vllm/vllm-omni:v0.22.0 — the serving stack Mistral's model card and
# the vllm-omni recipe document for this model, pinned to a release tag
# because vllm-omni moves fast. NOT v0.24.0: that release generates audio
# unrelated to the input text for Voxtral (vllm-omni #4949 — a feedback-frame
# key moved in #4527 and Voxtral's preprocess missed it; fixed only after the
# release). v0.22.0 predates the breakage and is the release the vendor
# recipe verified for this model. server.py launches `vllm serve --omni`
# (OpenAI-compatible /v1/audio/speech, native PCM streaming) as a subprocess
# and adapts the bench contract on 8877. Weights come from the mounted host
# HF cache; nothing is baked into the image.
FROM vllm/vllm-omni:v0.22.0

ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app
COPY engines/voxtral-tts/server.py /app/server.py

EXPOSE 8877
# The image's entrypoint is the vllm CLI — override it so the adapter owns
# the process tree (it spawns `vllm serve` itself, from the same env).
ENTRYPOINT []
CMD ["python3", "/app/server.py"]
