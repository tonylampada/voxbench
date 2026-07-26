# Higgs TTS 3 (bosonai/higgs-tts-3-4b) behind the shared bench contract.
# Build from the REPO ROOT:
#   docker build -f engines/higgs-tts-3/Dockerfile -t tts-higgs-tts-3:gpu .
#
# Base: lmsysorg/sglang-omni:dev — the serving stack Boson AI documents for
# this model. server.py launches `sgl-omni serve` (OpenAI-compatible
# /v1/audio/speech, native PCM streaming) as a subprocess and adapts the
# bench contract on 8877. Weights come from the mounted host HF cache;
# nothing is baked into the image.
FROM lmsysorg/sglang-omni:dev

ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app
COPY engines/higgs-tts-3/server.py /app/server.py

EXPOSE 8877
# The sgl-omni stack lives in the image's /opt/omni python3.11 venv (the
# system python3 has neither sgl_omni nor its CLI) — run the adapter with it
# so fastapi/uvicorn/sgl-omni all come from one env.
CMD ["/opt/omni/bin/python3", "/app/server.py"]
