# VibeVoice-Realtime-0.5B TTS engine — builds tts-vibevoice:gpu
# Build from the REPO ROOT (needs voices/ in context):
#   docker build -f engines/vibevoice/Dockerfile -t tts-vibevoice:gpu .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# git: the inference code and the voice presets come from a GitHub checkout, not
# from PyPI — Microsoft never published a vibevoice wheel.
# libsndfile1: soundfile/librosa need it.
RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# transformers pinned to 4.51.3: that is what the model's own extra
# (pip install .[streamingtts]) pins, and the streaming inference code reaches
# into DynamicCache internals that moved in later releases.
RUN pip install --no-cache-dir "transformers==4.51.3" accelerate diffusers \
        librosa soundfile "numpy<2" scipy ml-collections absl-py \
        huggingface_hub tqdm fastapi uvicorn

# The code AND the voice presets: microsoft/VibeVoice, pinned. Microsoft pulled
# this repo in September 2025 and restored it in 2026 (see README.md for the
# provenance trail) — the pin is what makes "which VibeVoice did we run" answerable.
# --no-deps because the package's declared deps drag in gradio, aiortc, av and
# pydub for its demo UI; the imports above are the inference path.
ARG VIBEVOICE_COMMIT=94da20d98b2fa7688e9cbfaf7692ddb4954f7600
RUN git clone https://github.com/microsoft/VibeVoice.git /src \
    && git -C /src checkout --detach "$VIBEVOICE_COMMIT" \
    && pip install --no-cache-dir --no-deps /src \
    && mkdir -p /app/presets \
    && cp /src/demo/voices/streaming_model/*.pt /app/presets/ \
    && rm -rf /src

WORKDIR /app
COPY engines/vibevoice/server.py /app/server.py
COPY voices/ /app/voices/

EXPOSE 8877
CMD ["python", "server.py"]
