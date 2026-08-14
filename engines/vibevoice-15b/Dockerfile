# VibeVoice-1.5B TTS engine — builds tts-vibevoice-15b:gpu
# Build from the REPO ROOT (needs voices/ in context):
#   docker build -f engines/vibevoice-15b/Dockerfile -t tts-vibevoice-15b:gpu .
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# git: the inference code comes from GitHub checkouts, not PyPI — Microsoft
# never published a vibevoice wheel. libsndfile1: soundfile/librosa need it to
# read the reference clips this engine clones from.
RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# transformers pinned to 4.51.3, the version the checkpoint's own config.json
# was saved with and what the vendor's install pins.
RUN pip install --no-cache-dir "transformers==4.51.3" accelerate diffusers \
        librosa soundfile "numpy<2" scipy ml-collections absl-py \
        huggingface_hub tqdm fastapi uvicorn

# Code from microsoft/VibeVoice, pinned to the same commit engines/vibevoice
# runs. TWO things it does not contain, both taken from the community fork it
# was itself restored from (MIT, see README.md for the provenance trail):
#   - modeling_vibevoice_inference.py — the non-streaming inference path, i.e.
#     the ONLY way to drive the 1.5B checkpoint. Microsoft ships only the
#     streaming 0.5B's variant.
#   - demo/voices/*.wav — the 1.5B's preset voices. Microsoft's demo/voices now
#     holds only streaming_model/*.pt, which are prefilled KV caches for the
#     0.5B and mean nothing to this model.
# --no-deps because the package's declared deps drag in gradio, aiortc, av and
# pydub for its demo UI; the imports above are the inference path.
ARG VIBEVOICE_COMMIT=94da20d98b2fa7688e9cbfaf7692ddb4954f7600
ARG COMMUNITY_COMMIT=631804b9c1f042e381207fe87c54603fe6accbc1
RUN git clone https://github.com/microsoft/VibeVoice.git /src \
    && git -C /src checkout --detach "$VIBEVOICE_COMMIT" \
    && git clone https://github.com/vibevoice-community/VibeVoice.git /community \
    && git -C /community checkout --detach "$COMMUNITY_COMMIT" \
    && cp /community/vibevoice/modular/modeling_vibevoice_inference.py \
          /src/vibevoice/modular/modeling_vibevoice_inference.py \
    && mkdir -p /app/presets \
    && cp /community/demo/voices/*.wav /app/presets/ \
    && pip install --no-cache-dir --no-deps /src \
    && rm -rf /src /community

WORKDIR /app
COPY engines/vibevoice-15b/server.py /app/server.py
COPY voices/ /app/voices/

EXPOSE 8877
CMD ["python", "server.py"]
