# Gepard 1.0 (nineninesix) TTS engine — image tts-gepard:gpu
#
# Plain python base: the CUDA *runtime* rides in with the cu128 torch wheels;
# the CUDA *driver* libs are mounted at /opt/nvdrv at run time (see README).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# torch first, index-only: the pytorch index hosts torch + its deps as a
# coherent set; adding PyPI as an extra index is what swaps in mismatched
# builds. cu128 wheels run on any driver >= 12.8 (incl. the CUDA-13 580 driver).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch torchaudio

# NeMo supplies the NanoCodec (AudioCodecModel). nemo-toolkit pins an older
# transformers — deliberately overwritten by the gepard install below.
RUN pip install --no-cache-dir "nemo-toolkit[tts]==2.4.0"

# Vendor reference inference stack, pinned commit (their upstream moves).
# The [inference] extra re-pins transformers==5.3.0 — the version the model
# was published against; newer releases (<=5.14.1) crash on Gepard's
# full-attention-only qwen3_5 config, which the package's FullAttnCache
# workaround only covers for 5.3.0.
RUN git clone https://github.com/nineninesix-ai/gepard-inference.git /opt/gepard \
    && git -C /opt/gepard checkout fa19f5793db3f9f8413da6da69bd32659bd4ac4d \
    && pip install --no-cache-dir "/opt/gepard[inference]"

COPY server.py /app/server.py
WORKDIR /app
EXPOSE 8877
CMD ["python", "server.py"]
