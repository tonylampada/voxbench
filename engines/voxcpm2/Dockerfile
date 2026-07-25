# VoxCPM2 TTS engine — builds tts-voxcpm2:gpu
# Build from the REPO ROOT (needs voices/ in context):
#   docker build -f engines/voxcpm2/Dockerfile -t tts-voxcpm2:gpu .
# Base pins torch 2.8.0 + CUDA 12.8 (voxcpm needs torch>=2.5, CUDA>=12).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# libsndfile1: soundfile/librosa need it to read flac/wav reference clips.
# gcc/g++: torch.compile (inductor) needs a host compiler — voxcpm compiles the
# model at load time when optimize=True.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# voxcpm installed --no-deps on purpose: its declared deps drag in torchcodec,
# funasr, modelscope, gradio and datasets, which are only used by its training
# code, denoiser and demo UI. The packages below are the actual imports of the
# inference path (verified against voxcpm 2.0.3 sources).
# transformers pinned <5 because voxcpm imports LlamaTokenizerFast from the 4.x API.
RUN pip install --no-cache-dir --no-deps voxcpm==2.0.3 && \
    pip install --no-cache-dir einops librosa soundfile "transformers>=4.36.2,<5" \
        huggingface_hub regex tqdm inflect wetext fastapi uvicorn

WORKDIR /app
COPY engines/voxcpm2/server.py /app/server.py
COPY voices/ /app/voices/

EXPOSE 8877
CMD ["python", "server.py"]
