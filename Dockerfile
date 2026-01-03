# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Set HuggingFace cache location inside the image
ENV HF_HOME=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY server.py download_models.py ./
COPY voices/ ./voices/

# Download models using BuildKit secret (NOT stored in image layers)
# The secret is mounted at /run/secrets/hf_token during build only
RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN=$(cat /run/secrets/hf_token) && \
    python download_models.py

# Expose port
EXPOSE 8877

# Run server
CMD ["python", "server.py"]
