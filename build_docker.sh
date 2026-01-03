#!/bin/bash
# Build Docker image with HuggingFace token as build secret
# The token is used during build but NOT stored in the image

set -e

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Create .env with: HF_TOKEN=your_token_here"
    exit 1
fi

# Extract token from .env
HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found in .env"
    exit 1
fi

echo "Building Docker image with models..."
echo "(This will take a while to download ~2GB of models)"

# Create temp file with just the token
echo "$HF_TOKEN" > .hf_token_tmp

# Build with BuildKit secret
DOCKER_BUILDKIT=1 docker build \
    --no-cache \
    --secret id=hf_token,src=.hf_token_tmp \
    -t chatterbox-tts:latest \
    .

# Clean up
rm -f .hf_token_tmp

echo ""
echo "Build complete!"
echo "Run with: docker run -p 8877:8877 chatterbox-tts:latest"
