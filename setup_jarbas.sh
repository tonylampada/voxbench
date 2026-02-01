#!/bin/bash
# Setup Chatterbox TTS Server on Jarbas
# 
# Requires: HuggingFace token with access to ResembleAI/chatterbox-turbo
# The model requires accepting the license agreement at:
# https://huggingface.co/ResembleAI/chatterbox-turbo

set -e

cd "$(dirname "$0")"

# Check for HF token
if [ -z "$HF_TOKEN" ] && [ ! -f .env ]; then
    echo "ERROR: HF_TOKEN not set!"
    echo ""
    echo "To fix:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a token with 'read' access"
    echo "3. Accept the license at https://huggingface.co/ResembleAI/chatterbox-turbo"
    echo "4. Run: echo 'HF_TOKEN=hf_xxxxx' > .env"
    echo "5. Re-run this script"
    exit 1
fi

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Activate venv
source .venv/bin/activate

# Download models
echo "Downloading models (this may take a while)..."
python download_models.py

echo ""
echo "Setup complete! Start the server with:"
echo "  cd ~/clawd/repos/chatterbox_server"
echo "  source .venv/bin/activate"
echo "  python server.py"
echo ""
echo "Or use the systemd service (recommended):"
echo "  sudo systemctl start chatterbox-tts"
