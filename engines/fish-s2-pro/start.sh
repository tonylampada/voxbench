#!/bin/bash
set -euo pipefail

# Weights come from the host HuggingFace cache mounted at /hf (never baked
# into the image). Resolve the snapshot dir at runtime so the SHA can change.
SNAP=$(ls -d /hf/hub/models--fishaudio--s2-pro/snapshots/*/ | head -1)
if [ -z "${SNAP}" ]; then
  echo "FATAL: fishaudio/s2-pro not found under /hf — mount the HF cache" >&2
  exit 1
fi

export LLAMA_CHECKPOINT_PATH="${SNAP%/}"
export DECODER_CHECKPOINT_PATH="${SNAP%/}/codec.pth"
export API_SERVER_NAME=127.0.0.1
export API_SERVER_PORT=8080

# Upstream fish-speech API server (loads the model, serves /v1/tts)
/app/start_server.sh &

# Contract adapter on 8877 (stdlib only, system python3)
exec python3 /app/engine_adapter.py
