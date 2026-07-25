#!/bin/bash
set -euo pipefail

# Weights come from the host HuggingFace cache mounted at /hf (never baked
# into the image). Resolve the snapshot dir at runtime so the SHA can change.
# `|| true` because set -e would otherwise kill the script on an unmatched
# glob before the readable error below ever prints.
SNAP=$(ls -d /hf/hub/models--fishaudio--s2-pro/snapshots/*/ 2>/dev/null | head -1 || true)
if [ -z "${SNAP}" ]; then
  echo "FATAL: fishaudio/s2-pro not found under /hf — mount the HF cache" >&2
  exit 1
fi

# The shipped config declares max_seq_len=32768, which makes the server
# preallocate a ~4.8GB KV cache and pushes total VRAM to ~16.5GB. Capping the
# context to 4096 tokens (~3 min of audio at the codec's 21Hz — far more than
# any bench utterance) brings it to the advertised ~12GB. Weights untouched.
MAX_SEQ_LEN="${FISH_MAX_SEQ_LEN:-4096}"
CKPT=/tmp/ckpt
mkdir -p "$CKPT"
ln -sf "${SNAP%/}"/* "$CKPT/"
rm -f "$CKPT/config.json"
sed "s/\"max_seq_len\": 32768/\"max_seq_len\": ${MAX_SEQ_LEN}/" "${SNAP%/}/config.json" > "$CKPT/config.json"

export LLAMA_CHECKPOINT_PATH="$CKPT"
export DECODER_CHECKPOINT_PATH="$CKPT/codec.pth"
export API_SERVER_NAME=127.0.0.1
export API_SERVER_PORT=8080
# Without this, `uv run` re-resolves the lockfile on every container start and
# re-downloads ~300MB of nvidia wheels (cu129 -> cu126 downgrade). The baked
# venv already works; never sync at runtime.
export UV_NO_SYNC=1

# Upstream fish-speech API server (loads the model, serves /v1/tts)
/app/start_server.sh &

# Contract adapter on 8877 (stdlib only, system python3)
exec python3 /app/engine_adapter.py
