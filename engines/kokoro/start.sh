#!/bin/bash
set -e
# Vendor Kokoro-FastAPI server, loopback only (not published outside the container).
# Model weights are baked into the vendor image, so no download step is needed.
uv run --extra "$DEVICE" --no-sync python -m uvicorn api.src.main:app --host 127.0.0.1 --port 8880 &
# Contract adapter on 8877 (the only port the container publishes).
exec uv run --extra "$DEVICE" --no-sync python -m uvicorn shim_server:app --host 0.0.0.0 --port 8877
