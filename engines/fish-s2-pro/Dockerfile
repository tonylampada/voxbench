# Build from the REPO ROOT:
#   docker build -f engines/fish-s2-pro/Dockerfile -t tts-fish-s2-pro:gpu .
#
# Base is the official fish-speech API-server image (rebuilt 2026-06-09).
# It ships the whole pinned uv environment + tools/api_server.py; we only
# drop in a stdlib adapter and an entrypoint that points the server at the
# host-mounted HF snapshot of fishaudio/s2-pro. Nothing is pip-installed,
# so nothing can break the image's pinned torch stack.
FROM fishaudio/fish-speech:server-cuda

COPY --chmod=644 engines/fish-s2-pro/server.py /app/engine_adapter.py
COPY --chmod=644 engines/testpage.html /app/testpage.html
COPY --chmod=644 voices/library/manifest.json /app/voices_manifest.json
COPY --chmod=755 engines/fish-s2-pro/start.sh /app/engine_start.sh

EXPOSE 8877

ENTRYPOINT ["/app/engine_start.sh"]
