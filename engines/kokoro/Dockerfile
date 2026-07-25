# Kokoro-82M behind the shared benchmark contract.
# Base image is the maintained community container (Apache-2.0); it bundles
# espeak-ng (needed for pt-BR G2P) and the kokoro-v1_0 weights baked at
# /app/api/src/models/v1_0, serving an OpenAI-compatible API on 8880.
# We add a thin shim on 8877 that translates our /tts contract to it.
FROM ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.4-master

COPY --chown=appuser:appuser engines/kokoro/server.py /app/shim_server.py
COPY --chown=appuser:appuser engines/testpage.html /app/testpage.html
COPY --chown=appuser:appuser engines/kokoro/start.sh /app/start_engine.sh

EXPOSE 8877
CMD ["bash", "/app/start_engine.sh"]
