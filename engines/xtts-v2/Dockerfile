# Based on the repo's own chatterbox image (build it first from the repo root:
# `docker build -t chatterbox-tts:gpu .`). Chosen over pytorch/pytorch because
# this box's WiFi could not pull a 3 GB base with five engine builds running in
# parallel — chatterbox-tts:gpu is already local and carries torch 2.6.0+cu124.
FROM chatterbox-tts:gpu

# coqui-tts is the maintained idiap fork (github.com/idiap/coqui-ai-TTS); the
# original `TTS` package is the dead upstream — do not install it. 0.25.2+
# handles torch>=2.6's weights_only=True default when loading XTTS pickles.
# transformers must be <5: the base image carries 5.2.0, coqui-tts's own spec
# accepts it, but XTTS's tortoise layers import isin_mps_friendly which
# transformers 5.x removed — the unpinned install dies at `from TTS.api import TTS`.
RUN pip install --no-cache-dir coqui-tts==0.27.5 "transformers>=4.57,<4.58"

# XTTS-v2 weights are CPML (non-commercial); first load normally prompts for
# TOS acceptance on stdin, which would hang a container. This is the
# documented non-interactive accept.
ENV COQUI_TOS_AGREED=1

WORKDIR /app
COPY server.py .
EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
