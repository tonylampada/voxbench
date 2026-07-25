# torch 2.5, not >=2.6: XTTS checkpoints are pickle-based and torch 2.6 flipped
# torch.load to weights_only=True by default, which breaks loading them.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# coqui-tts is the maintained idiap fork (github.com/idiap/coqui-ai-TTS).
# The original `TTS` package is the dead upstream — do not install it.
# torchaudio pinned to match the base image's torch so pip doesn't upgrade torch.
RUN pip install --no-cache-dir coqui-tts==0.27.5 torchaudio==2.5.1 fastapi uvicorn

# XTTS-v2 weights are CPML (non-commercial); first load normally prompts for TOS
# acceptance on stdin, which would hang a container. This is the documented
# non-interactive accept.
ENV COQUI_TOS_AGREED=1

WORKDIR /app
COPY server.py .
EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
