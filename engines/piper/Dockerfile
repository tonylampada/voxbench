FROM python:3.12-slim

# piper-tts 1.6.0 is the live piper1-gpl project (rhasspy/piper is archived);
# pinned so the GPL-3.0 licensing note in the README stays true for what ships.
RUN pip install --no-cache-dir piper-tts==1.6.0 fastapi uvicorn

# Voices are small ONNX files (~20-65 MB each, ~250 MB total), baked at build
# time because this CPU engine's fleet run command mounts no cache volume.
RUN python -m piper.download_voices --download-dir /voices \
    pt_BR-faber-medium pt_BR-cadu-medium pt_BR-edresson-low pt_BR-jeff-medium \
    en_US-lessac-medium

WORKDIR /app
COPY engines/piper/server.py .
COPY engines/testpage.html /app/testpage.html
EXPOSE 8877
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8877"]
