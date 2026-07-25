#!/usr/bin/env python3
"""
Fish Audio S2 Pro adapter — translates the shared benchmark contract
(/health, /voices, /tts, /tts/stream) to the fish-speech API server
running on localhost:8080 in the same container.

stdlib only on purpose: the fish-speech image's uv venv has kui, not
FastAPI, and installing into it risks breaking the pinned resolution.
"""
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8080"
SUPPORTED_LANGS = {"en", "pt"}  # both Tier 1/2 for s2-pro; model infers language from text
MODEL_NAME = "fishaudio/s2-pro"


def upstream_health(timeout=3.0):
    req = urllib.request.Request(f"{UPSTREAM}/v1/health")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status == 200


def synthesize(text, extra):
    body = {"text": text, "format": "wav", "streaming": False}
    for k in ("temperature", "top_p", "seed", "chunk_length"):
        if extra.get(k) is not None:
            body[k] = extra[k]
    req = urllib.request.Request(
        f"{UPSTREAM}/v1/tts",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in (s.strip() for s in sentences) if s]


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            try:
                page = open("/app/testpage.html", "rb").read()
            except OSError as e:
                return self._json(500, {"detail": f"testpage.html not found in image: {e}"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/health":
            try:
                upstream_health()
            except Exception:
                return self._json(503, {"status": "loading"})
            return self._json(200, {"status": "ok", "models_loaded": [MODEL_NAME], "queue_size": 0})
        if self.path == "/voices":
            # s2-pro is a voice-cloning model with no built-in speaker list.
            # The repo's voices/ clips lack transcripts, and fish-speech
            # reference cloning requires (audio, text) pairs — so only the
            # model's unconditioned default voice is offered. See README.
            return self._json(200, {"voices": ["default"]})
        return self._json(404, {"detail": "not found"})

    def do_POST(self):
        if self.path not in ("/tts", "/tts/stream"):
            return self._json(404, {"detail": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._json(400, {"detail": "invalid JSON body"})

        text = body.get("text", "")
        lang = body.get("lang", "en")
        voice = body.get("voice")
        if not text.strip():
            return self._json(400, {"detail": "text is required"})
        if lang not in SUPPORTED_LANGS:
            return self._json(400, {"detail": f"lang must be one of {sorted(SUPPORTED_LANGS)}"})
        if voice not in (None, "", "default"):
            return self._json(400, {"detail": "only voice 'default' is available: reference cloning "
                                              "needs a transcript per clip and voices/ has none"})

        try:
            if self.path == "/tts":
                wav = synthesize(text, body)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                self.end_headers()
                self.wfile.write(wav)
            else:
                # No native framed streaming upstream: chunk by sentence and
                # emit one framed WAV per chunk (honest first-audio latency).
                sentences = split_sentences(text)
                size = max(1, int(body.get("chunk_size", 3)))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Audio-Format", "wav")
                self.end_headers()
                for i in range(0, len(sentences), size):
                    wav = synthesize(" ".join(sentences[i:i + size]), body)
                    self.wfile.write(len(wav).to_bytes(4, "big") + wav)
        except urllib.error.HTTPError as e:
            self._json(502, {"detail": f"fish-speech upstream error: {e.read().decode(errors='replace')[:500]}"})
        except BrokenPipeError:
            pass

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8877), Handler)
    print("fish-s2-pro adapter listening on :8877", flush=True)
    server.serve_forever()
