#!/usr/bin/env python3
"""
Fish Audio S2 Pro adapter — serves the bench contract v2 (docs/contract.md)
by translating it to the fish-speech API server on localhost:8080 in the
same container.

stdlib only on purpose: the fish-speech image's uv venv has kui, not
FastAPI, and installing into it risks breaking the pinned resolution.
The adapter runs on the system python3 instead.
"""
import base64
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "http://127.0.0.1:8080"
ENGINE = "fish-s2-pro"
LANGS = ("en", "pt")  # both Tier 1/2 for s2-pro; the model infers language from text
DEFAULT_LANG = "en"

PARAMS = {
    "temperature": {"type": "number", "min": 0.1, "max": 1.0, "default": 0.8,
                    "description": "sampling temperature of the AR token decoder (lower = more stable)"},
    "top_p": {"type": "number", "min": 0.1, "max": 1.0, "default": 0.8,
              "description": "nucleus sampling cutoff of the AR token decoder"},
}

# S2 Pro clones from a (reference audio, transcript) pair. Every library clip
# reads the bench seed sentence in its own language (bench/mkvoicelib.py), so
# the transcript of any library reference is known by construction.
SEED_TEXT = {
    "en": "The harbour was quiet that morning, and the light came in sideways across the water. Nothing much happened, which was exactly what everyone had hoped for.",
    "pt": "O porto estava quieto naquela manhã, e a luz entrava de lado sobre a água. Não aconteceu grande coisa, que era exatamente o que todo mundo esperava.",
}

# Library manifest: mounted live if voices/ is mounted, else the copy baked at build.
LIB_DIR = Path("/app/voices/library")
_manifest_file = next((p for p in (LIB_DIR / "manifest.json",
                                   Path("/app/voices_manifest.json")) if p.exists()), None)
MANIFEST = json.loads(_manifest_file.read_text()) if _manifest_file else []
REFERENCES = {e["file"][:-4]: e for e in MANIFEST if e.get("file")}

BUILTINS = {"default": {"name": "S2 Pro default", "kind": "builtin",
                        "langs": list(LANGS), "gender": "unknown"}}

# CORS: the console calls this engine from a page it does not serve
# (docs/contract.md, "Cross-origin"). No auth here, so never "*". Same env var
# and same default as the FastAPI adapters; this one is hand-rolled because the
# fish-speech image has no FastAPI (see the module docstring).
CORS_ORIGINS = os.environ.get("TTS_CORS_ORIGINS", "http://127.0.0.1:*,http://localhost:*")
CORS_RE = re.compile("|".join(re.escape(o.strip()).replace(r"\*", ".*")
                              for o in CORS_ORIGINS.split(",") if o.strip()))


class Bad(Exception):
    """A contract 400 with its human sentence."""


def resolve(body):
    """-> (reference manifest entry or None, lang), enforcing the 400 catalogue."""
    lang = body.get("lang")
    if lang is not None and lang not in LANGS:
        raise Bad(f"lang '{lang}' not supported — one of {list(LANGS)}")
    voice = body.get("voice")
    if voice in (None, "default"):
        return None, lang or DEFAULT_LANG
    entry = REFERENCES.get(voice)
    if entry is None:
        raise Bad(f"unknown voice '{voice}' — see /v1/voices")
    if lang is not None and lang != entry["lang"]:
        raise Bad(f"voice '{voice}' speaks '{entry['lang']}', not '{lang}'")
    return entry, entry["lang"]


def knobs_from(body):
    params = body.get("params", {})
    if not isinstance(params, dict):
        raise Bad("params must be an object")
    for k, v in params.items():
        p = PARAMS.get(k)
        if p is None:
            raise Bad(f"unknown param '{k}' — this engine has {sorted(PARAMS)} (see /v1/params)")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise Bad(f"params.{k} must be a number")
        if not p["min"] <= v <= p["max"]:
            raise Bad(f"params.{k} {v} out of range [{p['min']}, {p['max']}]")
    speed = body.get("speed")
    if speed is not None and speed != 1.0:
        raise Bad("this engine has no speed knob — only speed 1.0 is accepted")
    return {k: params[k] for k in PARAMS if k in params}


def upstream_body(text, ref, knobs):
    body = {"text": text, "format": "wav", "streaming": False, **knobs}
    if ref is not None:
        clip = (LIB_DIR / ref["file"]).read_bytes()
        body["references"] = [{"audio": base64.b64encode(clip).decode(),
                               "text": SEED_TEXT[ref["lang"]]}]
    return body


def upstream_tts(body, timeout=600):
    req = urllib.request.Request(f"{UPSTREAM}/v1/tts", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):  # every response, one hook
        origin = self.headers.get("Origin")
        if origin and CORS_RE.fullmatch(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers", "x-sample-rate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            try:
                with urllib.request.urlopen(f"{UPSTREAM}/v1/health", timeout=3):
                    pass
            except Exception:
                return self._json(503, {"status": "loading"})
            return self._json(200, {"status": "ok", "engine": ENGINE})
        if self.path == "/v1/voices":
            out = [{"id": vid, **spec} for vid, spec in BUILTINS.items()]
            out += [{"id": vid, "name": e["voice"], "kind": "reference",
                     "origin": e["engine"], "langs": [e["lang"]],
                     "gender": e.get("gender", "unknown"), "seconds": e["seconds"]}
                    for vid, e in sorted(REFERENCES.items())]
            return self._json(200, {"voices": out})
        if self.path == "/v1/params":
            return self._json(200, {"params": PARAMS})
        return self._json(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            return self._json(404, {"detail": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._json(400, {"detail": "invalid JSON body"})
        try:
            return self.speech(body)
        except Bad as e:
            return self._json(400, {"detail": str(e)})
        except urllib.error.HTTPError as e:
            return self._json(502, {"detail": f"fish-speech upstream error: "
                                              f"{e.read().decode(errors='replace')[:500]}"})
        except BrokenPipeError:
            pass

    def speech(self, body):
        text = body.get("input", "")
        if not isinstance(text, str) or not text.strip():
            raise Bad("input is required and must be non-empty")
        ref, _lang = resolve(body)
        knobs = knobs_from(body)
        fmt = body.get("response_format")
        if body.get("stream"):
            # fish-speech's streaming=true is cosmetic for s2-pro: measured
            # against upstream directly, every PCM byte of an 8-sentence text
            # arrives in one burst at the very end (TTFB == total). The
            # contract says do not fake it.
            raise Bad("this engine cannot stream: fish-speech buffers the whole "
                      "synthesis before sending — request stream: false and play the WAV")
        if fmt not in (None, "wav"):
            raise Bad(f"response_format '{fmt}' not supported — this engine serves wav")
        with upstream_tts(upstream_body(text, ref, knobs)) as r:
            wav = r.read()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8877), Handler)
    print("fish-s2-pro adapter listening on :8877", flush=True)
    server.serve_forever()
