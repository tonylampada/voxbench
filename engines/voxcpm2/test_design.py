#!/usr/bin/env python3
"""params.voice_prompt: the description reaches the model, on every piece.

No GPU, no model: `voxcpm` is stubbed and the handler is called directly, so
what this checks is the one thing the server owns — the text it hands the
model. Whether that text produces a voice you'd call "tired" is the model's
business and the samples' job.

    pip install fastapi numpy && python engines/voxcpm2/test_design.py
"""
import sys
import types
from pathlib import Path

import numpy as np

sys.modules["voxcpm"] = types.SimpleNamespace(VoxCPM=None)  # never loaded: no startup, no GPU
sys.modules.setdefault("torch", types.SimpleNamespace(manual_seed=lambda seed: None))
sys.path.insert(0, str(Path(__file__).parent))
import server  # noqa: E402

from fastapi import HTTPException  # noqa: E402


class Recorder:
    """Remembers every text it was asked to decode."""

    def __init__(self):
        self.texts = []

    def generate(self, text, reference_wav_path=None, **kw):
        self.texts.append(text)
        return np.zeros(480, dtype=np.float32)


def speak(**body) -> list[str]:
    """-> the texts the model was handed for this request."""
    server.model = Recorder()
    server.speech(server.SpeechRequest(**body))
    return server.model.texts


def refused(**body) -> str:
    try:
        speak(**body)
    except HTTPException as exc:
        assert exc.status_code == 400, f"expected 400, got {exc.status_code}"
        return exc.detail
    raise AssertionError(f"expected a 400, got audio: {body}")


# A reference clip the resolver will accept, without a voice library on disk.
server.REFERENCES["a-voice"] = {"file": "a-voice.wav", "lang": "pt", "voice": "A", "seconds": 8.0}

# No description: the text is the text. This is the board's production path.
assert speak(input="Bom dia.") == ["Bom dia."]
assert speak(input="Bom dia.", voice="a-voice") == ["Bom dia."]

# A description rides in a leading parenthesis — voxcpm's own voice-design
# convention (build_final_text in voxcpm/cli.py).
assert speak(input="Bom dia.", params={"voice_prompt": "a tired man"}) == ["(a tired man)Bom dia."]
assert speak(input="Bom dia.", params={"voice_prompt": "  a tired man  "}) == \
    ["(a tired man)Bom dia."]

# EVERY piece carries it. A long text is several independent decodes, and one
# that lost the description would change voice mid-message.
long_text = "Uma frase completa e razoavelmente longa para encher o pedaco. " * 12
pieces = speak(input=long_text, params={"voice_prompt": "a tired man"})
assert len(pieces) > 1, f"expected several pieces, got {len(pieces)}"
assert all(p.startswith("(a tired man)") for p in pieces), pieces
# ...and the description counts against the per-decode budget, not on top of it.
assert all(len(p) <= server.MAX_CHUNK_CHARS for p in pieces), [len(p) for p in pieces]

# Two voice sources in one request is a caller error, and the 400 names both.
detail = refused(input="Bom dia.", voice="a-voice", params={"voice_prompt": "a tired man"})
assert "a-voice" in detail and "voice_prompt" in detail, detail

# An empty description is not a second source — it is no description at all.
assert speak(input="Bom dia.", voice="a-voice", params={"voice_prompt": "  "}) == ["Bom dia."]

refused(input="Bom dia.", params={"voice_prompt": "a (tired) man"})   # would end the control early
refused(input="Bom dia.", params={"voice_prompt": "x" * 400})         # a description, not a script
refused(input="Bom dia.", params={"voice_prompt": 7})

# Same request, same text handed over twice — the seed does the rest at decode time.
assert speak(input="Bom dia.", params={"voice_prompt": "a tired man", "seed": 7}) == \
    speak(input="Bom dia.", params={"voice_prompt": "a tired man", "seed": 7})

print("ok — voice_prompt reaches every piece, and refuses to share a request with a voice")
