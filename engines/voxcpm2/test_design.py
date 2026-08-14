#!/usr/bin/env python3
"""params.voice_prompt: the description reaches the model, and how each piece
of a long message is conditioned on the ones before it.

No GPU, no model: `voxcpm` is stubbed and the handler is called directly, so
what this checks is the one thing the server owns — the text and the prompt
cache it hands the model. Whether that text produces a voice you'd call "tired"
is the model's business and the samples' job.

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


class Wav:
    """Stands in for the model's audio tensor, which the server squeezes."""

    def squeeze(self, _dim):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros(480, dtype=np.float32)


class Recorder:
    """Remembers every text it was asked to decode and what it was conditioned on."""

    def __init__(self):
        self.texts, self.caches = [], []
        self.tts_model = self

    def build_prompt_cache(self, prompt_text=None, prompt_wav_path=None, reference_wav_path=None):
        return {"ref": reference_wav_path, "prompt_wav": prompt_wav_path, "prompt_text": prompt_text}

    def generate_with_prompt_cache(self, target_text, prompt_cache=None, **kw):
        self.texts.append(target_text)
        self.caches.append(prompt_cache)
        return Wav(), None, "feat"


def speak(**body) -> list[str]:
    """-> the texts the model was handed for this request."""
    server.model = Recorder()
    server.speech(server.SpeechRequest(**body))
    return server.model.texts


def caches(**body) -> list:
    """-> the prompt cache each piece of this request was conditioned on."""
    speak(**body)
    return server.model.caches


def refused(**body) -> str:
    try:
        speak(**body)
    except HTTPException as exc:
        assert exc.status_code == 400, f"expected 400, got {exc.status_code}"
        return exc.detail
    raise AssertionError(f"expected a 400, got audio: {body}")


# Two reference clips the resolver will accept, without a voice library on disk:
# one whose words the manifest knows, one from before transcripts existed.
server.REFERENCES["a-voice"] = {"file": "a-voice.wav", "lang": "pt", "voice": "A", "seconds": 8.0,
                                "text": "O porto estava quieto."}
server.REFERENCES["mute-voice"] = {"file": "m.wav", "lang": "pt", "voice": "M", "seconds": 8.0}

# No description: the text is the text. This is the board's production path.
assert speak(input="Bom dia.") == ["Bom dia."]
assert speak(input="Bom dia.", voice="a-voice") == ["Bom dia."]

# A description rides in a leading parenthesis — voxcpm's own voice-design
# convention (build_final_text in voxcpm/cli.py).
assert speak(input="Bom dia.", params={"voice_prompt": "a tired man"}) == ["(a tired man)Bom dia."]
assert speak(input="Bom dia.", params={"voice_prompt": "  a tired man  "}) == \
    ["(a tired man)Bom dia."]

long_text = "Uma frase completa e razoavelmente longa para encher o pedaco. " * 12

# With continuity OFF every piece has to carry the description itself, because
# each is an independent decode and one that lost it would change voice.
pieces = speak(input=long_text, params={"voice_prompt": "a tired man", "continuity": False})
assert len(pieces) > 1, f"expected several pieces, got {len(pieces)}"
assert all(p.startswith("(a tired man)") for p in pieces), pieces
# ...and the description counts against the per-decode budget, not on top of it.
assert all(len(p) <= server.MAX_CHUNK_CHARS for p in pieces), [len(p) for p in pieces]
# Nothing links the pieces: every one is conditioned on the same nothing.
assert caches(input=long_text, params={"voice_prompt": "a tired man", "continuity": False}) == \
    [None] * len(pieces)

# With continuity ON the description is spent once: the first piece is written
# out and becomes the reference clip every later piece clones, so repeating the
# description would only eat decode budget.
on = speak(input=long_text, params={"voice_prompt": "a tired man"})
assert on[0].startswith("(a tired man)"), on[0]
assert not any(p.startswith("(") for p in on[1:]), on[1:]
anchored = caches(input=long_text, params={"voice_prompt": "a tired man"})
assert anchored[0] is None, anchored[0]  # nothing to clone yet — the description invents it
assert all(c and c["ref"].endswith(".wav") for c in anchored[1:]), anchored
assert len({c["ref"] for c in anchored[1:]}) == 1, "the anchor must not be rebuilt per piece"

# A cloned voice: every piece gets the same anchor, the clip AND its transcript.
held = caches(input=long_text, voice="a-voice")
assert all(c["ref"].endswith("a-voice.wav") and c["prompt_wav"].endswith("a-voice.wav")
           and c["prompt_text"] == "O porto estava quieto." for c in held), held
# Continuity off is the old take, exactly: the clip on its own, no transcript.
off = caches(input=long_text, voice="a-voice", params={"continuity": False})
assert all(c["prompt_wav"] is None and c["prompt_text"] is None for c in off), off
# A clip whose words nobody wrote down still clones — it just cannot be told
# what it says, so the transcript half of the anchor is left out.
assert caches(input="Bom dia.", voice="mute-voice")[0]["prompt_wav"] is None

# Text reaches the model the way voxcpm's own generate() would have cleaned it:
# we call one level below generate(), so its newline handling comes with us.
assert speak(input="Bom\n  dia.") == ["Bom dia."]

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

refused(input="Bom dia.", params={"continuity": 1})  # a switch, not a number

print("ok — voice_prompt reaches the model, and one message stays conditioned on itself")
