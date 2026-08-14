#!/usr/bin/env python3
"""Every word the client sent must survive the processor's script parser.

VibeVoiceProcessor parses its input as a podcast script and silently DROPS any
line that is not "Speaker N: ...", so a paragraph with newlines in it would be
spoken down to its first line and nothing would say so. to_script() exists to
stop that; this checks it against the real parser, not a copy of its regex.

Runs inside the engine image, no GPU and no weights needed:

    docker run --rm -v "$PWD/engines/vibevoice-15b:/t" tts-vibevoice-15b:gpu python /t/test_script_shape.py
"""
import sys

from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

sys.path.insert(0, "/app")
from server import to_script  # noqa: E402

CASES = {
    "one line": "Captain, the deploy is green.",
    "newlines": "Captain, the deploy is green.\nTwo pull requests merged.\n\nOne waiting.",
    "windows newlines": "Primeira linha.\r\nSegunda linha.",
    "curly quotes": "It\u2019s green, the \u201cdeploy\u201d passed.",
    "leading blank": "\n\n  Captain.  ",
    "a line that looks like a speaker": "Speaker 2: this is text, not a cue.",
}

parse = VibeVoiceProcessor(tokenizer=None, audio_processor=None)._parse_script
failures = 0
for name, text in CASES.items():
    lines = parse(to_script(text))
    spoken = " ".join(t.strip() for _, t in lines)
    # Straight quotes are a deliberate substitution; everything else is the
    # client's own words, in order, with no line lost.
    want = " ".join(text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').split())
    ok = len(lines) == 1 and spoken == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {spoken!r}")
    failures += not ok

sys.exit(f"{failures} case(s) lost text" if failures else 0)
