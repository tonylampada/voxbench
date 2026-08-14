#!/usr/bin/env python3
"""voicematrix reads engines/tts's registry and names its own files — both drift silently.

The port comes from parsing another file's table, and the <audio src> in the page
has to be the exact filename the generator wrote or the cell is a dead player
that still looks fine. Nothing else here is worth a test.

    python3 bench/test_voicematrix.py
"""
import re
import sys
from html import escape as html_escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voicematrix as vm  # noqa: E402

VOICES = [
    {"id": "en-Carter_man", "name": "Carter (en)", "kind": "builtin",
     "langs": ["en"], "gender": "male"},
    {"id": "in-Samuel_man", "name": "Samuel (en)", "kind": "builtin",
     "langs": ["en"], "gender": "male"},
]


def main():
    assert vm.registry_port("vibevoice") == 8895
    assert vm.registry_port("piper") == 8888
    print("ok   ports still come off engines/tts's registry")

    html = vm.page("vibevoice", "en", VOICES, excluded=18)
    srcs = re.findall(r'<audio[^>]*src="([^"]+)"', html)
    expected = [f"vibevoice-{v['id']}-{key}.wav" for v in VOICES for key, _ in vm.CELLS]
    assert srcs == expected, f"{srcs} != {expected}"
    print(f"ok   {len(srcs)} cells point at the names the generator writes")

    for sentence in (vm.SHORT, vm.LONG):
        assert html_escape(sentence) in html, "the sentences must be printed in full"
    for v in VOICES:
        assert f"<code>{v['id']}</code>" in html, f"{v['id']} not printed verbatim"
    assert "18 preset voices in other languages" in html
    print("ok   both sentences, every preset id, and the exclusion count are on the page")

    print("\nall good")


if __name__ == "__main__":
    main()
