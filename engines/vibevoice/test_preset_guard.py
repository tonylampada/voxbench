#!/usr/bin/env python3
"""A voice preset that names an unexpected pickle global must never be loaded.

The presets are .pt pickles, and torch cannot load these ones with
weights_only=True (see SAFE_GLOBALS in server.py), so the opcode scan is the
only thing between a swapped preset file and arbitrary code execution at
startup. This checks that the scan sees what is really in the file and that
load_preset refuses on anything unexpected.

Runs inside the engine image, no GPU needed:

    docker run --rm -v "$PWD/engines/vibevoice:/t" tts-vibevoice:gpu python /t/test_preset_guard.py
"""
import pickle
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "/app")
import server  # noqa: E402


class Evil:
    def __reduce__(self):
        return (__import__("os").system, ("echo pwned",))


def main():
    tmp = Path(tempfile.mkdtemp())

    benign = tmp / "benign.pt"
    torch.save({"a": torch.zeros(2), "b": {"c": torch.ones(3)}}, benign)
    globals_seen = server.preset_globals(benign)
    assert globals_seen <= server.SAFE_GLOBALS, f"benign save named {globals_seen - server.SAFE_GLOBALS}"
    assert torch.equal(server.load_preset(benign, "cpu")["a"], torch.zeros(2))
    print(f"ok   a plain tensor save names only {sorted(globals_seen)} and loads")

    evil = tmp / "evil.pt"
    torch.save({"a": Evil()}, evil)
    assert "posix system" in server.preset_globals(evil) or \
           "nt system" in server.preset_globals(evil), server.preset_globals(evil)
    try:
        server.load_preset(evil, "cpu")
    except RuntimeError as e:
        assert "unexpected pickle globals" in str(e), e
        print(f"ok   os.system in a preset is refused: {e}")
    else:
        sys.exit("FAIL: load_preset loaded a preset that calls os.system")

    # The real thing, if this is running in the built image: every shipped
    # preset must pass the scan, or the engine cannot start at all.
    shipped = sorted(server.PRESET_DIR.glob("*.pt"))
    if shipped:
        bad = {p.name: sorted(server.preset_globals(p) - server.SAFE_GLOBALS) for p in shipped}
        bad = {k: v for k, v in bad.items() if v}
        assert not bad, f"shipped presets name unexpected globals: {bad}"
        print(f"ok   all {len(shipped)} shipped presets name only allowed globals")
    else:
        print("skip shipped presets — not running in the engine image")
    print("\nall good")


if __name__ == "__main__":
    main()
