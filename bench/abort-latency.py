#!/usr/bin/env python3
"""Does aborting a stream free the engine? Usage: bench/abort-latency.py <engine> [base_url]

Brings nothing up — assumes the engine is already healthy (engines/tts up <e>).
Times the first audio byte of a short streaming request on an idle engine, then
starts a long one, drops the connection after one byte (what the board's
AbortController does on every new message) and times the same short request
again. Two similar numbers means the engine heard the hangup; ten seconds the
second time means it is still rendering audio nobody will hear, holding the GPU
lock. Exit non-zero on the second case. Stdlib only.
"""
import json
import re
import sys
import time
import urllib.parse
from http.client import HTTPConnection
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHORT = "Captain, the deploy is green."
# Long enough that finishing it costs many seconds — that cost is the bug.
LONG = "The quick brown fox jumps over the lazy dog. " * 20


def port_for(engine: str) -> int:
    registry = (REPO / "engines" / "tts").read_text()
    m = re.search(rf"^{re.escape(engine)}\s+(\d+)\s", registry, re.M)
    if not m:
        sys.exit(f"abort-latency: unknown engine '{engine}' (not in engines/tts registry)")
    return int(m.group(1))


def ttfb(base: str, text: str, abort: bool = False) -> float:
    """-> seconds to the first audio byte. abort=True hangs up right there,
    mid-utterance; otherwise the response is drained, leaving the engine idle."""
    u = urllib.parse.urlsplit(base)
    conn = HTTPConnection(u.hostname, u.port or 80, timeout=300)
    t0 = time.time()
    conn.request("POST", "/v1/audio/speech", json.dumps({"input": text, "stream": True}),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        sys.exit(f"abort-latency: engine answered {r.status}: {r.read()[:200]!r}")
    r.read(1)
    dt = time.time() - t0
    if not abort:
        r.read()
    conn.close()
    return dt


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.splitlines()[0])
    engine = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else f"http://127.0.0.1:{port_for(engine)}"

    ttfb(base, SHORT)  # warm-up, discarded: the first call after load pays CUDA warmup
    idle = ttfb(base, SHORT)
    aborted = ttfb(base, LONG, abort=True)
    after = ttfb(base, SHORT)

    limit = max(idle * 3, idle + 1.0)
    print(f"idle           {idle:6.2f}s to first audio byte")
    print(f"long, aborted  {aborted:6.2f}s to first audio byte, then hung up")
    print(f"after abort    {after:6.2f}s to first audio byte   (limit {limit:.2f}s)")
    if after > limit:
        sys.exit(f"FAIL — the abort did not free the engine: {after:.2f}s against {idle:.2f}s idle. "
                 f"It is still rendering the aborted utterance, holding gen_lock.")
    print(f"ok — the abort freed the engine ({after:.2f}s against {idle:.2f}s idle)")


main()
