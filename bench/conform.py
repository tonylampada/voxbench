#!/usr/bin/env python3
"""Conformance harness for the contract in docs/contract.md.

Usage: bench/conform.py <engine> [base_url]

Brings nothing up — assumes the engine is already healthy (engines/tts up <e>).
Checks every clause of the spec against the live server: one line per check,
exit non-zero on any violation. Stdlib only.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FAILURES = 0


def check(name: str, ok: bool, detail: str = ""):
    global FAILURES
    if not ok:
        FAILURES += 1
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


def port_for(engine: str) -> int:
    registry = (REPO / "engines" / "tts").read_text()
    m = re.search(rf"^{re.escape(engine)}\s+(\d+)\s", registry, re.M)
    if not m:
        sys.exit(f"conform: unknown engine '{engine}' (not in engines/tts registry)")
    return int(m.group(1))


def http(method: str, path: str, body: dict | None = None, timeout: float = 300):
    """-> (status, headers, bytes). Never raises on HTTP errors."""
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


def is_wav(b: bytes) -> bool:
    return b[:4] == b"RIFF" and b[8:12] == b"WAVE"


def expect_400(name: str, body: dict):
    st, _, raw = http("POST", "/v1/audio/speech", body)
    detail = ""
    try:
        detail = json.loads(raw)["detail"]
    except Exception:
        pass
    check(name, st == 400 and bool(detail),
          f"got {st}" + (f': "{str(detail)[:90]}"' if detail else " (no detail in body)"))


def main():
    global BASE
    engine = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    BASE = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else f"http://127.0.0.1:{port_for(engine)}"
    print(f"conform: {engine} @ {BASE}")

    # ── liveness ─────────────────────────────────────────────────────────
    st, _, raw = http("GET", "/health", timeout=5)
    h = json.loads(raw) if st == 200 else {}
    check("health", st == 200 and h.get("status") == "ok" and h.get("engine"),
          f"{st} {raw[:80]!r}")
    if st != 200:
        sys.exit("engine not healthy — conform assumes it is already up")
    st, hdr, raw = http("GET", "/", timeout=5)
    check("test page", st == 200 and "text/html" in hdr.get("content-type", ""), f"{st}")

    # ── old surface is gone ──────────────────────────────────────────────
    for m, p in [("POST", "/tts"), ("POST", "/tts/stream"), ("GET", "/voices")]:
        st, _, _ = http(m, p, {"text": "x"} if m == "POST" else None, timeout=10)
        check(f"gone: {m} {p}", st == 404, f"got {st}")

    # ── /v1/voices ───────────────────────────────────────────────────────
    st, _, raw = http("GET", "/v1/voices", timeout=10)
    voices = json.loads(raw).get("voices", []) if st == 200 else []
    check("/v1/voices answers", st == 200 and isinstance(voices, list) and voices, f"{st}")
    ids = [v.get("id") for v in voices]
    check("voice ids unique", len(set(ids)) == len(ids))
    check("voice ids opaque", all(i and "/" not in i and "\\" not in i and not i.endswith((".wav", ".flac", ".mp3")) for i in ids))
    bad = [v["id"] for v in voices if not (
        v.get("name") and v.get("kind") in ("builtin", "reference")
        and isinstance(v.get("langs"), list) and v["langs"]
        and v.get("gender") in ("female", "male", "unknown")
        and (v["kind"] == "builtin" or (v.get("origin") and isinstance(v.get("seconds"), (int, float))))
    )]
    check("voice entries well-formed", not bad, f"bad: {bad[:5]}")
    builtins = [v for v in voices if v.get("kind") == "builtin"]
    refs = [v for v in voices if v.get("kind") == "reference"]
    manifest = json.loads((REPO / "voices" / "library" / "manifest.json").read_text())
    want_refs = {e["file"][:-4] for e in manifest if e.get("file")}
    check("library fully listed", {v["id"] for v in refs} >= want_refs,
          f"{len(refs)} reference voices listed, manifest has {len(want_refs)}; "
          f"missing e.g. {sorted(want_refs - {v['id'] for v in refs})[:3]}")
    print(f"     ({len(builtins)} builtin + {len(refs)} reference)")

    # ── /v1/params ───────────────────────────────────────────────────────
    st, _, raw = http("GET", "/v1/params", timeout=10)
    params = json.loads(raw).get("params") if st == 200 else None
    ok = st == 200 and isinstance(params, dict) and all(
        p.get("type") in ("number", "integer", "boolean", "string")
        and "default" in p and p.get("description")
        and (p["type"] not in ("number", "integer") or ("min" in p and "max" in p))
        for p in params.values())
    check("/v1/params well-formed", ok, f"{st} {raw[:100]!r}")
    params = params or {}

    # ── synthesis ────────────────────────────────────────────────────────
    voice = (builtins or voices)[0] if voices else None
    lang = voice["langs"][0] if voice else "en"
    if voice:
        st, hdr, raw = http("POST", "/v1/audio/speech",
                            {"input": "Hello from the conformance harness.", "voice": voice["id"], "lang": lang})
        check(f"speech builtin ({voice['id']})", st == 200 and is_wav(raw)
              and "audio/wav" in hdr.get("content-type", ""), f"{st}, {len(raw)}b")
    st, _, raw = http("POST", "/v1/audio/speech", {"input": "Defaults work."})
    check("speech with defaults only", st == 200 and is_wav(raw), f"{st}")

    ref = next((v for v in refs if v.get("origin") == engine), refs[0] if refs else None)
    if ref:
        st, _, raw = http("POST", "/v1/audio/speech",
                          {"input": "Cloned greeting.", "voice": ref["id"], "lang": ref["langs"][0]})
        check(f"speech reference ({ref['id']})", st == 200 and is_wav(raw), f"{st}, {len(raw)}b")

    if params:
        knob_vals = {k: p["default"] for k, p in params.items()}
        st, _, raw = http("POST", "/v1/audio/speech",
                          {"input": "Knobs at defaults.", "params": knob_vals})
        check(f"speech with params {sorted(knob_vals)}", st == 200 and is_wav(raw), f"{st}")
    if "speed" in params:
        st, _, raw = http("POST", "/v1/audio/speech", {"input": "Faster now.", "speed": 1.5})
        check("top-level speed alias", st == 200 and is_wav(raw), f"{st}")

    # ── streaming ────────────────────────────────────────────────────────
    long_text = "The quick brown fox jumps over the lazy dog. " * 8
    req = urllib.request.Request(BASE + "/v1/audio/speech",
                                 data=json.dumps({"input": long_text, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            first = r.read(4096)
            ttfb = time.time() - t0
            rest = r.read()
            total = time.time() - t0
            rate = r.headers.get("x-sample-rate", "")
            n = len(first) + len(rest)
            check("stream: raw PCM + x-sample-rate", r.status == 200 and rate.isdigit()
                  and n > 0 and n % 2 == 0 and not is_wav(first),
                  f"{n}b @ {rate}Hz, TTFB {ttfb:.2f}s, total {total:.2f}s")
            check("stream: progressive", ttfb < total * 0.5 or total < 1.0,
                  f"first bytes at {ttfb:.2f}s of {total:.2f}s")
    except urllib.error.HTTPError as e:
        # An engine that cannot stream must say so with a 400 + sentence
        # (contract, "Streaming response") — that is conformant, not a failure.
        body = e.read()
        detail = ""
        try:
            detail = json.loads(body)["detail"]
        except Exception:
            pass
        if e.code == 400 and "stream" in str(detail).lower():
            check("stream: honest cannot-stream 400", True, f'"{str(detail)[:90]}"')
        else:
            check("stream", False, f"got {e.code}: {body[:100]!r}")

    # ── the 400 catalogue ────────────────────────────────────────────────
    expect_400("400: empty input", {"input": "   "})
    expect_400("400: missing input", {"voice": ids[0] if ids else "x"})
    expect_400("400: unknown voice", {"input": "hi", "voice": "no-such-voice-xyz"})
    expect_400("400: unsupported lang", {"input": "hi", "lang": "xx"})
    expect_400("400: unknown params key", {"input": "hi", "params": {"definitely_not_a_knob": 1}})
    expect_400("400: bad response_format", {"input": "hi", "response_format": "xyz"})
    expect_400("400: wav + stream", {"input": "hi", "response_format": "wav", "stream": True})
    if "speed" in params:
        expect_400("400: params.speed out of range",
                   {"input": "hi", "params": {"speed": params["speed"]["max"] * 10}})
        expect_400("400: speed in both places", {"input": "hi", "speed": 1.5, "params": {"speed": 1.5}})
    mismatch = next(((v, o["langs"][0]) for v in voices for o in voices
                     if o["langs"][0] not in v["langs"]), None)
    if mismatch:
        expect_400(f"400: voice/lang mismatch ({mismatch[0]['id']}+{mismatch[1]})",
                   {"input": "hi", "voice": mismatch[0]["id"], "lang": mismatch[1]})
    foreign = next((v for v in refs if v.get("origin") != engine), None)
    if foreign:
        # cloning engines synthesize any clip (200); non-cloning must refuse loudly (400)
        st, _, raw = http("POST", "/v1/audio/speech",
                          {"input": "Foreign clip.", "voice": foreign["id"], "lang": foreign["langs"][0]})
        detail = ""
        try:
            detail = json.loads(raw)["detail"]
        except Exception:
            pass
        check(f"foreign reference ({foreign['id']}): clone or loud 400",
              (st == 200 and is_wav(raw)) or (st == 400 and bool(detail)),
              f"got {st}" + (f': "{str(detail)[:80]}"' if detail else ""))

    print(f"\n{'CONFORMANT' if not FAILURES else f'{FAILURES} VIOLATION(S)'} — {engine}")
    sys.exit(1 if FAILURES else 0)


main()
