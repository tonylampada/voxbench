#!/usr/bin/env python3
"""A client that hangs up mid-stream must free gen_lock right away.

No GPU, no model: `voxcpm` is stubbed and the ASGI app is driven directly, so
http.disconnect can be delivered at the exact moment the first chunk lands.

    pip install fastapi numpy && python engines/voxcpm2/test_stop.py

This is the cheap regression guard, NOT the proof: it hands Starlette a
synthetic disconnect and a fake model that yields every 50 ms. The proof that
uvicorn delivers a real TCP hangup and that the real model yields often enough
is `bench/abort-latency.py <engine>`, against a live container.
"""
import asyncio
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.modules["voxcpm"] = types.SimpleNamespace(VoxCPM=None)  # never loaded: no startup, no GPU
sys.modules.setdefault(  # server.py seeds torch's RNG; there is no model here to seed
    "torch", types.SimpleNamespace(manual_seed=lambda seed: None))
sys.path.insert(0, str(Path(__file__).parent))
import server  # noqa: E402

CHUNK = 0.05  # seconds per fake chunk


class FakeWav:
    """Stands in for the model's audio tensor, which the server squeezes."""

    def squeeze(self, _dim):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.zeros(480, dtype=np.float32)


class FakeModel:
    """One chunk per CHUNK seconds; remembers whether it is still decoding."""

    def __init__(self):
        self.busy = False
        self.yielded = 0  # chunks the GPU actually paid for, not chunks a client read
        self.tts_model = self

    def build_prompt_cache(self, **kw):
        return dict(kw)

    def generate_with_prompt_cache_streaming(self, target_text, prompt_cache=None, **kw):
        self.busy = True
        try:
            for _ in range(int(len(target_text) / 4)):
                time.sleep(CHUNK)
                self.yielded += 1
                yield FakeWav(), None, None
        finally:
            self.busy = False


async def speak(text, hang_up_after=None):
    """-> (chunks received, seconds to the first one). Disconnects after
    `hang_up_after` chunks, the way the board's AbortController does."""
    body = json.dumps({"input": text, "stream": True}).encode()
    started, first, got = time.monotonic(), None, []
    hung_up = asyncio.Event()

    async def receive():
        if body_sent.done():
            await hung_up.wait()
            return {"type": "http.disconnect"}
        body_sent.set_result(True)
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg):
        nonlocal first
        if msg["type"] == "http.response.body" and msg.get("body"):
            got.append(msg["body"])
            first = first or time.monotonic() - started
            if hang_up_after and len(got) >= hang_up_after:
                hung_up.set()

    body_sent = asyncio.get_running_loop().create_future()
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "path": "/v1/audio/speech", "raw_path": b"/v1/audio/speech",
             "query_string": b"", "root_path": "", "scheme": "http", "client": ("1.2.3.4", 5),
             "server": ("test", 80),
             "headers": [(b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())]}
    await server.app(scope, receive, send)
    return got, first


async def main():
    model = server.model = FakeModel()

    idle, baseline = await speak("short")
    assert idle, "the fake engine produced nothing"

    long_one = asyncio.create_task(speak("x" * 400, hang_up_after=1))  # 100 chunks' worth
    await asyncio.sleep(CHUNK * 3)  # let it get going, then pull the plug
    await long_one
    await asyncio.sleep(CHUNK * 2)  # a chunk's grace to notice the hangup

    # Not "it got faster": the generation itself is over, and the lock with it.
    assert not model.busy, f"abandoned generation still decoding ({model.yielded} chunks in)"
    stopped_at = model.yielded
    assert stopped_at < 100, "the hangup was ignored — it ran the whole utterance"

    _, after_abort = await speak("short")
    assert after_abort < baseline + 3 * CHUNK, \
        f"second request waited {after_abort:.2f}s on the lock (idle: {baseline:.2f}s)"
    print(f"ok — first chunk {after_abort:.2f}s after an abort, {baseline:.2f}s idle; "
          f"abandoned generation stopped after {stopped_at - 1}/100 chunks")


if __name__ == "__main__":
    asyncio.run(main())
