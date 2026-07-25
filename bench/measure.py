#!/usr/bin/env python3
"""Measure one engine on contract v2. Usage: bench/measure.py <engine> [--cpu]

Brings the engine up through engines/tts (waiting politely if another GPU
engine is resident — never --force), measures, brings it down. Per language
(en, pt) and text length (short, long): generation wall time, audio duration,
realtime ratio. Streaming: time to first byte on stream:true, or the engine's
honest cannot-stream 400 recorded as "no_streaming". VRAM while resident (read
through the ollama container — host nvidia-smi is broken by the driver
mismatch) and container RAM idle → peak.

The texts are the bake-off originals, verbatim — change them and every number
stops being comparable with the artifact's history.

Warm-up policy: one discarded short synthesis per language right after load,
the same policy the v1 bake-off used. The first call after load pays CUDA
graph warmup and lazy kernel compilation (Higgs measured 1.0x on its first
request and 3.2x thirty seconds later); the discard keeps that cost off
whichever measurement happens to run first.

Resumable: results land in bench/results/<engine>.json, saved after every
measurement; a leaf already present is skipped, so an interrupted run picks up
where it stopped. --cpu (chatterbox-nano only) measures the CPU path into
bench/results/chatterbox-nano-cpu.json. Stdlib only.
"""
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "bench" / "results"

# The bake-off sentences: one short (the notification case) and one long
# (a real board message) per language. Same text for every engine.
SENT = {
    "en": {
        "short": "Captain, the deploy is green. Two pull requests merged, one waiting on your review.",
        "long": ("Captain, here is the state of the fleet. The audio artifact work is merged and live, so a "
                 "voice sample now plays inside the card viewer instead of only offering a download. Nine text "
                 "to speech engines are being wrapped in containers tonight, each behind the same interface, so "
                 "the comparison is honest. Nothing is blocked, and nothing is on fire."),
    },
    "pt": {
        "short": "Capitão, o deploy passou. Dois pull requests mergeados, um esperando a sua revisão.",
        "long": ("Capitão, segue o estado da frota. O trabalho de áudio nos artefatos está mergeado e no ar, "
                 "então uma amostra de voz agora toca dentro do visualizador do card em vez de só oferecer "
                 "download. Nove motores de síntese de voz estão sendo empacotados em containers esta noite, "
                 "cada um atrás da mesma interface, para a comparação ser honesta. Nada travado, nada pegando fogo."),
    },
}


def port_for(engine: str) -> int:
    m = re.search(rf"^{re.escape(engine)}\s+(\d+)\s", (REPO / "engines" / "tts").read_text(), re.M)
    if not m:
        sys.exit(f"measure: unknown engine '{engine}' (not in engines/tts registry)")
    return int(m.group(1))


def sh(*args, timeout=1800):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def vram_mib():
    r = sh("docker", "exec", "ollama", "nvidia-smi",
           "--query-gpu=memory.used", "--format=csv,noheader,nounits", timeout=30)
    return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else None


def ram_mib(container):
    """Container memory in MiB, computed the way docker stats does
    (memory.current minus inactive page cache — engines that read weights
    through the page cache would otherwise look gigabytes fatter than the v1
    column this feeds). Sampled after load (idle) and after synthesis (peak),
    the v1 bake-off's semantics; the cgroup high-watermark would also count
    load transients."""
    cur = sh("docker", "exec", container, "cat", "/sys/fs/cgroup/memory.current", timeout=20)
    stat = sh("docker", "exec", container, "cat", "/sys/fs/cgroup/memory.stat", timeout=20)
    if cur.returncode or stat.returncode:
        return None
    inactive = int(re.search(r"^inactive_file (\d+)$", stat.stdout, re.M).group(1))
    return round((int(cur.stdout.strip()) - inactive) / 2**20)


def wav_seconds(data):
    # Trust the byte count, not the header: a server that streams WAV writes a
    # placeholder frame count (Kokoro-FastAPI declares 0xFFFFFFFF).
    with wave.open(io.BytesIO(data)) as w:
        return max(0, len(data) - 44) / float(w.getframerate() * w.getnchannels() * w.getsampwidth())


def speech(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/audio/speech",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def detail_of(e: urllib.error.HTTPError) -> str:
    body = e.read()
    try:
        return json.loads(body)["detail"]
    except Exception:
        return body.decode(errors="replace")[:200]


def measure_gen(port, lang, kind):
    t0 = time.time()
    try:
        with speech(port, {"input": SENT[lang][kind], "lang": lang}) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return {"error": f"{e.code}: {detail_of(e)}"}
    gen = time.time() - t0
    audio = wav_seconds(data)
    return {"gen_s": round(gen, 2), "audio_s": round(audio, 2),
            "realtime_x": round(audio / gen, 2), "bytes": len(data)}


def measure_stream(port, lang):
    t0 = time.time()
    try:
        with speech(port, {"input": SENT[lang]["long"], "lang": lang, "stream": True}) as r:
            first = r.read(4096)
            ttfb = time.time() - t0
            rest = r.read()
            total = time.time() - t0
            rate = int(r.headers.get("x-sample-rate", 0))
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return {"no_streaming": detail_of(e)}  # the honest answer, not a failure
        return {"error": f"{e.code}: {detail_of(e)}"}
    n = len(first) + len(rest)
    return {"ttfb_s": round(ttfb, 2), "total_s": round(total, 2),
            "audio_s": round(n / (rate * 2), 2) if rate else None, "bytes": n}


def up(engine, cpu):
    """engines/tts up, waiting for a resident sibling instead of --force."""
    cmd = [str(REPO / "engines" / "tts"), "up", engine] + (["--cpu"] if cpu else [])
    while True:
        r = sh(*cmd, timeout=1800)
        if r.returncode == 0:
            return
        err = (r.stderr + r.stdout).strip()
        if "already running" in err:
            print(f"  GPU busy ({err.splitlines()[-1][:90]}) — waiting 60s", flush=True)
            time.sleep(60)
            continue
        sys.exit(f"measure: up {engine} failed:\n{err}")


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    cpu = "--cpu" in sys.argv[2:]
    if cpu and engine != "chatterbox-nano":
        sys.exit("measure: --cpu only makes sense for chatterbox-nano")
    port, container = port_for(engine), f"tts-{engine}"
    name = f"{engine}-cpu" if cpu else engine
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    row = json.loads(path.read_text()) if path.exists() else {
        "engine": name, "port": port,
        "method": {"texts": "bake-off v1 sentences, verbatim",
                   "warmup": "one discarded short synth per language after load"},
        "langs": {},
    }
    save = lambda: path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")

    leaves = [(lang, k) for lang in SENT for k in ("short", "long", "stream")
              if k not in row["langs"].get(lang, {})]
    hw = [k for k in ("vram_idle_mib", "vram_busy_mib", "ram_idle_mib", "ram_peak_mib")
          if k not in row] if not cpu else \
         [k for k in ("ram_idle_mib", "ram_peak_mib") if k not in row]
    if not leaves and not hw:
        print(f"{name}: complete — nothing to do (delete {path} to re-measure)")
        return

    print(f"{name}: measuring {len(leaves)} leaves -> {path}", flush=True)
    up(engine, cpu)
    if "ram_idle_mib" in hw:
        row["ram_idle_mib"] = ram_mib(container)
    if "vram_idle_mib" in hw:
        row["vram_idle_mib"] = vram_mib()
    save()

    warmed = set()
    for lang, kind in leaves:
        if lang not in warmed:
            try:
                with speech(port, {"input": SENT[lang]["short"], "lang": lang}) as r:
                    r.read()
            except urllib.error.HTTPError:
                pass  # a lang the engine can't speak — the measurement records the 400
            warmed.add(lang)
        res = measure_stream(port, lang) if kind == "stream" else measure_gen(port, lang, kind)
        row["langs"].setdefault(lang, {})[kind] = res
        save()
        print(f"  {lang}/{kind}: {json.dumps(res, ensure_ascii=False)[:120]}", flush=True)

    if "vram_busy_mib" in hw:
        row["vram_busy_mib"] = vram_mib()
    if "ram_peak_mib" in hw:
        row["ram_peak_mib"] = ram_mib(container)
    save()
    sh(str(REPO / "engines" / "tts"), "down", engine)
    print(f"{name}: done", flush=True)


if __name__ == "__main__":
    main()
