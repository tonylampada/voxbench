#!/usr/bin/env python3
# console — control plane for engines/. Runs on the host, outside docker, and
# drives containers by shelling out to engines/tts (the registry of record —
# nothing about names/ports/gpu is restated here). Loopback only: the engines
# have no auth and neither does this; the network boundary is the auth boundary.
import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS = os.path.join(ROOT, "engines", "tts")
PORT = int(os.environ.get("CONSOLE_PORT", "9000"))  # well clear of the engine block 8881-8896

# Host path of the driver libs that match the kernel module — engines/tts
# mounts this into every GPU container (see gpu_run_args there). Pulled out of
# the script so the workaround lives in exactly one place.
_m = re.search(r"-v (\S+):/opt/nvdrv", open(TTS).read())
NVDRV = _m.group(1) if _m else None

lock = threading.Lock()
ops = {}  # name -> {"proc": Popen, "log": [lines], "t0": epoch} while `tts up` runs
last_error = {}  # name -> last failed `tts up` message, cleared on next up/down

# ── Phrase library ───────────────────────────────────────────────────────────
# The test screen's phrases. Seeds are the bench texts (short/medium/long, en
# and pt); user additions/deletions persist in a JSON file next to this server.
# No file until the first change — the seeds live here and always come back.
PHRASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phrases.json")
# short/long verbatim from bench/measure.py's bake-off sentences, medium from
# bench/mkvoicelib.py's reference-clip seed — one text means one thing across
# the measurements, the listening pages and this screen. The long texts are
# multi-sentence on purpose: they expose context-wall re-reads and dropped
# final clauses that a repeated sentence would hide.
SEED_PHRASES = [
    {"label": "short", "lang": "en", "text": "Captain, the deploy is green. Two pull requests merged, one waiting on your review."},
    {"label": "short", "lang": "pt", "text": "Capitão, o deploy passou. Dois pull requests mergeados, um esperando a sua revisão."},
    {"label": "medium", "lang": "en", "text": "The harbour was quiet that morning, and the light came in sideways across the water. Nothing much happened, which was exactly what everyone had hoped for."},
    {"label": "medium", "lang": "pt", "text": "O porto estava quieto naquela manhã, e a luz entrava de lado sobre a água. Não aconteceu grande coisa, que era exatamente o que todo mundo esperava."},
    {"label": "long", "lang": "en", "text": "Captain, here is the state of the fleet. The audio artifact work is merged and live, so a "
                                   "voice sample now plays inside the card viewer instead of only offering a download. Nine text "
                                   "to speech engines are being wrapped in containers tonight, each behind the same interface, so "
                                   "the comparison is honest. Nothing is blocked, and nothing is on fire."},
    {"label": "long", "lang": "pt", "text": "Capitão, segue o estado da frota. O trabalho de áudio nos artefatos está mergeado e no ar, "
                                   "então uma amostra de voz agora toca dentro do visualizador do card em vez de só oferecer "
                                   "download. Nove motores de síntese de voz estão sendo empacotados em containers esta noite, "
                                   "cada um atrás da mesma interface, para a comparação ser honesta. Nada travado, nada pegando fogo."},
]


def load_phrases():
    if os.path.exists(PHRASES_FILE):
        return json.load(open(PHRASES_FILE))
    return [{"id": i + 1, **p} for i, p in enumerate(SEED_PHRASES)]


def save_phrases(phrases):
    tmp = PHRASES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PHRASES_FILE)


def run(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, timeout=120, **kw)


def registry():
    """name -> {port, gpu, container, health} parsed from `engines/tts ls`."""
    out = run([TTS, "ls"]).stdout
    reg = {}
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) == 6:
            reg[f[0]] = {"port": int(f[1]), "gpu": f[2] == "gpu",
                         "container": f[4], "health": f[5]}
    return reg


def uptime_s(name):
    r = run(["docker", "inspect", "-f", "{{.State.StartedAt}}", f"tts-{name}"])
    if r.returncode != 0:
        return None
    ts = re.sub(r"\.(\d{6})\d*", r".\1", r.stdout.strip()).replace("Z", "+00:00")
    return int(time.time() - datetime.fromisoformat(ts).timestamp())


def smi(query):
    """nvidia-smi with the env fallbacks; (stdout, None) or (None, reason).
    Plain invocation fails until the reboot aligns kernel module and host libs
    (NVML mismatch), so retry with the driver-matched libs from engines/tts."""
    argv = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    envs = [None] + ([{**os.environ, "LD_LIBRARY_PATH": NVDRV}] if NVDRV else [])
    reasons = []
    for env in envs:
        try:
            r = run(argv, env=env)
        except OSError as e:
            reasons.append(str(e))
            continue
        if r.returncode == 0:
            return r.stdout, None
        reasons.append(r.stderr.strip().splitlines()[-1] if r.stderr else "failed")
    return None, "nvidia-smi failed on host: " + "; ".join(reasons)


def read_vram():
    out, why = smi("--query-gpu=memory.used,memory.total")
    if out is None:
        return None, why
    used, total = (int(x) for x in out.split(","))
    return {"used_mb": used, "total_mb": total}, None


# Desktop/idle baseline. Above this with no registered engine resident, someone
# outside the registry holds the card (a container from another branch, a stray
# process) — report that, never "free".
IDLE_VRAM_MB = 1024


def gpu_other(vram):
    """What holds the card when no registered engine does, or None if idle."""
    if not vram or vram["used_mb"] <= IDLE_VRAM_MB:
        return None
    procs = []
    out, _ = smi("--query-compute-apps=pid,process_name,used_gpu_memory")
    for line in (out or "").splitlines():
        if line.strip():
            pid, rest = line.split(",", 1)
            name, used = rest.rsplit(",", 1)
            procs.append({"pid": int(pid), "name": name.strip(),
                          "used_mb": int(used)})
    return {"used_mb": vram["used_mb"], "processes": procs}


def gpu_holder(reg):
    """The engine currently resident on the card, or None. cpu-mode nano
    (NANO_DEVICE=cpu) doesn't hold the card — same rule as engines/tts."""
    for name, e in reg.items():
        if e["gpu"] and e["container"] == "running":
            r = run(["docker", "inspect", "-f", "{{.Config.Env}}", f"tts-{name}"])
            if "NANO_DEVICE=cpu" not in r.stdout:
                return name
    return None


def engines_json():
    reg = registry()
    vram, _ = read_vram()
    holder = gpu_holder(reg)
    out = []
    for name, e in reg.items():
        with lock:
            op = ops.get(name)
            # "dead": the engine itself says it can no longer synthesize (poisoned
            # CUDA context). Never paint that as loading — it will never finish.
            state = {"ok": "ready", "dead": "dead"}.get(e["health"], "loading") \
                if e["container"] == "running" else ("loading" if op else "stopped")
            item = {
                "name": name, "port": e["port"],
                "gpu": "gpu" if e["gpu"] else "cpu", "state": state,
                "healthy": e["health"] == "ok",
                # one card, one model: the resident engine owns whatever is used
                "vram_mb": vram["used_mb"] if vram and name == holder else None,
                "uptime_s": uptime_s(name) if e["container"] == "running" else None,
                "error": last_error.get(name),
                "detail": op["log"][-1].strip() if op and op["log"] else None,
            }
        out.append(item)
    return out


def start_engine(name):
    with lock:
        if name in ops:
            return 202, {"state": "loading"}
        last_error.pop(name, None)
        p = subprocess.Popen([TTS, "up", name], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        op = ops[name] = {"proc": p, "log": [], "t0": time.time()}

    def pump():
        for line in p.stdout:
            op["log"].append(line)
        p.wait()
        with lock:
            ops.pop(name, None)
            if p.returncode != 0:
                msg = [l for l in op["log"] if l.strip()]
                last_error[name] = (msg[-1].strip() if msg else "tts up failed") \
                    .removeprefix("tts: ")
    threading.Thread(target=pump, daemon=True).start()

    # The GPU-conflict refusal in `tts up` is instant — give it a moment so the
    # browser gets the refusal (naming the resident engine) as a real answer
    # instead of a phantom "loading".
    for _ in range(20):
        if p.poll() is not None:
            break
        time.sleep(0.1)
    if p.poll() is not None and p.returncode != 0:
        time.sleep(0.2)  # let pump() record the message
        with lock:
            return 409, {"error": last_error.get(name, "tts up failed")}
    return 202, {"state": "loading"}


def stop_engine(name):
    with lock:
        last_error.pop(name, None)
    r = run([TTS, "down", name])
    if r.returncode != 0:
        return 500, {"error": (r.stderr or r.stdout).strip()}
    return 200, {"state": "stopped"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def engine_or_404(self, name):
        reg = registry()
        if name not in reg:
            self.send_json(404, {"error": f"unknown engine '{name}'"})
            return None
        return reg[name]

    def proxy(self, name, path):
        """Forward to the engine on loopback, pass bodies through unchanged,
        stream the response (speech can be chunked PCM)."""
        e = self.engine_or_404(name)
        if not e:
            return
        if e["container"] != "running":
            return self.send_json(502, {"error": f"{name} is not running — start it first"})
        import http.client
        body = None
        if self.command == "POST":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        conn = http.client.HTTPConnection("127.0.0.1", e["port"], timeout=600)
        try:
            conn.request(self.command, path, body=body,
                         headers={"Content-Type": self.headers.get("Content-Type",
                                                                   "application/json")})
            r = conn.getresponse()
        except OSError as ex:
            return self.send_json(502, {"error": f"{name} did not answer: {ex}"})
        self.send_response(r.status)
        for h in ("Content-Type", "x-sample-rate"):
            if r.getheader(h):
                self.send_header(h, r.getheader(h))
        # stream with no Content-Length: delimit by close (works mid-synthesis)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        # The page pressing stop aborts its fetch, which breaks this pipe. That is
        # not an error, it is the message: let go of the engine's connection and
        # the engine stops synthesizing. Closing here rather than leaving it to the
        # garbage collector is the difference between "the GPU is freed" and "the
        # GPU is freed, probably" — and an abandoned synthesis is what poisons the
        # next one's CUDA context.
        try:
            while chunk := r.read(65536):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()

    def send_html(self, filename, ctype="text/html; charset=utf-8"):
        body = open(os.path.join(os.path.dirname(__file__), filename), "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # A query string belongs to the page, not to this server: no route here
        # has ever wanted one.
        self.path = self.path.split("?", 1)[0]
        if self.path == "/" or self.path == "/index.html":
            self.send_html("index.html")
        elif self.path == "/test":
            self.send_html("test.html")
        elif self.path == "/speech.js":
            self.send_html("speech.js", "text/javascript; charset=utf-8")
        elif self.path == "/api/phrases":
            with lock:
                self.send_json(200, {"phrases": load_phrases()})
        elif self.path == "/api/engines":
            self.send_json(200, engines_json())
        elif self.path == "/api/machine":
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":")
                mem[k] = int(v.split()[0]) // 1024
            vram, why = read_vram()
            holder = gpu_holder(registry())
            self.send_json(200, {
                "ram": {"used_mb": mem["MemTotal"] - mem["MemAvailable"],
                        "total_mb": mem["MemTotal"]},
                "vram": vram if vram else {"unavailable": why},
                "gpu_engine": holder,
                # the card can be taken by something the registry doesn't know
                # (a container from another branch, a stray process) — say so
                "gpu_other": None if holder else gpu_other(vram),
            })
        elif m := re.fullmatch(r"/api/engines/([\w.-]+)/(voices|params)", self.path):
            self.proxy(m.group(1), f"/v1/{m.group(2)}")
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/phrases":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self.send_json(400, {"error": "body must be JSON"})
            text = (body.get("text") or "").strip()
            if not text:
                return self.send_json(400, {"error": "text is required"})
            # a phrase is filed under a name and a language — that pair is what
            # the test screen's dropdown shows ("greeting · pt")
            label = (body.get("name") or body.get("label") or "").strip() or "custom"
            lang = (body.get("lang") or "").strip().lower()
            with lock:
                phrases = load_phrases()
                p = {"id": max((q["id"] for q in phrases), default=0) + 1,
                     "label": label, "lang": lang, "text": text}
                phrases.append(p)
                save_phrases(phrases)
            self.send_json(200, p)
        elif m := re.fullmatch(r"/api/engines/([\w.-]+)/(up|down)", self.path):
            name, verb = m.groups()
            if not self.engine_or_404(name):
                return
            self.send_json(*(start_engine if verb == "up" else stop_engine)(name))
        elif m := re.fullmatch(r"/api/engines/([\w.-]+)/speech", self.path):
            self.proxy(m.group(1), "/v1/audio/speech")
        else:
            self.send_json(404, {"error": "not found"})

    def do_DELETE(self):
        if m := re.fullmatch(r"/api/phrases/(\d+)", self.path):
            with lock:
                phrases = load_phrases()
                keep = [p for p in phrases if p["id"] != int(m.group(1))]
                if len(keep) == len(phrases):
                    return self.send_json(404, {"error": "no such phrase"})
                save_phrases(keep)
            self.send_json(200, {"deleted": int(m.group(1))})
        else:
            self.send_json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "GET /api/" in msg:
            return  # quiet the 2s polling; actions and errors still print
        print(self.log_date_time_string(), msg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=None,
                    help="ADD a second listener on this address (e.g. this "
                         "box's tailnet IP), same port. 127.0.0.1 is always "
                         "served — local tools depend on it. Same semantics as "
                         "`engines/tts --host`. 0.0.0.0 is refused.")
    args = ap.parse_args()
    # This console has NO authentication and it drives docker: whoever reaches
    # it can start and stop containers on this machine. The network boundary is
    # the auth boundary, so binding every interface — including whatever wifi
    # this box is on — is never a bounded choice.
    if args.host == "0.0.0.0":
        raise SystemExit("refusing --host 0.0.0.0: this console has no auth and "
                         "commands docker. Bind a specific address instead "
                         "(e.g. this box's tailnet IP).")
    extra = args.host if args.host and args.host != "127.0.0.1" else None
    if extra:
        # An engine started from the page must be reachable wherever this
        # console is: `tts up` reads TTS_HOST and adds a second publish on it,
        # so the child processes inherit our own reach. Without this, an engine
        # restarted from the page comes back loopback-only and every phone on
        # the tailnet loses its voice.
        # Reaching it is not enough: the engine's CORS allowlist defaults to
        # loopback, so a board loaded from this address gets `400 Disallowed
        # CORS origin` on every call and goes silent while looking healthy.
        # Same address, both spellings a browser may use for it (the tailnet
        # name is what the phone actually sends), any port. Explicit wins.
        os.environ["TTS_HOST"] = extra
        os.environ.setdefault("TTS_CORS_ORIGINS", ",".join(
            ["http://127.0.0.1:*", "http://localhost:*"]
            + [f"http://{n}:*" for n in dict.fromkeys([extra, socket.getfqdn(extra)])]))
        second = ThreadingHTTPServer((extra, PORT), Handler)
        threading.Thread(target=second.serve_forever, daemon=True).start()
        print(f"console also on http://{extra}:{PORT}")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"console on http://127.0.0.1:{PORT}"
          + ("" if extra else " (loopback only, by design)"))
    srv.serve_forever()
