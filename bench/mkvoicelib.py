#!/usr/bin/env python3
"""
mkvoicelib — render every engine's built-in voices into voices/library/.

One ~10 s clip per voice per language it speaks, all reading the same seed
sentence (translated), so any cloning engine can borrow any factory voice as a
reference clip. Engines are brought up and down one at a time through
engines/tts (the GPU fits one model). Clips that already exist are skipped, so
an interrupted run resumes. A voice that fails is recorded in the manifest
with its error, not dropped.

Run from the repo root:  python3 bench/mkvoicelib.py
Stdlib only.
"""
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TTS = ROOT / "engines" / "tts"
LIB = ROOT / "voices" / "library"
MANIFEST = LIB / "manifest.json"

# The same sentence for every voice, in the language being rendered — a
# reference clip must be speech in its own language or cloners inherit the
# wrong accent. en/pt are the pair already used across this repo's samples.
SEED = {
    "en": "The harbour was quiet that morning, and the light came in sideways across the water. Nothing much happened, which was exactly what everyone had hoped for.",
    "pt": "O porto estava quieto naquela manhã, e a luz entrava de lado sobre a água. Não aconteceu grande coisa, que era exatamente o que todo mundo esperava.",
    "es": "El puerto estaba tranquilo aquella mañana, y la luz entraba de lado sobre el agua. No pasó gran cosa, que era exactamente lo que todos esperaban.",
    "fr": "Le port était calme ce matin-là, et la lumière arrivait de biais sur l'eau. Il ne s'est pas passé grand-chose, ce qui était exactement ce que tout le monde espérait.",
    "hi": "उस सुबह बंदरगाह शांत था, और रोशनी पानी पर तिरछी आ रही थी। कुछ खास नहीं हुआ, और यही तो सब चाहते थे।",
    "it": "Il porto era tranquillo quella mattina, e la luce arrivava di lato sull'acqua. Non successe granché, ed era esattamente quello che tutti speravano.",
    "ja": "その朝、港は静かで、光が水面に斜めに差し込んでいた。特に何も起こらなかったが、それこそ皆が望んでいたことだった。",
    "zh": "那天早晨，港口很安静，光线斜斜地照在水面上。没有发生什么大事，而这正是大家所希望的。",
}

# Kokoro voice prefix → language (first letter) and gender (second letter,
# vendor-declared: f/m). The only engine that declares gender.
KOKORO_LANG = {"a": "en", "b": "en", "p": "pt", "e": "es", "f": "fr",
               "h": "hi", "i": "it", "j": "ja", "z": "zh"}


def plan_kokoro(voices):
    for v in voices:
        lang = KOKORO_LANG.get(v[0])
        gender = {"f": "female", "m": "male"}.get(v[1] if len(v) > 1 else "", "unknown")
        if lang is None:
            yield v, None, gender  # unknown prefix — recorded as an error
        else:
            yield v, lang, gender


def plan_multilingual(voices):
    for v in voices:
        for lang in ("en", "pt"):
            yield v, lang, "unknown"


def plan_piper(voices):
    for v in voices:
        lang = {"pt": "pt", "en": "en"}.get(v[:2])
        yield v, lang, "unknown"


# engine name (as in the engines/tts registry) → which langs each voice gets
ENGINES = [
    ("piper", plan_piper),
    ("kokoro", plan_kokoro),
    ("qwen3-tts", plan_multilingual),
    ("xtts-v2", plan_multilingual),
]


def port_of(engine):
    m = re.search(rf"^{re.escape(engine)}\s+(\d+)\s", TTS.read_text(), re.M)
    if not m:
        sys.exit(f"engine '{engine}' not in the engines/tts registry")
    return int(m.group(1))


def slug(name):
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def fix_wav_sizes(data):
    """Some engines stream WAVs with 0xFFFFFFFF RIFF/data sizes (kokoro-fastapi
    does) — patch in the real sizes so any reader gets the true duration."""
    i = data.find(b"data", 12)
    if not data.startswith(b"RIFF") or i < 0:
        return data
    data = bytearray(data)
    data[4:8] = (len(data) - 8).to_bytes(4, "little")
    data[i + 4:i + 8] = (len(data) - i - 8).to_bytes(4, "little")
    return bytes(data)


def wav_seconds(path):
    try:
        with wave.open(str(path), "rb") as w:
            secs = w.getnframes() / w.getframerate()
            if secs >= 3600:  # bogus streaming header
                raise ValueError("implausible duration")
            return round(secs, 2)
    except Exception:  # nonstandard header — ask ffprobe
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return round(float(out.stdout.strip()), 2)


def up(engine):
    # The GPU guard in engines/tts refuses to start over another resident GPU
    # engine — that guard is load-bearing on this box, so wait instead of --force.
    for attempt in range(60):
        r = subprocess.run([str(TTS), "up", engine], cwd=ROOT)
        if r.returncode == 0:
            return
        print(f"[{engine}] up refused/failed (attempt {attempt + 1}) — waiting 60 s", flush=True)
        time.sleep(60)
    sys.exit(f"could not bring up {engine} after an hour of retries")


def down(engine):
    subprocess.run([str(TTS), "down", engine], cwd=ROOT)


def get_voices(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/voices", timeout=10) as r:
        return json.load(r)["voices"]


def render(port, text, lang, voice, out_path):
    body = json.dumps({"text": text, "lang": lang, "voice": voice}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tts", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out_path.write_bytes(fix_wav_sizes(r.read()))


def main():
    LIB.mkdir(parents=True, exist_ok=True)
    records, done, failed, skipped = [], 0, 0, 0
    prev = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []

    for engine, plan in ENGINES:
        # If a previous run fully rendered this engine, don't even boot it.
        # (Voices changed? Delete the engine's clips or the manifest to force.)
        prev_e = [r for r in prev if r["engine"] == engine]
        if prev_e and all(r.get("file") and not r.get("error")
                          and (LIB / r["file"]).exists() for r in prev_e):
            print(f"[{engine}] all {len(prev_e)} clips already present — skipping engine", flush=True)
            records.extend(prev_e)
            skipped += len(prev_e)
            continue
        port = port_of(engine)
        up(engine)
        try:
            jobs = list(plan(get_voices(port)))
            print(f"[{engine}] {len(jobs)} clips planned", flush=True)
            for voice, lang, gender in jobs:
                rec = {"file": None, "engine": engine, "voice": voice,
                       "lang": lang, "gender": gender}
                if lang is None:
                    rec["error"] = "could not map voice to a language"
                    records.append(rec)
                    failed += 1
                    continue
                rec["file"] = f"{engine}-{slug(voice)}-{lang}.wav"
                path = LIB / rec["file"]
                if path.exists():
                    skipped += 1
                else:
                    try:
                        render(port, SEED[lang], lang, voice, path)
                        done += 1
                        print(f"  {rec['file']}", flush=True)
                    except urllib.error.HTTPError as e:
                        rec["error"] = e.read().decode(errors="replace")[:300]
                    except Exception as e:
                        rec["error"] = str(e)[:300]
                if "error" in rec:
                    rec["file"] = None
                    failed += 1
                    print(f"  FAIL {engine}/{voice}/{lang}: {rec['error']}", flush=True)
                else:
                    rec["seconds"] = wav_seconds(path)
                records.append(rec)
        finally:
            down(engine)

    records.sort(key=lambda r: (r["engine"], r["voice"], r["lang"] or ""))
    MANIFEST.write_text(json.dumps(records, ensure_ascii=False, indent=1) + "\n")
    print(f"\n{done} rendered, {skipped} already present, {failed} failed "
          f"— manifest: {len(records)} records at {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
