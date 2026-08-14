#!/usr/bin/env python3
"""
transcribelib — add a `text` field to every clip in voices/library/manifest.json.

VoxCPM2's ref-audio-plus-transcript mode needs to be told what its reference
clip says. Every clip was rendered from the same seed sentence
(bench/mkvoicelib.py), but not every engine read it faithfully — xtts-v2 speaks
its own full stops out loud ("pontos") — and a continuation prompt whose text
does not match its audio teaches the model the wrong alignment. So the
transcript is what was actually said, heard back by whisper, not the sentence
the clip was asked to read.

Needs a whisper server with POST /transcribe (multipart file + language).

    python3 bench/transcribelib.py                    # default http://127.0.0.1:8878
    python3 bench/transcribelib.py --url http://host:port [--force]

Clips that already carry a `text` are skipped unless --force, so an interrupted
run resumes. Stdlib only.
"""
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

LIB = Path(__file__).resolve().parent.parent / "voices" / "library"
MANIFEST = LIB / "manifest.json"


def post_file(url: str, path: Path, language: str) -> str:
    """One multipart POST, hand-rolled — the stdlib has no multipart encoder."""
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{path.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode(),
        path.read_bytes(), b"\r\n",
        f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n'
        f'{language}\r\n--{boundary}--\r\n'.encode(),
    ])
    req = urllib.request.Request(
        url + "/transcribe", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["text"].strip()


def main():
    args = sys.argv[1:]
    url = "http://127.0.0.1:8878"
    force = "--force" in args
    if "--url" in args:
        url = args[args.index("--url") + 1]
    manifest = json.loads(MANIFEST.read_text())
    todo = [e for e in manifest
            if e.get("file") and (force or not e.get("text"))]
    print(f"{len(todo)} of {len(manifest)} entries to transcribe via {url}")
    for i, e in enumerate(todo, 1):
        clip = LIB / e["file"]
        try:
            e["text"] = post_file(url, clip, e["lang"])
        except (urllib.error.URLError, OSError, KeyError) as exc:
            sys.exit(f"{e['file']}: {exc}\n(is whisper up at {url}?)")
        print(f"[{i}/{len(todo)}] {e['file']}: {e['text'][:70]}", flush=True)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
