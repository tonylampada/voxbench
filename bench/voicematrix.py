#!/usr/bin/env python3
"""Render every voice an engine has in one language as a listenable page.

    engines/tts up vibevoice
    python3 bench/voicematrix.py vibevoice en /path/to/outdir

Writes <outdir>/index.html plus one WAV per voice per sentence, beside it and
referenced relatively — the board serves an .html artifact from its own
directory, so a data URI or a second artifact would be the wrong shape.

Only builtin voices are rendered: a reference voice is the shared clip library,
which is the same catalogue for every engine and not what this page compares.
Stdlib only, like everything else host-side here.
"""
import html
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

SHORT = "Build's green, Captain."

LONG = (
    "Overnight the fleet ran clean. Every engine came up on the first try, the "
    "conformance harness found nothing new, and the long paragraph held together for "
    "forty-two seconds without a seam. Two pull requests are merged, one is waiting on "
    "your review, and the card that was blocked since Tuesday is moving again — the "
    "missing seed sentences turned out to be a typo in the manifest and not a problem "
    "with the model at all. Nothing is on fire. The card is idle, the disks have room, "
    "and I will hold here until you tell me otherwise."
)

CELLS = [("short", SHORT), ("long", LONG)]


def registry_port(engine: str) -> int:
    """The port engines/tts would use — parsed from its registry, not duplicated."""
    tts = (Path(__file__).resolve().parent.parent / "engines" / "tts").read_text()
    for line in tts.splitlines():
        f = line.split()
        if len(f) == 5 and f[0] == engine and f[1].isdigit():
            return int(f[1])
    sys.exit(f"no engine '{engine}' in engines/tts's registry")


def get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def speak(base: str, lang: str, voice: str, text: str, out: Path):
    body = json.dumps({"input": text, "lang": lang, "voice": voice}).encode()
    req = urllib.request.Request(base + "/v1/audio/speech", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out.write_bytes(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{voice}: {e.code} {e.read()[:300].decode('utf-8', 'replace')}")


def page(engine: str, lang: str, voices: list[dict], excluded: int) -> str:
    rows = []
    for v in voices:
        cells = "".join(
            f'<td><audio controls preload="none" src="{engine}-{v["id"]}-{key}.wav"></audio></td>'
            for key, _ in CELLS)
        rows.append(
            f'<tr><th scope="row"><span class="name">{html.escape(v["name"])}</span>'
            f'<span class="gender">{html.escape(v.get("gender", ""))}</span>'
            f'<code>{html.escape(v["id"])}</code></th>{cells}</tr>')
    note = (f"<p class=\"note\">{excluded} preset voices in other languages are not on this "
            f"page — this is every <b>{lang}</b> voice the engine has.</p>") if excluded else ""
    return f"""<title>{engine} — every {lang} voice</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 2.5rem 3rem 5rem; background: #0d1013; color: #d7dee6;
          font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 1.5rem; font-weight: 600; margin: 0 0 .3rem; }}
  .sub {{ color: #7d8894; margin: 0 0 2rem; }}
  .note {{ color: #7d8894; font-size: .9rem; }}
  .sentences {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
                align-items: start; gap: 1.25rem; margin-bottom: 2.5rem; }}
  .sentence {{ background: #151a20; border: 1px solid #232b34; border-radius: 10px; padding: 1.1rem 1.3rem; }}
  .sentence h2 {{ font-size: .75rem; letter-spacing: .12em; text-transform: uppercase;
                  color: #6ea8fe; margin: 0 0 .6rem; }}
  .sentence p {{ margin: 0; font-size: 1.2rem; line-height: 1.55; color: #eef3f8; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: .7rem 1rem; border-bottom: 1px solid #1e252d; }}
  thead th {{ font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: #7d8894;
              border-bottom: 1px solid #2c3641; }}
  tbody th {{ font-weight: 400; white-space: nowrap; }}
  .name {{ display: block; color: #eef3f8; }}
  .gender {{ display: inline-block; font-size: .75rem; color: #7d8894; margin-right: .5rem; }}
  code {{ font: 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace; color: #9ecbff; }}
  audio {{ width: 100%; min-width: 17rem; height: 34px; }}
  tbody tr:hover {{ background: #12171d; }}
</style>
<h1>{engine} — every {lang} voice</h1>
<p class="sub">{len(voices)} builtin presets, same two sentences each. Press anything.</p>
<div class="sentences">
  <div class="sentence"><h2>Short</h2><p>{html.escape(SHORT)}</p></div>
  <div class="sentence"><h2>Long</h2><p>{html.escape(LONG)}</p></div>
</div>
<table>
  <thead><tr><th>Voice</th><th>Short</th><th>Long</th></tr></thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>
{note}
"""


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    engine, lang, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    base = f"http://127.0.0.1:{registry_port(engine)}"
    catalogue = get(base + "/v1/voices")["voices"]
    builtin = [v for v in catalogue if v["kind"] == "builtin"]
    voices = [v for v in builtin if lang in v["langs"]]
    if not voices:
        sys.exit(f"{engine} has no builtin {lang} voice")

    outdir.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(voices, 1):
        for key, text in CELLS:
            wav = outdir / f"{engine}-{v['id']}-{key}.wav"
            if wav.exists():  # a rerun is usually for the page, not the audio
                continue
            print(f"[{i}/{len(voices)}] {v['id']} {key}", flush=True)
            speak(base, lang, v["id"], text, wav)
    (outdir / "index.html").write_text(page(engine, lang, voices, len(builtin) - len(voices)))
    print(f"wrote {outdir}/index.html — {len(voices)} voices x {len(CELLS)} sentences")


if __name__ == "__main__":
    main()
