#!/usr/bin/env python3
"""Build the described-voices gallery: one long passage per language, spoken by
every voice in DESCRIPTIONS, plus one description at two seeds.

    python3 gallery.py --out <dir> [--engine http://127.0.0.1:8883] [--force]

The wavs land next to index.html in --out and are referenced relatively, so the
directory is the whole artifact — move it, zip it, serve it, it still plays.
"""
import argparse
import io
import json
import re
import time
import unicodedata
import urllib.request
import wave
from html import escape
from pathlib import Path

# One passage per language, long enough to judge a voice on: forty-plus seconds
# with a statement, a question, a pause, a number and a name in it. A voice that
# holds together for ten words often falls apart at forty.
PASSAGES = {
    "en": (
        "The lighthouse keeper wrote the same three words in the log every night, for eleven years. "
        "Nobody ever asked him what they meant. When the inspector finally came out on the ferry — a woman "
        "named Halloran, in a coat far too thin for that weather — she read the whole book, first page to "
        "last, then closed it and sat there a while. Did he ever say anything to you about it? she asked. "
        "No, I said. He was not a talkative man. She nodded slowly, the way people do when the answer is "
        "exactly the one they expected and it still is not enough. Outside, the light went round, and round, "
        "and round again. Eleven years. Three words. And a hundred and forty pages of weather, tides, and "
        "the count of ships that passed in the night without once looking back at the shore. She took the "
        "book with her on the evening boat. I stood on the pier until I could not see it any more, and then "
        "I climbed back up and lit the lamp, because that is what the job is."
    ),
    "pt": (
        "O faroleiro escreveu as mesmas três palavras no diário todas as noites, durante onze anos. "
        "Ninguém nunca perguntou o que elas queriam dizer. Quando a fiscal enfim desembarcou da balsa — uma "
        "mulher chamada Halloran, de casaco fino demais para aquele frio — leu o caderno inteiro, da primeira "
        "à última página, fechou o caderno e ficou ali sentada um tempo. Ele chegou a te dizer alguma coisa "
        "sobre isso? ela perguntou. Não, eu disse. Ele não era homem de falar. Ela balançou a cabeça devagar, "
        "do jeito que a gente faz quando a resposta é exatamente a que esperava e mesmo assim não basta. "
        "Lá fora, a luz girava, e girava, e girava de novo. Onze anos. Três palavras. E cento e quarenta "
        "páginas de tempo, maré, e a conta dos navios que passaram na noite sem nunca olhar para trás. "
        "Ela levou o caderno no barco da tarde. Fiquei no píer até não enxergar mais nada, subi de volta e "
        "acendi a lâmpada, porque o trabalho é esse."
    ),
}

DESCRIPTIONS = {
    "en": [
        "a young woman, bright and quick, smiling as she speaks",
        "a woman in her forties, low and dry, quietly amused by all of it",
        "an old woman, frail and papery, taking her time with every word",
        "a female newsreader, crisp and neutral, reading the evening headlines",
        "a cold clinical woman, precise, with no warmth at all",
        "a warm radio narrator, low and unhurried, telling a bedtime story",
        "a very old man, gravelly and slow, running out of breath",
        "an excited teenage boy talking fast, words tumbling over each other",
    ],
    "pt": [
        "uma mulher jovem e calorosa, falando devagar e com carinho",
        "uma apresentadora de telejornal, dicção perfeita, tom neutro e firme",
        "uma senhora idosa, voz trêmula e doce, contando uma história antiga",
        "uma mulher séria e impaciente, falando rápido, sem paciência nenhuma",
        "uma menina pequena, voz aguda e animada",
        "um locutor de rádio brasileiro, voz grave e confiante",
        "um homem velho e cansado, voz rouca, arrastando as palavras",
        "um homem irritado, falando alto, quase gritando",
    ],
}

# Which of the above are women, so the page can say so and the count can be checked.
WOMEN = 5

SEED_PAIR = ("en", "a smoker's voice, deep and rough, amused by something", [42, 7])
SEED = 42


def slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", ascii_text.lower())).strip("-")[:72]


def synth(engine: str, text: str, lang: str, description: str, seed: int) -> bytes:
    body = json.dumps({
        "input": text, "lang": lang, "response_format": "wav",
        "params": {"seed": seed, "voice_prompt": description},
    }).encode()
    req = urllib.request.Request(f"{engine}/v1/audio/speech", body,
                                 {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800).read()


def seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / w.getframerate()


def take(engine: str, out: Path, lang: str, description: str, seed: int, force: bool) -> dict:
    name = f"{lang}-{slug(description)}"
    if seed != SEED:
        name += f"-seed{seed}"
    path = out / f"{name}.wav"
    if path.exists() and not force:
        return {"desc": description, "lang": lang, "seed": seed, "file": path.name,
                "secs": seconds(path.read_bytes())}
    t = time.time()
    audio = synth(engine, PASSAGES[lang], lang, description, seed)
    path.write_bytes(audio)
    secs = seconds(audio)
    print(f"  {secs:5.1f}s audio in {time.time() - t:5.1f}s  {path.name}")
    return {"desc": description, "lang": lang, "seed": seed, "file": path.name, "secs": secs}


CSS = """
  :root { color-scheme: dark; }
  body { margin: 0; padding: 2.5rem 3rem 5rem; background: #0d1013; color: #d7dee6;
          font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  h1 { font-size: 1.5rem; font-weight: 600; margin: 0 0 .3rem; }
  .sub { color: #7d8894; margin: 0 0 2rem; max-width: 46rem; }
  .note { color: #7d8894; font-size: .9rem; max-width: 46rem; }
  .passages { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
               align-items: start; gap: 1.25rem; margin-bottom: 2.5rem; }
  .passage { background: #151a20; border: 1px solid #232b34; border-radius: 10px; padding: 1.1rem 1.3rem; }
  .passage h2 { font-size: .75rem; letter-spacing: .12em; text-transform: uppercase;
                 color: #6ea8fe; margin: 0 0 .6rem; }
  .passage p { margin: 0; font-size: 1.02rem; line-height: 1.6; color: #eef3f8; }
  h2.band { font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: #6ea8fe;
             margin: 2.5rem 0 .8rem; }
  h2.band + p.note { margin: -.4rem 0 .9rem; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: .7rem 1rem; border-bottom: 1px solid #1e252d; }
  tbody th { font-weight: 400; width: 52%; }
  .desc { display: block; color: #eef3f8; font-size: 1.05rem; }
  .meta { display: block; font-size: .75rem; color: #7d8894; margin-top: .15rem;
           font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  code { font: 13px/1 ui-monospace, SFMono-Regular, Menlo, monospace; color: #9ecbff; }
  audio { width: 100%; min-width: 17rem; height: 34px; }
  tbody tr:hover { background: #12171d; }
"""


def rows(takes: list[dict]) -> str:
    return "\n".join(
        f'<tr><th scope="row"><span class="desc">{escape(t["desc"])}</span>'
        f'<span class="meta">{t["lang"]} · seed {t["seed"]} · {t["secs"]:.0f}s</span></th>'
        f'<td><audio controls preload="none" src="{escape(t["file"])}"></audio></td></tr>'
        for t in takes)


def render(en: list[dict], pt: list[dict], pair: list[dict]) -> str:
    shortest = min(t["secs"] for t in en + pt + pair)
    return f"""<title>voices from a description</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<h1>voxcpm2 — voices from a written description</h1>
<p class="sub">No reference clip anywhere on this page. Each voice was asked for in words —
<code>params.voice_prompt</code> — and the model invented one to match. The passage is the same
inside each language, so the description is the only thing that changes. It runs about a minute
on purpose: a voice that holds together for ten words often falls apart at forty.</p>
<div class="passages">
  <div class="passage"><h2>English</h2><p>{escape(PASSAGES["en"])}</p></div>
  <div class="passage"><h2>Português</h2><p>{escape(PASSAGES["pt"])}</p></div>
</div>
<h2 class="band">English descriptions</h2>
<p class="note">{WOMEN} of {len(en)} are women.</p>
<table><tbody>
{rows(en)}
</tbody></table>
<h2 class="band">Descrições em português</h2>
<p class="note">{WOMEN} de {len(pt)} são mulheres.</p>
<table><tbody>
{rows(pt)}
</tbody></table>
<h2 class="band">One description, two seeds</h2>
<p class="note">How much a description pins down, and how much it leaves to the dice. Same words,
same passage, different seed — the character survives, the person does not.</p>
<table><tbody>
{rows(pair)}
</tbody></table>
<p class="note" style="margin-top:2.5rem">Everything above is default knobs (cfg 2.0, 10 timesteps)
at seed {SEED} unless the row says otherwise. Shortest take on the page is {shortest:.0f}s. Same
description + same seed is byte-identical audio, so a described voice you like is a voice you can
keep.</p>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", default="http://127.0.0.1:8883")
    ap.add_argument("--force", action="store_true", help="re-synthesize wavs that already exist")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    done = {}
    for lang in ("en", "pt"):
        print(f"{lang}: {len(PASSAGES[lang])} characters")
        done[lang] = [take(args.engine, args.out, lang, d, SEED, args.force)
                      for d in DESCRIPTIONS[lang]]
    pair_lang, pair_desc, pair_seeds = SEED_PAIR
    pair = [take(args.engine, args.out, pair_lang, pair_desc, s, args.force) for s in pair_seeds]

    (args.out / "index.html").write_text(render(done["en"], done["pt"], pair))
    short = min(t["secs"] for t in done["en"] + done["pt"] + pair)
    print(f"\n{args.out / 'index.html'} — {len(done['en']) + len(done['pt']) + len(pair)} takes, "
          f"shortest {short:.1f}s")


if __name__ == "__main__":
    main()
