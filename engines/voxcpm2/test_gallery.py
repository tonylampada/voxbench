#!/usr/bin/env python3
"""The gallery names its own files, twice, in two places that drift silently.

`take()` slugs a description into a filename and `render()` writes that name into
an <audio src>. Get them out of step and the page still looks finished — the
player is just dead. Same for the descriptions: the point of the page is reading
the words that produced the voice, so a truncated or escaped-wrong one is a bug
you cannot see. Nothing else here needs a test; the audio is the model's job.

    python3 engines/voxcpm2/test_gallery.py
"""
import re
import sys
from html import escape as html_escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "samples"))
import gallery as g  # noqa: E402


def fake(lang: str, desc: str, seed: int, secs: float = 51.0) -> dict:
    """What take() returns, built by the same naming rule take() uses."""
    name = f"{lang}-{g.slug(desc)}" + (f"-seed{seed}" if seed != g.SEED else "")
    return {"desc": desc, "lang": lang, "seed": seed, "file": f"{name}.wav", "secs": secs}


def main():
    en = [fake("en", d, g.SEED) for d in g.DESCRIPTIONS["en"]]
    pt = [fake("pt", d, g.SEED) for d in g.DESCRIPTIONS["pt"]]
    pair_lang, pair_desc, pair_seeds = g.SEED_PAIR
    pair = [fake(pair_lang, pair_desc, s) for s in pair_seeds]
    html = g.render(en, pt, pair)

    srcs = re.findall(r'<audio[^>]*src="([^"]+)"', html)
    expected = [t["file"] for t in en + pt + pair]
    assert srcs == expected, f"{srcs} != {expected}"
    print(f"ok   {len(srcs)} players point at the names the generator writes")

    # The apostrophe in "a smoker's voice" is why this is checked as escaped text:
    # the description is printed inside markup, and a raw one would break the row.
    for t in en + pt + pair:
        assert html_escape(t["desc"]) in html, f"description not printed in full: {t['desc']}"
    print(f"ok   all {len(en) + len(pt)} descriptions printed in full")

    assert len(pair) == 2 and pair[0]["file"] != pair[1]["file"], "the seed pair is one file twice"
    assert f"seed {pair_seeds[1]}" in html, "the second seed is not named on the page"
    print("ok   the seed pair is two takes of one description")

    for lang in ("en", "pt"):
        assert html_escape(g.PASSAGES[lang]) in html, f"the {lang} passage must be printed in full"
        assert len(g.PASSAGES[lang]) > 700, f"the {lang} passage is too short to judge a voice on"
        assert g.WOMEN * 2 >= len(g.DESCRIPTIONS[lang]), f"{lang} is not at least half women"
    print("ok   both passages printed in full, and half of each set is women")

    print("\nall good")


if __name__ == "__main__":
    main()
