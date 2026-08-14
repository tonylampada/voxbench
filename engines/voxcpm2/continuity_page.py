#!/usr/bin/env python3
"""Turn a continuity.py run into a page you can listen to.

    python3 engines/voxcpm2/continuity_page.py <dir-with-metrics.json>

Reads metrics.json and the wavs beside it, writes index.html in the same
directory referencing them relatively. Stdlib only, like everything host-side
here. The broken take comes first; every lever after it says what changed.
"""
import html
import json
import sys
from pathlib import Path

# The order the takes are told in. Two families, and the page says so: a voice
# cloned from a library clip, and a voice invented from a description. Anything
# in metrics.json not named here still gets a card, at the end of its family.
FAMILIES = [
    ("A cloned voice", "A library clip is the reference, which is how the board speaks today.",
     ["v0-baseline", "v2-ref-plus-transcript", "v1-chained", "v1b-chained-noref",
      "v7-transcript-then-chained"]),
    ("A described voice", "No clip at all — the voice is invented from a sentence of "
     "description, and this is where the drift is worst.",
     ["v4-description", "v5-description-anchored", "v6-description-chained",
      "v8-description-anchored-chained"]),
    ("Fewer, longer pieces", "Same conditioning as the shipped take, cut coarser — the "
     "README's ~800-character ceiling, tested rather than trusted.", []),
]

# What ships today -> what this page says should ship instead. These are the
# pairs put up top, back to back; picking by best score alone would put the
# chained takes there, and the chained takes are the trap the page is about.
RECOMMENDED = {"v0-baseline": "v2-ref-plus-transcript",
               "v4-description": "v5-description-anchored"}


def rate_spread(take) -> float:
    """How much the speaking rate moves across a take, as a percentage of its
    own median. Identity embeddings are deliberately blind to pace; the ear is
    not, and 'became a different character' is as often tempo as timbre."""
    r = sorted(s["chars_per_s"] for s in take.get("stats", []) if s["chars_per_s"])
    if len(r) < 2:
        return 0.0
    mid = r[len(r) // 2]
    return (r[-1] - r[0]) / mid * 100


def bar(value, lo, hi):
    """A similarity as a filled bar between the run's own floor and ceiling.
    Below the halfway mark it turns the fault colour — that is the half of the
    scale where the take has stopped being one person."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0
    cls = "fill" + ("" if frac >= 0.75 else " warn" if frac >= 0.5 else " bad")
    return (f'<span class="bar"><span class="{cls}" style="width:{frac * 100:.0f}%"></span></span>'
            f'<span class="num">{value:.3f}</span>')


def seam_strip(take, floor, ceil):
    """The take's pieces in a row, with the seam similarity printed between
    them — this is where the ear hears the person change."""
    stats = take.get("stats", [])
    seams = take.get("seams") or []
    out = []
    for i, st in enumerate(stats):
        out.append(
            f'<div class="piece"><div class="phead">piece {i + 1}'
            f'<span class="pm">{st["pitch_hz"]:.0f} Hz &middot; {st["chars_per_s"]:.1f} ch/s</span></div>'
            f'<audio controls preload="none" src="{take["name"]}/piece_{i:02d}.wav"></audio>'
            f'<p class="ptext">{html.escape(st["text"][:110])}{"…" if len(st["text"]) > 110 else ""}</p></div>')
        if i < len(seams):
            s = seams[i]
            klass = "seam" + (" break" if s < floor + (ceil - floor) * 0.5 else "")
            out.append(f'<div class="{klass}"><span>{s:.3f}</span></div>')
    return f'<div class="strip">{"".join(out)}</div>'


def take_card(take, floor, ceil, rank=None):
    if take.get("failed"):
        return (f'<section class="take failed"><h3>{html.escape(take["name"])}</h3>'
                f'<p class="note">{html.escape(take["note"])}</p>'
                f'<p class="fail">did not survive: <code>{html.escape(take["failed"])}</code></p></section>')
    badge = f'<span class="rank">{rank}</span>' if rank else ""
    if take.get("seam_min") is None:  # one piece: no seam exists to measure
        return f"""<section class="take" id="{take['name']}">
  <h3>{badge}{html.escape(take['name'])}</h3>
  <p class="note">{html.escape(take['note'])}</p>
  <div class="headline">
    <audio controls preload="none" src="{take['wav']}"></audio>
    <dl><div><dt>seams</dt><dd><span class="dim">one decode, nothing to compare —
      listen for whether it holds together at all</span></dd></div>
    <div><dt>cost</dt><dd><span class="num">{take['realtime_x']:.2f}&times; realtime</span>
      <span class="dim">{take['gen_s']:.0f} s for {take['audio_s']:.0f} s</span></dd></div></dl>
  </div>
</section>"""
    return f"""<section class="take" id="{take['name']}">
  <h3>{badge}{html.escape(take['name'])}</h3>
  <p class="note">{html.escape(take['note'])}</p>
  <div class="headline">
    <audio controls preload="none" src="{take['wav']}"></audio>
    <dl>
      <div><dt>worst seam</dt><dd>{bar(take['seam_min'], floor, ceil)}</dd></div>
      <div><dt>mean seam</dt><dd>{bar(take['seam_mean'], floor, ceil)}</dd></div>
      <div><dt>furthest from its own opening</dt><dd>{bar(take['drift_from_first'], floor, ceil)}</dd></div>
      <div><dt>pitch spread</dt><dd><span class="num">{take['pitch_spread_hz']:.0f} Hz</span></dd></div>
      <div><dt>speaking-rate spread</dt><dd><span class="num">{rate_spread(take):.0f}%</span></dd></div>
      <div><dt>cost</dt><dd><span class="num">{take['realtime_x']:.2f}&times; realtime</span>
        <span class="dim">{take['pieces']} pieces, {take['gen_s']:.0f} s for {take['audio_s']:.0f} s</span></dd></div>
    </dl>
  </div>
  <details><summary>piece by piece</summary>{seam_strip(take, floor, ceil)}</details>
</section>"""


def table(takes, floor, ceil):
    rows = []
    for t in sorted([t for t in takes if not t.get("failed")
                     and t.get("seam_min") is not None], key=lambda t: -t["seam_min"]):
        rows.append(
            f'<tr><th scope="row"><a href="#{t["name"]}">{html.escape(t["name"])}</a></th>'
            f'<td>{t["seam_min"]:.3f}</td><td>{t["seam_mean"]:.3f}</td>'
            f'<td>{t["drift_from_first"]:.3f}</td><td>{t["pitch_spread_hz"]:.0f} Hz</td>'
            f'<td>{rate_spread(t):.0f}%</td>'
            f'<td>{t["pieces"]}</td><td>{t["realtime_x"]:.2f}&times;</td></tr>')
    for t in takes:
        if t.get("failed"):
            rows.append(f'<tr class="dead"><th scope="row">{html.escape(t["name"])}</th>'
                        f'<td colspan="7">{html.escape(t["failed"][:90])}</td></tr>')
        elif t.get("seam_min") is None:
            rows.append(f'<tr><th scope="row">{html.escape(t["name"])}</th>'
                        f'<td colspan="5" class="dim">one decode — no seam to measure</td>'
                        f'<td>{t["pieces"]}</td><td>{t["realtime_x"]:.2f}&times;</td></tr>')
    return f"""<div class="scroll"><table>
<thead><tr><th>take</th><th>worst seam</th><th>mean seam</th><th>vs opening</th>
<th>pitch spread</th><th>rate spread</th><th>pieces</th><th>speed</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>"""


def back_to_back(broken, best, floor, ceil, family=""):
    """The whole card in one panel: the shipped take and the best one, same
    passage, one after the other. Nothing on screen says which to prefer."""
    family = f"{family} &middot; " if family else ""
    def side(t, label):
        frac = (t["seam_min"] - floor) / (ceil - floor)
        cls = "good" if frac >= 0.75 else "bad"
        return (f'<div class="panel"><h4>{family}{label}</h4>'
                f'<audio controls preload="none" src="{t["wav"]}"></audio>'
                f'<p class="note" style="margin:.7rem 0 .2rem">worst seam</p>'
                f'<div class="big {cls}">{t["seam_min"]:.3f}</div>'
                f'<p class="note" style="margin:.4rem 0 0">{html.escape(t["note"])}</p></div>')
    return f'<div class="ab">{side(broken, "take one")}{side(best, "take two")}</div>'


def passage_panel(text: str) -> str:
    """Long enough to be the point, long enough to bury the page. Folded."""
    head = text[:180].rsplit(" ", 1)[0]
    return (f'<details class="panel"><summary class="passage">{html.escape(head)}…'
            f'<span class="dim"> ({len(text)} characters — open it)</span></summary>'
            f'<p class="passage" style="margin:.8rem 0 0">{html.escape(text)}</p></details>')


def page(m: dict, verdict: str) -> str:
    a = m["anchors"]
    floor, ceil = a["different_voice"], a["same_voice"]
    takes = {t["name"]: t for t in m["takes"]}
    named = {n for _, _, names in FAMILIES for n in names}
    cards = ""
    for title, blurb, names in FAMILIES:
        here = [n for n in names if n in takes] or \
            ([n for n in takes if n not in named] if not names else [])
        if not here:
            continue
        cards += (f'<h2>{html.escape(title)}</h2><p class="note">{html.escape(blurb)}</p>'
                  + "".join(take_card(takes[n], floor, ceil,
                                      rank="what ships today" if n in ("v0-baseline",
                                                                       "v4-description") else None)
                            for n in here))
    ab = "".join(back_to_back(takes[a], takes[b], floor, ceil,
                              next(t for t, _, n in FAMILIES if a in n).lower())
                 for a, b in RECOMMENDED.items() if a in takes and b in takes)
    sep = m.get("description_separation", [])
    seprows = "".join(
        f'<tr><th scope="row">{s["tag"]} / seed {s["seed"]}</th>'
        f'<td><audio controls preload="none" src="{s["wav"]}"></audio></td>'
        f'<td>{s["pitch_hz"]:.0f} Hz</td>'
        f'<td>{" &middot; ".join(f"{k}&nbsp;{v:.3f}" for k, v in sorted(s["vs"].items()))}</td>'
        f'</tr>' for s in sep)
    # Same description, different seed vs different description, same seed. If the
    # second is the higher number, the seed is choosing the voice, not the words.
    def pair(a, b):
        for s in sep:
            if s["tag"] + str(s["seed"]) == a and b in s["vs"]:
                return s["vs"][b]
        return None
    same_desc = [v for v in (pair("A42", "A7"), pair("B42", "B7")) if v is not None]
    same_seed = [v for v in (pair("A42", "B42"), pair("A7", "B7")) if v is not None]
    sepverdict = ""
    if same_desc and same_seed:
        sepverdict = (
            f'<div class="panel scale"><div>same description, two seeds &rarr; '
            f'<b>{sum(same_desc) / len(same_desc):.3f}</b> '
            f'<span class="dim">two people</span></div>'
            f'<div>two opposite descriptions, same seed &rarr; '
            f'<b>{sum(same_seed) / len(same_seed):.3f}</b> '
            f'<span class="dim">nearly one person</span></div></div>')
    return f"""<title>Voice continuity — voxcpm2</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  /* The console's listening room, same palette as console/test.html: warm
     near-black ground, one amber signal reserved for sound and action, a green
     lamp for good and a clay fault for bad, mono for evidence. */
  :root {{
    color-scheme: dark;
    --ground: #141110; --panel: #1c1815; --inset: #241d17; --line: #332b23;
    --text: #ede6d9; --muted: #9c9183; --faint: #6f6659;
    --signal: #e8a343; --lamp: #8fc172; --fault: #d97b6c;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 2rem clamp(1rem, 4vw, 2.4rem) 5rem; background: var(--ground);
          color: var(--text); font: 15px/1.55 system-ui, sans-serif; }}
  main {{ max-width: 66rem; margin: 0 auto; }}
  h1 {{ margin: 0 0 .3rem; font-size: 1.05rem; font-weight: 650; letter-spacing: .14em;
        text-transform: uppercase; }}
  h1::before {{ content: ""; display: inline-block; width: .55em; height: .55em;
                background: var(--signal); border-radius: 1px; margin-right: .6em; }}
  h2 {{ font-size: .68rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted);
        font-weight: 600; margin: 2.6rem 0 .8rem; }}
  h3 {{ font-size: .95rem; margin: 0 0 .2rem; font-family: var(--mono); color: var(--text); }}
  .sub {{ color: var(--muted); font-size: .88rem; margin: 0 0 1.6rem;
          border-bottom: 1px solid var(--line); padding-bottom: .9rem; }}
  .note {{ color: var(--faint); font-size: .82rem; margin: .5rem 0 1rem; }}
  .dim {{ color: var(--faint); font-size: .78rem; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
            padding: 1rem 1.15rem; margin-bottom: .8rem; }}
  .passage {{ color: #d8cfbf; }}
  .scale {{ display: flex; flex-wrap: wrap; gap: 1.6rem; align-items: center; }}
  .scale b {{ font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--signal); }}
  .take {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
           padding: 1.05rem 1.15rem; margin-bottom: .8rem; }}
  .take.failed {{ border-color: #4a3028; }}
  .fail {{ color: var(--fault); font-size: .85rem; margin: 0; }}
  .rank {{ display: inline-block; background: #3a2a18; color: var(--signal);
           font: 600 10px/1.8 system-ui; letter-spacing: .14em; text-transform: uppercase;
           border-radius: 999px; padding: 0 .7rem; margin-right: .6rem; vertical-align: middle; }}
  .headline {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 1.2rem;
               align-items: start; margin-top: .8rem; }}
  @media (max-width: 48rem) {{ .headline {{ grid-template-columns: minmax(0, 1fr); }} }}
  dl {{ margin: 0; display: grid; gap: .3rem; }}
  dl div {{ display: grid; grid-template-columns: minmax(0, 12rem) minmax(0, 1fr); gap: .6rem;
            align-items: center; }}
  dt {{ color: var(--muted); font-size: .78rem; }}
  dd {{ margin: 0; display: flex; align-items: center; gap: .55rem; flex-wrap: wrap; }}
  .bar {{ display: inline-block; width: 6.5rem; height: 6px; border-radius: 4px;
          background: var(--inset); overflow: hidden; flex: none; }}
  .fill {{ display: block; height: 100%; background: var(--lamp); }}
  .fill.warn {{ background: var(--signal); }}
  .fill.bad {{ background: var(--fault); }}
  .ab {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr));
         gap: .8rem; }}
  .ab .panel {{ margin: 0; }}
  .ab h4 {{ margin: 0 0 .5rem; font: 600 .68rem/1.6 system-ui; letter-spacing: .16em;
            text-transform: uppercase; color: var(--muted); }}
  .ab .big {{ font: 1.5rem/1 var(--mono); color: var(--text); }}
  .ab .big.bad {{ color: var(--fault); }}
  .ab .big.good {{ color: var(--lamp); }}
  .num {{ font: 13px/1 var(--mono); color: var(--text); font-variant-numeric: tabular-nums; }}
  audio {{ width: 100%; min-width: 0; height: 34px; }}
  details {{ margin-top: .9rem; }}
  summary {{ cursor: pointer; color: var(--muted); font-size: .8rem; }}
  .strip {{ display: flex; gap: 0; align-items: stretch; overflow-x: auto; padding: .9rem 0 .3rem; }}
  .piece {{ flex: 0 0 15rem; background: var(--inset); border: 1px solid var(--line);
            border-radius: 9px; padding: .55rem .7rem; }}
  .phead {{ font-size: .72rem; color: var(--muted); display: flex; justify-content: space-between;
            gap: .5rem; margin-bottom: .35rem; }}
  .pm {{ font: 11px/1.6 var(--mono); color: var(--faint); }}
  .ptext {{ font-size: .74rem; color: var(--faint); margin: .35rem 0 0; }}
  .seam {{ flex: 0 0 4.6rem; display: flex; align-items: center; justify-content: center; }}
  .seam span {{ font: 12px/1 var(--mono); color: var(--lamp); border-top: 1px dashed var(--line);
                padding-top: .3rem; }}
  .seam.break span {{ color: var(--fault); border-color: #4a3028; }}
  /* Eight columns of numbers do not fit a phone. The table scrolls inside its
     own box rather than making the page scroll sideways. */
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 34rem; font-size: .88rem; }}
  th, td {{ text-align: left; padding: .5rem .75rem; border-bottom: 1px solid var(--line);
            font-variant-numeric: tabular-nums; }}
  thead th {{ font-size: .66rem; letter-spacing: .14em; text-transform: uppercase;
              color: var(--muted); }}
  tbody th {{ font-weight: 400; font-family: var(--mono); white-space: nowrap; }}
  tbody tr:first-child td {{ color: var(--lamp); }}
  tr.dead th, tr.dead td {{ color: var(--fault); }}
  a {{ color: var(--signal); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ font: 13px/1.4 var(--mono); color: var(--signal); }}
  .verdict {{ background: var(--panel); border: 1px solid var(--signal); border-radius: 14px;
              padding: 1.1rem 1.3rem; }}
  .verdict p {{ margin: 0 0 .7rem; }}
  .verdict p:last-child {{ margin: 0; }}
  .verdict b {{ color: var(--signal); }}
</style>
<main>
<h1>The voice changes character mid-sentence</h1>
<p class="sub">One passage, one reference clip, one seed. Everything below differs by
exactly one thing: what each piece of the take was conditioned on.</p>

<h2>The passage</h2>
{passage_panel(m["passage"])}
<p class="note">Reference clip <code>{html.escape(m["reference"])}</code>, seed {m["seed"]},
cut by the shipped 400-character splitter unless a take says otherwise.</p>

<h2>The number, and its scale</h2>
<div class="panel scale">
  <div>Speaker-embedding cosine similarity between consecutive pieces (resemblyzer d-vector).</div>
</div>
<div class="panel scale">
  <div>same person, two takes &rarr; <b>{ceil:.3f}</b>
    <span class="dim">the ceiling</span></div>
  <div>two different library voices &rarr; <b>{floor:.3f}</b>
    <span class="dim">the floor</span></div>
</div>
<p class="note">A seam near the ceiling is one person talking. A seam down near the floor
is the moment the captain hears somebody else. Anchors are on this page too:
<audio controls preload="none" src="anchor_same_a.wav" style="width:11rem"></audio>
<audio controls preload="none" src="anchor_same_b.wav" style="width:11rem"></audio> are the
same voice; <audio controls preload="none" src="anchor_other.wav" style="width:11rem"></audio>
is the other one.</p>

<h2>Back to back</h2>
<p class="note">Two pairs, each one the same passage, the same voice and the same seed.
In both, one of them is what the board speaks with today.</p>
{ab}

<h2>Verdict</h2>
<div class="verdict">{verdict}</div>

<h2>Every take, ranked</h2>
{table(m["takes"], floor, ceil)}

{cards}

<h2>Does a description steer anything?</h2>
<p class="note">Two descriptions, two seeds, same sentence. A: <i>{html.escape(m["descriptions"]["A"])}</i>
&middot; B: <i>{html.escape(m["descriptions"]["B"])}</i>. If A and B score like the same person,
the description is decoration.</p>
{sepverdict}
<div class="scroll"><table><thead><tr><th>take</th><th>listen</th><th>pitch</th><th>similarity to the others</th></tr></thead>
<tbody>{seprows}</tbody></table></div>
</main>
"""


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = Path(sys.argv[1])
    m = json.loads((d / "metrics.json").read_text())
    verdict = (d / "verdict.html")
    (d / "index.html").write_text(page(m, verdict.read_text() if verdict.exists()
                                       else "<p>Pending.</p>"))
    print(f"wrote {d}/index.html — {len(m['takes'])} takes")


if __name__ == "__main__":
    main()
