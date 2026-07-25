# Voice library

One ~10 s reference clip for every built-in voice of every engine that ships
voices, in every language that voice speaks — so the factory voices of Kokoro,
XTTS-v2, Qwen3-TTS and Piper are usable as cloning references by any engine
that clones (chatterbox, f5, nano, …). Every clip reads the same seed sentence,
translated into the language being rendered — a reference clip must be speech
in its own language or cloners inherit the wrong accent.

- **Naming**: `<engine>-<voice>-<lang>.wav` (voice lowercased, non-alphanumerics
  collapsed to `-`, accents stripped: `Ana Florence` → `xtts-v2-ana-florence-pt.wav`).
- **`manifest.json`**: one record per planned clip — `file`, `engine`, `voice`
  (the vendor's exact name, what the engine's `/tts` accepts), `lang`, `gender`
  (only where the vendor declares it — Kokoro's name prefix; everyone else is
  `unknown`), `seconds`. A voice that failed to render has `"file": null` and
  an `error` instead.
- **Languages**: Kokoro voices are single-language (mapped from the name
  prefix); XTTS-v2 and Qwen3-TTS speakers are multilingual and are rendered in
  both `en` and `pt`; Piper voices are rendered in the one language they were
  trained for.
- **Length**: ~10 s targets every cloner's window in this repo (XTTS ≥ 6 s,
  Chatterbox ≥ 5 s, F5 ≤ 12 s). Actual range is 7.3–13.6 s (see the manifest's
  `seconds`); every clip clears the ≥ 5/6 s minimums, but 22 slower takes run
  12–13.6 s, just past F5's 12 s ceiling — trim those before feeding them to F5.
- **Known gaps**: Kokoro's 5 Japanese voices (`jf_*`, `jm_*`) fail upstream —
  the pinned kokoro-fastapi image ships without Japanese G2P (`misaki[ja]`) —
  and sit in the manifest with their error.

Regenerate (resumes — existing clips are skipped, only missing/failed ones are
re-rendered; engines are brought up and down one at a time via `engines/tts`):

```
python3 bench/mkvoicelib.py
```
