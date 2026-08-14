# VibeVoice-1.5B engine

Wraps [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B)
(MIT, 1.5B Qwen2.5 backbone + a 7.5 Hz acoustic tokenizer and a 4-layer diffusion head)
behind the v2 contract. 24 kHz mono out, native PCM streaming — and unlike the 0.5B
realtime sibling on 8895, **it clones**: 9 built-in presets plus every one of the 221
clips in `voices/library/`, all of them the same thing to this model, a reference wav.

A second engine, not a replacement. 8895 stays the fastest streamer in the fleet;
this is the one that can be given a voice.

## Where this came from

| Thing | Source | State when this engine was built (2026-08-14) |
|---|---|---|
| Weights | `microsoft/VibeVoice-1.5B` on Hugging Face, `c00898d` | public, ungated, MIT, 5.4 GB |
| Code | `github.com/microsoft/VibeVoice`, pinned to `94da20d` | public, MIT — the same pin `engines/vibevoice` runs |
| `modeling_vibevoice_inference.py` | `github.com/vibevoice-community/VibeVoice`, pinned to `631804b` | MIT |
| Preset voices | `demo/voices/*.wav` in the same community checkout | 9 wavs |

**Two files come from the community fork, and both are load-bearing.**

`modeling_vibevoice_inference.py` is the non-streaming inference path — the only way to
drive this checkpoint. Microsoft's repo ships only the streaming 0.5B's variant of it.
That fork is where Microsoft restored its own code from in the first place:
`vibevoice/modular/modeling_vibevoice.py` in Microsoft's repo opens with
`# copied from https://github.com/vibevoice-community/VibeVoice/...` and is otherwise
byte-identical to the fork's. So the missing file is Microsoft's own work coming back by
the same road, not a stranger's patch — but the lineage is not an unbroken first-party
line, and MNC-98 is the card that traced it.

The presets, likewise. Microsoft's `demo/voices/` now holds only `streaming_model/*.pt`:
25 prefilled KV caches belonging to the 0.5B, which mean nothing to this model. **This
engine has 9 presets, not 25**, and the difference is not a shortfall — a preset here is
just a reference clip, and the library supplies 221 more.

MNC-98's headline claim, verified against the safetensors index before anything was run:
this checkpoint carries **552 encoder tensors** the 0.5B has zeroed out. That is what
makes cloning possible at all.

## Build / run

```
docker build -f engines/vibevoice-15b/Dockerfile -t tts-vibevoice-15b:gpu .
../tts up vibevoice-15b        # port 8896
```

Weights come from the mounted HF cache (5.4 GB); first run downloads them. After that,
**healthy 10 s** after `up` — no torch.compile, no warmup pass.

## It clones, and here is the proof

Same text, same seed, two references from our own library, pitch measured with
`librosa.yin`:

| | reference | clone |
|---|---|---|
| `kokoro-af-alloy-en` | 140.5 Hz | **141.4 Hz** |
| `kokoro-am-adam-en` | 116.7 Hz | **122.4 Hz** |

The conditioning is real and the two clones are distinct from each other. Whether they
*sound* like the reference is the captain's ear to judge — samples are in
`.bridge-commander/reports/MNC-99-samples/`, each clone next to the clip it came from.

Every voice takes one path: `processor(voice_samples=[[wav]])` → the acoustic encoder →
a voice prompt in the prefill. Presets are not special-cased, which is why
`bench/conform.py` passes the "foreign reference" clause with a 200 instead of the
explained 400 the non-cloning engines owe.

A request that names a `lang` and no `voice` gets the preset for that language if there
is one (en, zh) and otherwise **the first library clip of that language by id** — a
cloning engine with 221 references has no business refusing to pick one. That is what
`bench/measure.py` exercises for pt.

## Knobs

| Param | Range | Default | Notes |
|---|---|---|---|
| `cfg_scale` | 1.0–4.0 | 1.3 | vendor default; above ~2 the voice gets tense |
| `inference_steps` | 1–50 | 10 | vendor default is 10; DDPM steps per acoustic latent |
| `seed` | 0–2³¹−1 | 42 | see below |

`seed` is spent on torch's global RNG immediately before the decode. VibeVoice takes
neither a seed nor a usable generator: the diffusion head draws its initial noise with a
bare `torch.randn` and `dpm_solver` calls `randn_tensor(generator=None)`, so the global
generator *is* the sampler's RNG. Worth knowing — **the vendor's own gradio demo builds a
`torch.Generator`, passes it to `generate()`, and it lands in `**kwargs` and is never
read**; seeding it that way does nothing. Verified here: same text + same knobs + seed 42
twice → byte-identical WAV, seed 7 → a different one.

Two knobs the vendor's UI has that this does not expose: `is_prefill=False` ("disable
voice cloning", which makes the model invent a voice instead of using the reference —
the opposite of the point of this checkpoint) and `refresh_negative`, a CFG internal the
demo pins to True and never varies.

## Contract mapping

- One decode per request, **no text chunking**. 64 k context, long form is what the model
  is sold on, and it windows the text itself.
- Input is reshaped into the processor's podcast-script form (`Speaker 1: ...`) with
  newlines collapsed first. That parser silently **drops** every line that is not a
  speaker cue, so a pasted paragraph would otherwise be spoken down to its first line and
  nothing would say so. `test_script_shape.py` holds that seam against the real parser.
- **An abandoned stream really stops.** `generate()` polls `stop_check_fn` once per
  decoding step, exactly as the 0.5B path does — the hook survived the trip through the
  community fork. `bench/abort-latency.py vibevoice-15b`: **0.11 s** to first audio on the
  request right after an abort, against 0.10 s idle. The abandoned utterance is not
  waited out at all.
- `attn_implementation="sdpa"`. The vendor tested `flash_attention_2`; there is no wheel
  for this base image.
- `bench/conform.py vibevoice-15b`: **CONFORMANT**, no violations.

## Measured

`bench/results/vibevoice-15b.json`, bake-off sentences, RTX 4090, measured beside the
resident whisper service.

| | vibevoice-15b | voxcpm2 | vibevoice (0.5B) |
|---|---|---|---|
| clones? | **yes** | yes | no |
| VRAM, engine process | **6198 MiB** | 7978 MiB | 3214 MiB |
| healthy after `up` | 10 s | — | 10 s |
| realtime, en short / long | 2.84× / 2.92× | 2.73× / 2.72× | 4.66× / 4.52× |
| realtime, pt short / long | 2.80× / 2.86× | 2.75× / 2.71× | 4.28× / 4.62× |
| first PCM byte, en / pt | 100 ms / 60 ms | 40 ms / 30 ms | 40 ms / 40 ms |
| abort frees the engine | yes (0.11 s vs 0.10 s idle) | — | yes (0.19 s vs 0.17 s) |

**It is lighter than voxcpm2 and it clones** — 1.8 GB less VRAM, at the same speed.
MNC-98 estimated ~7 GB; it came in at 6.1.

RAM 2884 MiB idle → 2898 MiB peak. Image 8.58 GB.

## What its Portuguese actually does

`samples/wer.json` — the bake-off sentences back through whisper large-v3-turbo (word
error rate; lower is better, and it scores diction, not charm).

| | en short | en long | pt short | pt long |
|---|---|---|---|---|
| this engine, preset `en-Carter_man` | 0.000 | 0.049 | 0.385 | 0.016 |
| this engine, cloning `kokoro-pm-alex-pt` | | | 0.077 | |
| voxcpm2 | 0.000 | 0.000 | 0.077 | 0.016 |
| vibevoice 0.5B | 0.000 | 0.000 | **0.538** | 0.062 |

(the two right-hand rows are from `engines/vibevoice/README.md`, same sentences, same
whisper)

**The 0.5B's short-Portuguese failure is not inherited — it is a reference problem, and
it is fixable by choosing the reference.** An English preset reading Portuguese mangles
the opening clause (0.385) the same way the 0.5B does. Hand it a Portuguese clip and the
opening clause survives.

How much the clip matters, across the 12 pt references in the library
(`samples/pt-reference-sweep.json`, same sentence, same seed):

| best | median | worst |
|---|---|---|
| **0.000** (`kokoro-pf-dora-pt`, `qwen3-tts-eric-pt`) | 0.115 | 0.385 (`piper-pt-br-jeff-medium-pt`) |

The bad end is piper's own low-bitrate renders; the model reproduces whatever mush it is
given. On a clean pt clip it matches or beats voxcpm2's 0.077 — with the reference's
voice instead of an arbitrary one.

Long-form Portuguese is fine either way (0.016, same as voxcpm2) and holds together for
20-odd seconds in one take with no seams.

## Warts

- **Nine presets, not twenty-five** (above). If you want more built-in voices, they come
  from the library, not from Microsoft.
- Two of the nine presets are `_bgm` — the reference clip has music under the speech, and
  **the clone reproduces the music**. Their display names say so.
- The vendor supports **English and Chinese** on this checkpoint. Everything else in the
  catalogue, Portuguese included, is the model trying; the numbers above are what that is
  worth. `lang` is validated against the catalogue, never passed to the model — the model
  reads whatever the text says in whatever the reference sounds like.
- `transformers` pinned to `4.51.3`, the version the checkpoint's `config.json` was saved
  with.
- Research licence in spirit: MIT weights and code, but the card says "limited to research
  purpose use" and "we do not recommend using VibeVoice in commercial or real-world
  applications". The MIT grant and that sentence are not the same instrument. Same
  unresolved question as 8895, and cloning arbitrary voices sharpens it.
- The model card's claimed audible disclaimer and imperceptible watermark are **not in the
  released code** — the same absence documented for the 0.5B. What comes out of here is
  unmarked audio.
- `VibeVoice-Large` (7B) exists only as an unofficial re-upload —
  `microsoft/VibeVoice-Large` returns 401 — and is 18.7 GB. Out of scope, per MNC-98.
