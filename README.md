# voxbench

Fourteen text-to-speech engines, each in its own container, all answering the **same
HTTP API**. Switching engines is switching a port.

It exists to answer a question no benchmark table can: *which of these actually sounds
right for Brazilian Portuguese?* So it ships a shared voice library, a conformance
harness, per-engine measurements, and a browser console where you load an engine, type a
sentence and listen.

## What you need

- Docker.
- An NVIDIA GPU for most engines — one 24 GB card fits one big model at a time.
  `piper` is CPU-only, and `chatterbox-nano` can run on CPU too.
- `python3` for the console. No dependencies, no build step, nothing to install.

## Your first sixty seconds

```bash
git clone git@github.com:tonylampada/voxbench.git
cd voxbench

engines/tts up piper                                   # builds the image, starts it, waits until healthy
engines/tts say piper "O deploy está verde, capitão." --lang pt --out /tmp/ok.wav
```

`piper` is CPU-only and small, so this works on any machine and proves the install
without touching the GPU. Every other engine is the same two commands with a different
name — and the first `up` of a big one downloads weights, so give it minutes.

## Then open the console

```bash
python3 console/server.py     # → http://127.0.0.1:9000
```

After this point you do not need the terminal:

- **`/`** — every engine, its state, and the machine. Load and unload from the browser;
  watch RAM and VRAM; see who is holding the card. Starting a second GPU engine is
  refused, and the refusal names the one that has to come down first.
- **`/test`** — the bench. A phrase library you add to, the voices each engine offers,
  one control per knob that engine actually has, and two buttons: generate, or stream.
  Stream plays as it arrives — the point is to *feel* the latency, not read it. Each
  take lands in a session tape with time-to-first-sound against total time.

The voices are already there: `voices/library/` holds 221 rendered clips with a manifest
carrying language, gender and duration, and every engine serves the whole catalogue at
`GET /v1/voices`.

## The engines

Realtime factor and first-sound latency are measured — English, short text, warm model.
Raw numbers in `bench/results/*.json`, reproducible with `bench/measure.py`.

| Engine | Port | Speaks | Realtime | First sound | Watch out |
|---|---|---|---|---|---|
| chatterbox-ptbr | 8881 | en pt | 7.5× | — | Watermarks every output |
| qwen3-tts | 8882 | en pt | 1.5× | — | No native pt-BR speaker |
| voxcpm2 | 8883 | en pt | 2.7× | **40 ms** | Minutes of warmup at start |
| f5-tts-ptbr | 8884 | en pt | 12.6× | — | Weights non-commercial |
| xtts-v2 | 8885 | en pt | 7.8× | — | Non-commercial licence |
| gepard | 8886 | en pt | 4.2× | — | pt comes out English-accented |
| omnivoice-brpt | 8887 | en pt | 12.6× | — | Best measured pt-BR; licence conflicted |
| piper | 8888 | en pt | 37× (CPU) | **50 ms** | No cloning; GPL engine |
| kokoro | 8889 | en pt | 83× | **90 ms** | pt is pan-lusophone, not Brazilian |
| chatterbox-flash | 8890 | en | 5.0× | — | Repeats itself past 30 s |
| chatterbox-nano | 8891 | en | 12.5× | — | Clones *and* runs on CPU (1.8×) |
| fish-s2-pro | 8892 | en pt | 0.8× | — | Slower than realtime |
| higgs-tts-3 | 8893 | en pt | 3.9× | 1.71 s | Non-commercial |
| voxtral-tts | 8894 | en pt | 5.0× | **130 ms** | Cannot clone; non-commercial |

Five engines stream for real. The rest answer `stream: true` with an honest 400 saying
why — none of them fakes it by cutting the text into sentences.

Licences, version pins, warts and per-engine measurements live in
`engines/<name>/README.md`.

## Using an engine from your own code

**`docs/agent.md` is written to be handed to a coding agent.** Drop it in the context,
say which port is up, and it will consume the API without reading anything else here. It
is self-contained on purpose.

For humans, the same surface in one breath:

```bash
curl -s localhost:8888/v1/voices                       # what can I say it with
curl -s localhost:8888/v1/params                       # what knobs does this one have
curl -s localhost:8888/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input": "Olá, capitão.", "lang": "pt"}' -o out.wav
```

The full contract — every field, every rule, every case that must return 400 — is
`docs/contract.md`. It is the single source of truth, and `bench/conform.py <engine>`
checks a live server against all of its clauses.

## Repo map

```
engines/          one directory per engine: Dockerfile, adapter, README with its warts
engines/tts       the only way to build, start, stop and talk to an engine
console/          the control plane and the test bench (host-side, stdlib, no deps)
voices/library/   221 reference clips + manifest.json (language, gender, duration)
bench/conform.py  judges a live engine against docs/contract.md
bench/measure.py  produces the measurements in bench/results/
bench/abort-latency.py  proves a hung-up stream frees the engine, over real HTTP
docs/contract.md  the API, normatively
docs/agent.md     the API, packaged for an agent's context
```

## A note on exposure

Every engine and the console publish on `127.0.0.1` and have **no authentication of any
kind** — and the console starts and stops containers. The network boundary is the auth
boundary.

To also reach them from another device on your tailnet, name that address —
`127.0.0.1` keeps working either way, both add a listener rather than move one:

```bash
engines/tts up piper --host 100.x.y.z
python3 console/server.py --host 100.x.y.z
```

`0.0.0.0` is refused outright, by both.
