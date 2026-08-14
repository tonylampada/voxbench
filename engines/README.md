# TTS engine fleet

Fifteen TTS engines, each in its own container, all speaking the **same HTTP
contract** — so switching engines is switching a port. They exist to benchmark
pt-BR (and en) synthesis against each other; each `engines/<name>/README.md`
carries that engine's pins, measurements and warts.

Run them with the `engines/tts` script. Nobody should ever type a `docker run`
for these by hand.

## The script

```
engines/tts up <engine>        # build image if missing, start, wait until healthy
engines/tts up chatterbox-nano --cpu   # nano only: run the model on CPU (no GPU flags)
engines/tts down <engine>      # stop it
engines/tts ls                 # dashboard: port / image / container / health
engines/tts logs <engine> [-f] # tail its logs
engines/tts say <engine> "text" [--lang en|pt] [--voice V] [--out FILE]
engines/tts voices <engine>    # list its voices
```

Worked example (paste this):

```
engines/tts up piper
engines/tts say piper "O deploy está verde, capitão." --lang pt --out /tmp/ok.wav
ffprobe /tmp/ok.wav        # real WAV, ~2 s
engines/tts down piper
```

To hear an engine in a browser, run the console (`python3 console/server.py`) and
open its `/test` bench — engines serve no HTML of their own.

`up` needs nothing but the name — port, build context, GPU flags and mounts all
live in the registry table at the top of the script (adding an engine is one
line there). `up` polls `/health` with a heartbeat until the model is loaded; a
first run may also download weights into the shared HF cache, so give it
minutes, not seconds (`TTS_UP_TIMEOUT` to extend). It refuses to start a second
GPU engine while one is running — there is one 24 GB card — unless you pass
`--force`.

Everything publishes on `127.0.0.1`, always — local tools depend on it. To
also reach an engine from another tailnet device — the
captain's phone — `--host` adds a second publish on this box's tailnet
address, same port:

```
engines/tts up piper --host 100.114.197.65   # or: export TTS_HOST=100.114.197.65
```

`0.0.0.0` is refused outright: these engines have no authentication of any
kind, so the network boundary is the auth boundary. A tailnet address is a
bounded, deliberate choice; `0.0.0.0` would put an unauthenticated TTS service
on every interface, including whatever public wifi the laptop is on.

## The contract

Every engine serves, on container port 8877 (published on its host port below):

| Endpoint | Method | Behaviour |
|---|---|---|
| `/health` | GET | 200 `{"status": "ok", "engine": "<name>"}` once the model is loaded; 503 while loading |
| `/v1/voices` | GET | the voice catalogue, built from `voices/library/manifest.json` |
| `/v1/params` | GET | this engine's tunable knobs, with type, range and a description each |
| `/v1/audio/speech` | POST | synthesize — WAV body, or raw PCM when `stream: true` |

Anything else is 404, including the whole v1 surface (`/tts`, `/tts/stream`,
`GET /voices`) — it was deleted, not deprecated.

Request body:

```json
{"input": "...", "voice": "af_bella", "lang": "pt",
 "response_format": "wav", "speed": 1.0, "stream": false, "params": {}}
```

Only `input` is required. `voice` is an **opaque id** from `/v1/voices`, never a
file path; absent → the engine's default for `lang`. `params` carries the
engine's own knobs and every key must appear in `/v1/params`. Streaming is raw
signed 16-bit LE mono PCM with the rate in `x-sample-rate` and no framing at
all; an engine that cannot stream says so with a 400 instead of faking it by
splitting sentences.

Every client error is 400 with `{"detail": "<human sentence>"}` — no 422s.

The normative version of all this, clause by clause, is `docs/contract.md`;
`bench/conform.py <engine>` checks a live server against it. For handing to a
coding agent that just needs to consume an engine, `docs/agent.md` is
self-contained.

## The engines

Verdicts are from the bake-off measurements recorded in each engine's README.

| Engine | Port | Licence | Verdict |
|---|---|---|---|
| chatterbox-ptbr | 8881 | MIT | Dedicated pt-BR finetune, reported CER 0.35%; embeds a Perth watermark in every output |
| qwen3-tts | 8882 | Apache-2.0 | No native Portuguese speaker; single "Portuguese" id risks pt-PT phonemes; ~0.7× realtime |
| voxcpm2 | 8883 | Apache-2.0 | 48 kHz output, first sound in 40 ms — the quickest streamer here; ~6 GiB VRAM and minutes of torch.compile warmup at start |
| f5-tts-ptbr | 8884 | weights CC-BY-NC-4.0 | pt-BR quality yardstick only — non-commercial weights, cannot ship |
| xtts-v2 | 8885 | CPML (non-commercial) | The frozen 2023 control; quality bar to judge the rest, nothing more |
| gepard | 8886 | Apache-2.0 | Small (555M) and quick, but no pt-BR reference voice — pt comes out English-accented |
| omnivoice-brpt | 8887 | conflicted (Apache tag on CC-BY-NC base) | Best measured pt-BR: WER 0.040 / CER 0.038; research-only until the licence conflict resolves |
| piper | 8888 | GPL-3.0 engine, MIT voices | CPU-only and instant; no cloning; GPL applies if embedded in a product |
| kokoro | 8889 | Apache-2.0 | 82M, ~100× realtime; Portuguese is pan-lusophone "D" grade, not Brazilian |
| chatterbox-flash | 8890 | MIT | **English-only** (`lang: "pt"` → 400); marketed ~13× realtime, measured 5.0×; cannot stream; repeats itself past a ~30 s context window |
| chatterbox-nano | 8891 | MIT | **English-only** (`lang: "pt"` → 400); 110M, 12× realtime on GPU and the only engine that both clones voices and runs acceptably on CPU (~1.8×, `up chatterbox-nano --cpu`) |
| fish-s2-pro | 8892 | Apache-2.0 code, weights non-commercial | Slower than realtime (0.8×) on this card; cannot stream; runs as uid 1000, so it reads the HF cache from a separate `/hf` mount |
| higgs-tts-3 | 8893 | Boson Research & Non-Commercial | 4B multilingual (102 languages, en and pt both top tier); served by SGLang-Omni, native PCM streaming; see `higgs-tts-3/README.md` for measurements |
| voxtral-tts | 8894 | CC-BY-NC-4.0 (weights inherit the preset voices' licence) | Mistral's 4B, 9 languages, 20 presets; first streamed PCM at 130 ms AND ~5× realtime — quick at both ends, unlike VoxCPM2; public checkpoint cannot clone (no encoder weights) |
| vibevoice | 8895 | MIT (card says research-use only) | Microsoft's 0.5B realtime; 40 ms to first PCM byte AND 4.5× realtime, the quickest at both ends; 3.2 GiB; cannot clone (encoder withheld); pt is vendor-experimental and mangles the opening clause of short utterances |
| vibevoice-15b | 8896 | MIT (card says research-use only) | the 0.5B's big sibling and the one that **clones** — 6.1 GiB, 1.8 GB lighter than voxcpm2 at the same 2.9× realtime; 100 ms to first PCM byte; an aborted stream really stops; 9 presets and every library clip; pt quality tracks the reference clip you give it (0.000 WER on a clean one) |

Every number above is measured, not quoted: `bench/results/*.json`, produced by
`bench/measure.py`.

## The GPU workaround (pre-reboot)

This box's NVIDIA kernel module (580.159.03) and host userspace libs
(580.173.02) are mismatched until the next reboot, so `--gpus all` fails:
nvidia-container-toolkit tries to mount `/run/nvidia-persistenced/socket`,
which does not exist. The script instead passes the device nodes explicitly and
mounts driver libs that match the kernel module (see `gpu_run_args` in
`engines/tts` — the **only** place the recipe lives).

After the reboot the versions align and plain `--gpus all` works again:

```
TTS_GPU_MODE=gpus-all engines/tts up <engine>
```

…and once that's confirmed, flip the default inside `gpu_run_args` and delete
the workaround arm.
