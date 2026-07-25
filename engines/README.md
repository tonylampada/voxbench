# TTS engine fleet

Nine TTS engines, each in its own container, all speaking the **same HTTP
contract** — so switching engines is switching a port. They exist to benchmark
pt-BR (and en) synthesis against each other; each `engines/<name>/README.md`
carries that engine's pins, measurements and warts.

Run them with the `engines/tts` script. Nobody should ever type a `docker run`
for these by hand.

## The script

```
engines/tts up <engine>        # build image if missing, start, wait until healthy
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

`up` needs nothing but the name — port, build context, GPU flags and mounts all
live in the registry table at the top of the script (adding an engine is one
line there). `up` polls `/health` with a heartbeat until the model is loaded; a
first run may also download weights into the shared HF cache, so give it
minutes, not seconds (`TTS_UP_TIMEOUT` to extend). It refuses to start a second
GPU engine while one is running — there is one 24 GB card — unless you pass
`--force`. Everything publishes on `127.0.0.1` only.

## The contract

Every engine serves, on container port 8877 (published on its host port below):

| Endpoint | Method | Behaviour |
|---|---|---|
| `/health` | GET | 200 `{"status": "ok", ...}` once the model is loaded; 503 (or no answer) while loading |
| `/voices` | GET | `{"voices": [...]}` — names accepted in the `voice` field |
| `/tts` | POST | WAV binary (mono 16-bit; sample rate per engine, see its README) |
| `/tts/stream` | POST | framed stream: repeated `[4-byte big-endian length][complete WAV]` |

Request body for `/tts` and `/tts/stream`:

```json
{"text": "...", "lang": "en" | "pt", "voice": "optional-name", "chunk_size": 3}
```

`voice` null/absent → the engine's default voice. `chunk_size` (stream only) =
sentences per frame. Unsupported `lang` → 400. Some engines accept extra
optional knobs (temperature, speed, …) — per-engine READMEs list them. Note:
for most engines `/tts/stream` is sentence-chunked pseudo-streaming, not native
(voxcpm2 is the exception).

## The engines

Verdicts are from the bake-off measurements recorded in each engine's README.

| Engine | Port | Licence | Verdict |
|---|---|---|---|
| chatterbox-ptbr | 8881 | MIT | Dedicated pt-BR finetune, reported CER 0.35%; embeds a Perth watermark in every output |
| qwen3-tts | 8882 | Apache-2.0 | No native Portuguese speaker; single "Portuguese" id risks pt-PT phonemes; ~0.7× realtime |
| voxcpm2 | 8883 | Apache-2.0 | 48 kHz output, the only native streamer; ~6 GiB VRAM and minutes of torch.compile warmup at start |
| f5-tts-ptbr | 8884 | weights CC-BY-NC-4.0 | pt-BR quality yardstick only — non-commercial weights, cannot ship |
| xtts-v2 | 8885 | CPML (non-commercial) | The frozen 2023 control; quality bar to judge the rest, nothing more |
| gepard | 8886 | Apache-2.0 | Small (555M) and quick, but no pt-BR reference voice — pt comes out English-accented |
| omnivoice-brpt | 8887 | conflicted (Apache tag on CC-BY-NC base) | Best measured pt-BR: WER 0.040 / CER 0.038; research-only until the licence conflict resolves |
| piper | 8888 | GPL-3.0 engine, MIT voices | CPU-only and instant; no cloning; GPL applies if embedded in a product |
| kokoro | 8889 | Apache-2.0 | 82M, ~100× realtime; Portuguese is pan-lusophone "D" grade, not Brazilian |

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
