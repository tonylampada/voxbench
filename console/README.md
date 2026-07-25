# console — control plane for engines/

Manage the TTS engines and see the machine from the browser.

```
python3 console/server.py        # → http://127.0.0.1:8880
```

Runs on the host, outside docker, and drives containers by shelling out to
`engines/tts` — that script stays the only registry of names, ports and
GPU/CPU. Loopback only, on purpose: the engines have no auth and neither does
this; the network boundary is the auth boundary.

Stdlib only, no dependencies, no build step. `CONSOLE_PORT` overrides the port.

## API

```
GET  /api/engines                 [{name, port, gpu, state, healthy, vram_mb, uptime_s, error, detail}]
GET  /api/machine                 {ram, vram, gpu_engine}
POST /api/engines/<name>/up       202 loading | 409 with the tts script's refusal
POST /api/engines/<name>/down
GET  /api/engines/<name>/voices   proxies GET /v1/voices
GET  /api/engines/<name>/params   proxies GET /v1/params
POST /api/engines/<name>/speech   proxies POST /v1/audio/speech (streaming too)
```

`state` is `stopped`, `loading` or `ready`. One 24 GB card fits one big model:
starting a second GPU engine returns the refusal from `engines/tts up`
verbatim, naming the engine that holds the card. VRAM is read with host
`nvidia-smi` pointed at the driver-matched libs the tts script already mounts
into containers (plain `nvidia-smi` fails until the reboot aligns driver and
libs); if no route works the panel says "unavailable" and why, never a fake
number.
