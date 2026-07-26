# console — control plane for engines/

Manage the TTS engines and see the machine from the browser.

```
python3 console/server.py        # → http://127.0.0.1:9000  (control plane)
                                 #   http://127.0.0.1:9000/test  (test bench)
```

`/test` is the test bench: a phrase library (seeded with the bench texts,
additions persist in `console/phrases.json`), engines selected side by side,
voices from `/v1/voices` (filter by lang and kind), one control per knob
rendered from `/v1/params` and nothing else, and normal vs streaming synthesis
with time-to-first-audio against total time on screen. Engine refusals (can't
stream, foreign cloning reference, GPU held) are shown verbatim — they are
answers, not failures.

A live stream is scheduled into an AudioContext as it arrives, but it leaves
through a `MediaStreamAudioDestinationNode` into a hidden `<audio>` element
rather than straight out of the context: to the OS that is media playback, so
it survives the screen going off, and the phrase and a stop button appear on
the lock screen. If the element is not allowed to play, the buffers go to the
speakers as before.

`/test?like=<name>` is a temporary probe, and it is here to settle one question:
Bridge Commander's board runs the same pause/resume code as this page and never
comes back from a lock-screen play on the captain's iPhone, while this page
does. Same iOS, same verbs — so the cause is in what surrounds them. Each name
makes this page do what the board does for exactly ONE of the differences, so
they can be walked until one of them breaks pause-then-resume here too:

| name | what it copies from the board |
|---|---|
| `hidden` | nothing player-like on the take while streaming |
| `ctx` | a fresh AudioContext per stream at the engine's rate, built after the response headers, closed when the stream ends |
| `tap` | the element's `play()` and the lock-screen metadata happen after the headers instead of inside the captain's tap |
| `srcobject` | closing drops `srcObject` and the metadata, not just pause |
| `element` | the element is built by script, not written in the page |

Several can be combined (`?like=ctx,tap`); `all` is the board. No query string
is the bench as it has always been. Desktop Chrome resumes every one of them —
only WebKit tells them apart. **This comes out the moment the difference has a
name.**

Runs on the host, outside docker, and drives containers by shelling out to
`engines/tts` — that script stays the only registry of names, ports and
GPU/CPU. `127.0.0.1` is always served, on purpose — local tools depend on it —
and `--host <addr>` ADDS a second listener on that address (same semantics as
`engines/tts --host`); `0.0.0.0` is refused. The engines have no auth and
neither does this: the network boundary is the auth boundary.

Stdlib only, no dependencies, no build step. `CONSOLE_PORT` overrides the port.

## API

```
GET  /api/engines                 [{name, port, gpu, state, healthy, vram_mb, uptime_s, error, detail}]
GET  /api/machine                 {ram, vram, gpu_engine, gpu_other}
GET  /api/phrases                 {phrases: [{id, label, text}]}
POST /api/phrases                 {text} → new phrase, persisted to console/phrases.json
DELETE /api/phrases/<id>
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

If VRAM is well above idle and no registered engine is resident — a container
from another branch, a stray process — `gpu_other` reports how much is taken
and by which pids, and the panel warns instead of claiming the card is free.
