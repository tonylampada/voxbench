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
answers, not failures. Every take names the voice it used: there is no "engine
default" entry, because a take that did not choose its voice cannot be compared
with one that did. The speaking itself is `speech.js`, below.

## speech.js — the speech path

`console/speech.js` is the whole speech path as one dependency-free ES module:
the request to the engine, the streaming PCM read, the route to the speakers,
the lock screen and the three verbs. The bench imports it; so does anything else
that wants a message spoken out loud on a phone.

```js
import * as speech from './speech.js';

const out = await speech.speak({
  url: '/api/engines/voxcpm2/speech',   // required — no default endpoint
  voice: 'kokoro-pf-dora-pt',           // required — no default voice
  input: 'Capitão, o deploy passou.',
  params: {},                           // engine knobs, optional
  title: 'deploy', artist: 'voxcpm2',   // what the lock screen says
  onFirstSound: ms => console.log('first sound', ms),
});
// out: {blob, bytes, rate, ttfs, stopped} — the same audio, replayable
speech.pause(); speech.resume(); speech.stop();
```

Call `speak()` from inside the tap that asked for it: everything iOS insists on
seeing in a gesture happens before the first await. What the module knows, in
the order it learned it:

- The sound leaves through an `<audio>` element fed by a
  `MediaStreamAudioDestinationNode`, not straight out of the AudioContext. To
  the OS an element playing is media playback and survives the screen locking; a
  bare page is suspended, and the AudioContext with it.
- A **visible** transport — the phrase, play, pause, stop — is on screen for as
  long as the speech runs, and gone when it is not. WebKit hands the lock screen
  to a player it can see and takes it back from one it cannot; this is the
  finding an afternoon of probing bought, and it belongs to the module, not to a
  page. Its three buttons are wired to the same three functions as the lock
  screen's, so pressing them on screen and on the lock screen are one act.
- `pause` pauses the element first and then freezes the context clock (pull on a
  MediaStream with nothing new in it and the last fragment is chewed over and
  over); `resume` goes back inside out. `stop` aborts the request so the ENGINE
  stops synthesizing, and resolves the `speak()` promise with what was heard.
- Both the endpoint and the voice are arguments and neither has a default. An
  empty voice means "whatever the engine feels like", which is how a page ends
  up speaking English in a Portuguese room without anyone choosing it.
- When the engine refuses, `speak()` rejects with the engine's own words and
  `.status`. The module never quietly finds another way to speak — a caller that
  hears nothing is told why.

If the element is not allowed to play at all, the buffers go to the speakers:
sound without the lock screen, never silence.

The `/test?like=<name>` probes are gone. They existed to name why Bridge
Commander's board behaved differently from this page on the captain's iPhone
(`hidden`, `ctx`, `tap`, `srcobject`, `element`, `anchor`, `loud`, `ghost`,
`mini`). The difference has its name — a player WebKit can see — and it is now
this module's ordinary behaviour, so the questions have no one left to ask.

Runs on the host, outside docker, and drives containers by shelling out to
`engines/tts` — that script stays the only registry of names, ports and
GPU/CPU. `127.0.0.1` is always served, on purpose — local tools depend on it —
and `--host <addr>` ADDS a second listener on that address (same semantics as
`engines/tts --host`); `0.0.0.0` is refused. That address is handed down to the
engines this console starts (as `TTS_HOST`), so an engine brought up from the
page is published exactly where the console itself is reachable — not loopback
only. The engines have no auth and neither does this: the network boundary is
the auth boundary.

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
