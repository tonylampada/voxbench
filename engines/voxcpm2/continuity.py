#!/usr/bin/env python3
"""Does one long take stay one person? Measure it, then try the fixes.

Runs INSIDE a voxcpm2 container (it imports voxcpm and needs the GPU), loads the
model once and renders the same passage under every strategy, so the takes differ
by exactly one thing. Writes the wavs and a metrics.json the page is built from.

    docker build -t voxcpm2-lab:gpu -f engines/voxcpm2/Dockerfile.lab .
    docker run --rm --gpus all -v $PWD/voices:/app/voices:ro \
        -v /home/ai/.cache/huggingface:/root/.cache/huggingface \
        -v $PWD/engines/voxcpm2:/lab -v $OUT:/out voxcpm2-lab:gpu \
        python /lab/continuity.py /out

The metric is a speaker embedding (resemblyzer d-vector) per piece. Cosine
similarity between consecutive pieces is the number that matters: it is how
much the same take changes character at a seam. It is only meaningful against
its own scale, so the run also measures two anchors — two takes of the same
text that ARE the same voice (the ceiling) and two different library voices
(the floor).
"""
import json
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from voxcpm import VoxCPM

sys.path.insert(0, str(Path(__file__).parent))
from server import chunk_text, designed  # noqa: E402  the shipped splitter, verbatim

LIB = Path("/app/voices/library")
REF = LIB / "xtts-v2-nova-hogarth-pt.wav"      # a voice the board actually speaks with
OTHER = LIB / "xtts-v2-damien-black-pt.wav"    # a different person, for the floor
# Every library clip reads the same seed sentence per language (bench/mkvoicelib.py),
# but not always faithfully — this one speaks its own full stops out loud ("pontos"),
# an xtts-v2 artifact. So the transcript is what whisper heard in THIS clip, not the
# sentence it was asked to read: a continuation prompt whose text does not match its
# audio teaches the model the wrong alignment.
REF_TEXT = ("O porto estava quieto naquela manhã, e a luz entrava de lado sobre a água, "
            "pontos. Não aconteceu grande coisa, que era exatamente o que todo mundo "
            "esperava, ponto.")

SEED = 42
DESC = "uma mulher jovem, voz clara e animada, falando rápido"
DESC_B = "um homem muito velho, voz rouca e lenta, quase sem fôlego"

# ~3k characters: long enough that the shipped 400-char splitter cuts it into
# eight pieces. Length is the whole point — the drift is cumulative, so a take
# short enough to stay together proves nothing.
PASSAGE = (
    "Capitão, o relatório da noite. A frota rodou limpa: todos os motores subiram "
    "na primeira tentativa e o harness de conformidade não encontrou nada novo. "
    "O parágrafo longo aguentou quarenta e dois segundos sem uma emenda audível, "
    "o que já é mais do que conseguimos na semana passada. "
    "Dois pull requests foram integrados e um terceiro está esperando a sua revisão "
    "desde ontem à tarde. O card que estava travado desde terça voltou a andar; "
    "as frases de referência que faltavam eram um erro de digitação no manifesto, "
    "e não um problema com o modelo, como tínhamos suposto no começo. "
    "A placa está com folga, os discos têm espaço, e o consumo de memória ficou "
    "estável durante toda a madrugada, mesmo com o motor grande carregado. "
    "Houve um susto por volta das três da manhã, quando um cliente desligou no meio "
    "de uma síntese e o servidor demorou meio segundo a mais para responder, mas "
    "nada quebrou e o próximo pedido foi atendido normalmente. "
    "Anotei três coisas que valem a sua atenção quando o senhor acordar. "
    "A primeira é que a biblioteca de vozes cresceu para duzentos e vinte e seis "
    "clipes e o manifesto ainda cabe em um arquivo só. "
    "A segunda é que o console continua servindo a página de testes pelo mesmo "
    "endereço de sempre, e ninguém precisou mexer nele. "
    "A terceira é que ainda não temos uma medida honesta de quanto uma voz muda "
    "ao longo de uma fala comprida, e é exatamente isso que eu fui atrás hoje. "
    "Sobre o motor de voz, tenho uma coisa mais longa para contar, e peço um "
    "minuto porque ela só faz sentido inteira. "
    "O texto que o senhor manda não vira áudio de uma vez: ele é cortado em "
    "pedaços de quatrocentos caracteres, e cada pedaço é uma síntese separada, "
    "que não se lembra do pedaço anterior. "
    "Cada um deles é ancorado no mesmo clipe de referência, o que resolve metade "
    "do problema e deixa a outra metade solta, porque a referência diz quem está "
    "falando e não diz como a frase anterior terminou. "
    "O resultado é que a voz vai escorregando devagar ao longo da fala, e num "
    "parágrafo comprido como este ela chega no fim sendo uma pessoa um pouco "
    "diferente da que começou. "
    "Não é um salto, é um deslize, e é justamente por ser devagar que o ouvido "
    "demora a nomear o que está estranho e mesmo assim recusa o que ouviu. "
    "Medi isso hoje com uma distância de identidade entre pedaços consecutivos, "
    "e o número acompanha exatamente o que se escuta. "
    "Fora isso, nada está pegando fogo. O quadro está parado, a fila está vazia, "
    "e eu fico aqui esperando até o senhor dizer o contrário."
)

MAX_CHUNK_CHARS = 400  # the shipped limit


def say(msg):
    print(f"[continuity] {msg}", flush=True)


# ── the metric ───────────────────────────────────────────────────────────────
encoder = None


def embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """resemblyzer d-vector for one piece of audio."""
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    return encoder.embed_utterance(preprocess_wav(wav, source_sr=16000))


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pitch_hz(wav: np.ndarray, sr: int) -> float:
    """Median F0 over voiced frames — the cheap tell that a voice moved.
    Measured at 16 kHz: yin's cost is per-sample per-lag and nothing under
    400 Hz needs 48."""
    x = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=16000)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else 0.0


def piece_stats(wav: np.ndarray, sr: int, text: str) -> dict:
    dur = len(wav) / sr
    return {
        "seconds": round(dur, 2),
        "pitch_hz": round(pitch_hz(wav, sr), 1),
        "chars_per_s": round(len(text) / dur, 2) if dur else 0.0,
        "rms": round(float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))), 4),
    }


def score(pieces: list[dict]) -> dict:
    """Consecutive-seam similarity plus how far the take gets from its own start."""
    if len(pieces) < 2:
        return {"seam_min": None, "seam_mean": None, "drift_from_first": None,
                "pitch_spread_hz": 0.0}
    embs = [np.array(p["_emb"]) for p in pieces]
    seams = [cos(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
    firsts = [cos(embs[0], e) for e in embs[1:]]
    pitches = [p["pitch_hz"] for p in pieces if p["pitch_hz"] > 0]
    return {
        "seam_min": round(min(seams), 3),
        "seam_mean": round(sum(seams) / len(seams), 3),
        "seams": [round(s, 3) for s in seams],
        "drift_from_first": round(min(firsts), 3),
        "to_first": [round(f, 3) for f in firsts],
        "pitch_spread_hz": round(max(pitches) - min(pitches), 1) if pitches else 0.0,
    }


# ── the strategies ───────────────────────────────────────────────────────────
class Lab:
    def __init__(self, model, out: Path):
        self.m = model
        self.tts = model.tts_model
        self.sr = self.tts.sample_rate
        self.out = out

    def seed(self):
        torch.manual_seed(SEED)

    def _cache(self, ref=None, prompt_wav=None, prompt_text=None):
        return self.tts.build_prompt_cache(prompt_text=prompt_text,
                                           prompt_wav_path=prompt_wav,
                                           reference_wav_path=str(ref) if ref else None)

    def gen(self, text, cache):
        """-> (wav float32, audio_feat). None cache = zero-shot."""
        wav, _, feat = self.tts.generate_with_prompt_cache(
            target_text=text, prompt_cache=cache, cfg_value=2.0, inference_timesteps=10,
            retry_badcase=True)  # what VoxCPM.generate() does, so the takes match production
        return wav.squeeze(0).cpu().numpy(), feat

    def run(self, name, texts, strategy, note):
        """strategy(i, text, state) -> prompt cache for piece i; may read state."""
        say(f"{name}: {len(texts)} piece(s)")
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        state, pieces, wavs = {}, [], []
        t0 = time.time()
        for i, text in enumerate(texts):
            self.seed()
            cache = strategy(i, text, state)
            wav, feat = self.gen(text, cache)
            state["feat"], state["text"], state["wav"] = feat, text, wav
            sf.write(d / f"piece_{i:02d}.wav", wav, self.sr)
            st = piece_stats(wav, self.sr, text)
            st["text"] = text
            st["_emb"] = embed(wav, self.sr).tolist()
            pieces.append(st)
            wavs.append(wav)
        gen_s = time.time() - t0
        full = np.concatenate(wavs)
        sf.write(d / "full.wav", full, self.sr)
        result = {"name": name, "note": note, "pieces": len(texts),
                  "gen_s": round(gen_s, 1),
                  "audio_s": round(len(full) / self.sr, 2),
                  "realtime_x": round((len(full) / self.sr) / gen_s, 2) if gen_s else 0,
                  "wav": f"{name}/full.wav",
                  "stats": [{k: v for k, v in p.items() if k != "_emb"} for p in pieces],
                  **score(pieces)}
        say(f"  seam_min={result['seam_min']} seam_mean={result['seam_mean']} "
            f"{result['realtime_x']}x")
        return result


def main(outdir: str):
    global encoder
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    say("loading resemblyzer")
    encoder = VoiceEncoder()
    say("loading VoxCPM2 (torch.compile warmup takes minutes)")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=True)
    lab = Lab(model, out)
    say(f"sample rate {lab.sr}")

    pieces_400 = chunk_text(PASSAGE, MAX_CHUNK_CHARS)
    say(f"passage {len(PASSAGE)} chars -> {len(pieces_400)} pieces at {MAX_CHUNK_CHARS}")
    ref_cache = lab._cache(ref=REF)
    results = []
    report = {"passage": PASSAGE, "reference": REF.name, "seed": SEED,
              "descriptions": {"A": DESC, "B": DESC_B}, "takes": results}

    def save():
        (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def take(*args):
        """One strategy, rendered and scored, saved before the next one starts.
        A take that dies is recorded and the run goes on — an over-long decode
        can poison the CUDA context, and losing the earlier takes with it would
        mean re-running the whole thing."""
        try:
            results.append(lab.run(*args))
        except Exception as exc:  # noqa: BLE001 — the failure IS a result
            say(f"  FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            results.append({"name": args[0], "note": args[3], "pieces": len(args[1]),
                            "failed": f"{type(exc).__name__}: {str(exc)[:300]}"})
        save()

    # ── anchors: what the metric's own scale is ──────────────────────────────
    # Ceiling: the same reference, two different sentences — same person, so
    # this is as high as an honest seam gets. Floor: two different library
    # voices, which is what "became someone else" scores.
    say("anchors")
    a, b = pieces_400[0], pieces_400[1]
    lab.seed(); wa, _ = lab.gen(a, ref_cache)
    lab.seed(); wb, _ = lab.gen(b, ref_cache)
    lab.seed(); wc, _ = lab.gen(b, lab._cache(ref=OTHER))
    ea, eb, ec = embed(wa, lab.sr), embed(wb, lab.sr), embed(wc, lab.sr)
    sf.write(out / "anchor_same_a.wav", wa, lab.sr)
    sf.write(out / "anchor_same_b.wav", wb, lab.sr)
    sf.write(out / "anchor_other.wav", wc, lab.sr)
    anchors = {"same_voice": round(cos(ea, eb), 3), "different_voice": round(cos(eb, ec), 3)}
    say(f"  same={anchors['same_voice']} different={anchors['different_voice']}")
    report["anchors"] = anchors
    save()

    # ── v0: what ships today ────────────────────────────────────────────────
    take("v0-baseline", pieces_400,
         lambda i, t, s: ref_cache,
         "shipped: every piece conditioned on the reference clip alone, nothing else")

    # ── v1: condition each piece on the previous piece's audio ──────────────
    # merge_prompt_cache is upstream's own "stabilize voice" helper: it keeps the
    # ref prefix and appends the last piece's text and latents as a continuation
    # prompt. Merging into ref_cache (not into the growing cache) carries ONE
    # piece of history, so the prompt stays bounded by the chunk limit.
    def chained(i, t, s):
        return ref_cache if i == 0 else lab.tts.merge_prompt_cache(ref_cache, s["text"], s["feat"])

    take("v1-chained", pieces_400, chained,
         "ref clip + the previous piece's own audio and text as a continuation prompt")

    # ── v1b: chaining with no reference clip at all ─────────────────────────
    def chained_noref(i, t, s):
        if i == 0:
            return ref_cache
        return lab.tts.merge_prompt_cache(None, s["text"], s["feat"])

    take("v1b-chained-noref", pieces_400, chained_noref,
         "piece 1 from the ref clip, every piece after it from the previous piece only")

    # ── v2: ref audio PLUS its transcript, every piece ──────────────────────
    rt_cache = lab._cache(ref=REF, prompt_wav=str(REF), prompt_text=REF_TEXT)
    take("v2-ref-plus-transcript", pieces_400, lambda i, t, s: rt_cache,
         "the unwired mode: ref clip + the same clip as a transcribed continuation prompt")

    # ── v7: both. The transcript pins identity to the clip, the chain smooths
    # the seam; they are prompts of different kinds and nothing stops us using
    # each piece's own audio as the continuation instead of the clip's.
    def both(i, t, s):
        return rt_cache if i == 0 else lab.tts.merge_prompt_cache(ref_cache, s["text"], s["feat"])

    take("v7-transcript-then-chained", pieces_400, both,
         "first piece from ref clip + its transcript, then each piece from the one before")

    # ── v4/v5/v6: the described voice ───────────────────────────────────────
    dlimit = MAX_CHUNK_CHARS - len(DESC) - 2
    plain = chunk_text(PASSAGE, dlimit)
    dpieces = [designed(p, DESC) for p in plain]
    take("v4-description", dpieces, lambda i, t, s: None,
         "shipped: the description prefixed onto every piece, nothing else carried over")

    (out / "v5-description-anchored").mkdir(parents=True, exist_ok=True)
    anchor_wav = out / "v5-description-anchored" / "anchor.wav"

    def anchored(i, t, s):
        """First piece invents the voice; every piece after clones the wav it made."""
        if i == 0:
            return None
        if "ref" not in s:
            sf.write(anchor_wav, s["wav"], lab.sr)
            s["ref"] = lab._cache(ref=anchor_wav)
        return s["ref"]

    take("v5-description-anchored", [dpieces[0]] + plain[1:], anchored,
         "described once, then that take's own wav becomes the reference clip")

    def desc_chained(i, t, s):
        return None if i == 0 else lab.tts.merge_prompt_cache(None, s["text"], s["feat"])

    take("v6-description-chained", [dpieces[0]] + plain[1:], desc_chained,
         "described once, then every piece continues from the previous piece")

    (out / "v8-description-anchored-chained").mkdir(parents=True, exist_ok=True)
    anchor8 = out / "v8-description-anchored-chained" / "anchor.wav"

    def anchored_chained(i, t, s):
        """Both at once: the first take's wav becomes the reference that holds
        identity, and each piece still continues from the one before it."""
        if i == 0:
            return None
        if "ref" not in s:
            sf.write(anchor8, s["wav"], lab.sr)
            s["ref"] = lab._cache(ref=anchor8)
        return lab.tts.merge_prompt_cache(s["ref"], s["text"], s["feat"])

    take("v8-description-anchored-chained", [dpieces[0]] + plain[1:], anchored_chained,
         "described once; that wav becomes the reference AND each piece continues from the last")

    # ── does a description condition at all? ────────────────────────────────
    # Four takes, two descriptions, the shipped seed and one other. If the two
    # descriptions land on the same voice, the description is not steering.
    say("description separation")
    sep = []
    for tag, desc in (("A", DESC), ("B", DESC_B)):
        for s in (SEED, 7):
            torch.manual_seed(s)
            w, _ = lab.gen(designed(pieces_400[0], desc), None)
            f = f"desc_{tag}_seed{s}.wav"
            sf.write(out / f, w, lab.sr)
            sep.append({"desc": desc, "tag": tag, "seed": s, "wav": f,
                        "pitch_hz": round(pitch_hz(w, lab.sr), 1),
                        "_emb": embed(w, lab.sr)})
    vs = [{y["tag"] + str(y["seed"]): round(cos(x["_emb"], y["_emb"]), 3)
           for y in sep if y is not x} for x in sep]
    for x, v in zip(sep, vs):
        x["vs"] = v
        del x["_emb"]
    report["description_separation"] = sep
    save()

    # ── v3: cut fewer pieces. LAST on purpose — a decode this long is the one
    # that can walk off the codebook and poison the CUDA context, and everything
    # above is already saved by then.
    for limit in (800, 1200, len(PASSAGE) + 1):
        p = chunk_text(PASSAGE, limit)
        take(f"v3-chunk{limit}", p, lambda i, t, s: ref_cache,
             f"same shipped strategy, {limit}-character pieces instead of 400")

    say(f"wrote {out}/metrics.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/out")
