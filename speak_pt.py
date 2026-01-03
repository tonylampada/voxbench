#!/usr/bin/env python3
"""Portuguese TTS - pass text as argument, optionally clone a voice."""
import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

import torch
import sounddevice as sd
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

parser = argparse.ArgumentParser(description="Portuguese TTS with Chatterbox")
parser.add_argument("text", nargs="+", help="Text to speak")
parser.add_argument("-v", "--voice", help="Reference audio file for voice cloning (~10s)")
args = parser.parse_args()

text = " ".join(args.text)

# Patch torch.load for CPU loading
original_load = torch.load
torch.load = lambda *a, **kw: original_load(*a, map_location="cpu", **{k: v for k, v in kw.items() if k != 'map_location'})

model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
torch.load = original_load

wav = model.generate(text, audio_prompt_path=args.voice, language_id="pt")
sd.play(wav.cpu().numpy().squeeze(), getattr(model, 'sr', 24000))
sd.wait()
