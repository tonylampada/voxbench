#!/usr/bin/env python3
"""English TTS - pass text as argument, optionally clone a voice."""
import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

import torch
import sounddevice as sd
from chatterbox.tts_turbo import ChatterboxTurboTTS

parser = argparse.ArgumentParser(description="English TTS with Chatterbox")
parser.add_argument("text", nargs="+", help="Text to speak")
parser.add_argument("-v", "--voice", help="Reference audio file for voice cloning (~10s)")
args = parser.parse_args()

text = " ".join(args.text)

model = ChatterboxTurboTTS.from_pretrained(device="cpu")
wav = model.generate(text, audio_prompt_path=args.voice)
sd.play(wav.cpu().numpy().squeeze(), getattr(model, 'sr', 24000))
sd.wait()
