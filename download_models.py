#!/usr/bin/env python3
"""Download and cache Chatterbox models."""
import os
import torch

print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
print(f"HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

# Patch torch.load for CPU
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Downloading English (Turbo) model...")
ChatterboxTurboTTS.from_pretrained(device="cpu")
print("English model downloaded!")

print("Downloading Portuguese (Multilingual) model...")
ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("Portuguese model downloaded!")

print("\nAll models cached successfully!")

# Verify cache
import subprocess
hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f"\nCache contents at {hf_home}:")
subprocess.run(['ls', '-la', hf_home], check=False)
