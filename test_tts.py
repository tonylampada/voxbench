#!/usr/bin/env python3
"""Quick test script for Chatterbox TTS models."""
import sys
from dotenv import load_dotenv
load_dotenv()

import torch

def get_device():
    # MPS (Mac) may have issues with some operations, try CPU first
    # Uncomment below to try GPU acceleration
    # if torch.backends.mps.is_available():
    #     return "mps"
    # elif torch.cuda.is_available():
    #     return "cuda"
    return "cpu"

def test_english():
    print("="*50)
    print("Testing English TTS (Turbo model)...")
    print("="*50)

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    device = get_device()
    print(f"Using device: {device}")

    model = ChatterboxTurboTTS.from_pretrained(device=device)
    print("Model loaded!")

    text = "Hello, this is a test of the Chatterbox text to speech system."
    print(f"Generating: '{text}'")

    wav = model.generate(text)
    print(f"Generated audio shape: {wav.shape}")
    print(f"Sample rate: {getattr(model, 'sr', 24000)}")

    # Save to file
    import torchaudio
    sample_rate = getattr(model, 'sr', 24000)
    torchaudio.save("test_english.wav", wav.cpu(), sample_rate)
    print("Saved to test_english.wav")
    print("English TTS: SUCCESS!\n")
    return True

def test_portuguese():
    print("="*50)
    print("Testing Portuguese TTS (Multilingual model)...")
    print("="*50)

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = get_device()
    print(f"Using device: {device}")

    # Patch torch.load to handle CPU loading of CUDA checkpoints
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = device
        return original_load(*args, **kwargs)
    torch.load = patched_load

    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    finally:
        torch.load = original_load
    print("Model loaded!")

    text = "Olá, este é um teste do sistema de síntese de fala."
    print(f"Generating: '{text}'")

    wav = model.generate(text, language_id="pt")
    print(f"Generated audio shape: {wav.shape}")
    print(f"Sample rate: {getattr(model, 'sr', 24000)}")

    # Save to file
    import torchaudio
    sample_rate = getattr(model, 'sr', 24000)
    torchaudio.save("test_portuguese.wav", wav.cpu(), sample_rate)
    print("Saved to test_portuguese.wav")
    print("Portuguese TTS: SUCCESS!\n")
    return True

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"

    if arg in ("en", "english", "both"):
        test_english()

    if arg in ("pt", "portuguese", "both"):
        test_portuguese()

    print("All tests passed!")
