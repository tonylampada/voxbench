# Chatterbox TTS Server

A FastAPI server wrapping [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech models with support for voice cloning, emotion control, and paralinguistic tags.

## Features

- **Two models**: English (Turbo) and Portuguese (Multilingual)
- **Voice cloning**: Clone any voice with 5+ seconds of reference audio
- **Emotion control**: Adjust expressiveness with `exaggeration` parameter
- **Paralinguistic tags**: Add `[laugh]`, `[sigh]`, `[whisper]` and more (English only)
- **Streaming**: Generate long text in chunks for faster time-to-first-audio
- **Local playback**: Play audio directly on the server machine

## Quick Start

```bash
# Install dependencies
uv sync

# Run server
uv run python server.py

# Test it
./run_en.sh "Hello world!"
```

## Endpoints

### POST /speak
Generate and play audio locally on the server.

```bash
curl -X POST localhost:8877/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "lang": "en"}'
```

### POST /tts
Generate and return WAV audio.

```bash
curl -X POST localhost:8877/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "lang": "en"}' \
  -o speech.wav
```

### GET /tts
Same as POST but with query parameters.

```bash
curl "localhost:8877/tts?text=Hello%20world&lang=en" -o speech.wav
```

### POST /tts/stream
Stream audio in chunks (one per sentence group).

### GET /voices
List available voice files for cloning.

### GET /health
Health check and model status.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `lang` | string | "en" | Language: "en" or "pt" |
| `voice` | string | null | Voice file for cloning (e.g., "de_female.flac") |
| `exaggeration` | float | 0.0 (en) / 0.5 (pt) | Emotion intensity (0.25-2.0, neutral=0.5) |
| `cfg_weight` | float | 0.0 (en) / 0.5 (pt) | Pacing control (0.2-1.0, lower=faster) |
| `temperature` | float | 0.8 | Sampling randomness (0.05-2.0) |
| `chunk_size` | int | 3 | Sentences per chunk (higher=better prosody) |
| `split_sentences` | bool | true | Whether to split text into chunks |

### Parameter Tips

- **Expressive speech**: `exaggeration=0.7`, `cfg_weight=0.3`
- **Neutral/professional**: `exaggeration=0.5`, `cfg_weight=0.5`
- **Fast speaker reference**: Lower `cfg_weight` to ~0.3
- **Better prosody**: Increase `chunk_size` to 4-5

## Paralinguistic Tags (English Only)

Add these tags in your text for sound effects and emotions:

### Sound Effects
| Tag | Description |
|-----|-------------|
| `[laugh]` | Laughter |
| `[chuckle]` | Light chuckle |
| `[cough]` | Coughing |
| `[clear throat]` | Throat clearing |
| `[gasp]` | Gasping |
| `[groan]` | Groaning |
| `[sigh]` | Sighing |
| `[sniff]` | Sniffing |
| `[shush]` | Shushing |
| `[crying]` | Crying |

### Emotions/Styles
| Tag | Description |
|-----|-------------|
| `[angry]` | Angry tone |
| `[happy]` | Happy tone |
| `[fear]` | Fearful tone |
| `[surprised]` | Surprised tone |
| `[sarcastic]` | Sarcastic tone |
| `[dramatic]` | Dramatic delivery |
| `[whispering]` | Whispered speech |
| `[narration]` | Narrator style |
| `[advertisement]` | Ad-read style |

### Example
```bash
./run_en.sh "Oh wow [gasp], I can't believe it! [laugh] This is amazing."
```

## Shell Scripts

### run_en.sh / run_pt.sh
Play audio locally.

```bash
./run_en.sh "Hello world"
./run_en.sh -e 0.7 "Very expressive!"
./run_en.sh -v de_female.flac "Clone this voice"
./run_en.sh -s 5 "Long text with better prosody across many sentences."
./run_en.sh --help
```

### tts_en.sh / tts_pt.sh
Download WAV files.

```bash
./tts_en.sh "Hello world"                    # → output.wav
./tts_en.sh -o speech.wav "Hello world"      # → speech.wav
./tts_en.sh -e 0.7 -o expressive.wav "Wow!"  # with emotion
```

### Options
| Flag | Description |
|------|-------------|
| `-v, --voice <name>` | Voice file for cloning |
| `-e, --exaggeration <float>` | Emotion intensity |
| `-c, --cfg <float>` | Pacing control |
| `-t, --temperature <float>` | Sampling randomness |
| `-s, --chunk-size <int>` | Sentences per chunk |
| `-o, --output <file>` | Output file (tts_*.sh only) |
| `-l, --list` | List available voices |
| `-h, --help` | Show help |

## Voice Cloning

Place audio files (`.flac`, `.wav`, `.mp3`) in the `voices/` directory. Requirements:
- **Minimum 5 seconds** of clear speech
- Single speaker, minimal background noise

```bash
# List available voices
./run_en.sh --list

# Use a voice
./run_en.sh -v de_female.flac "Hello in a German female voice"
```

## Docker

### Build
```bash
# Requires HuggingFace token in .env file
echo "HF_TOKEN=your_token_here" > .env
./build_docker.sh
```

### Run
```bash
docker run -p 8877:8877 chatterbox-tts:latest
```

The Docker image includes pre-downloaded models and runs fully offline.

## Testing

```bash
# Run test suite
./test_server.sh

# Run with audio playback
./test_server.sh --play
```

## Demo

```bash
# Demo all paralinguistic tags
./demo_tags.sh
```

## API Examples

### Python
```python
import requests

# Generate speech
response = requests.post("http://localhost:8877/tts", json={
    "text": "Hello world!",
    "lang": "en",
    "exaggeration": 0.7
})

with open("speech.wav", "wb") as f:
    f.write(response.content)
```

### JavaScript
```javascript
const response = await fetch("http://localhost:8877/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        text: "Hello world!",
        lang: "en"
    })
});

const blob = await response.blob();
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

## Models

| Model | Languages | Parameters | Features |
|-------|-----------|------------|----------|
| Chatterbox Turbo | English | 350M | Paralinguistic tags, fast |
| Chatterbox Multilingual | 23+ languages | 500M | Emotion exaggeration |

## License

Based on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox).
