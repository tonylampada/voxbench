#!/bin/bash
# Portuguese TTS - download WAV file

show_help() {
  echo "Usage: ./tts_pt.sh [OPTIONS] <text>"
  echo ""
  echo "Options:"
  echo "  -o, --output <file>  Output WAV file (default: output.wav)"
  echo "  -v, --voice <name>   Use a specific voice (e.g., pt_male.flac)"
  echo "  -e, --exaggeration <float>  Emotion intensity (default: 0.5, try 0.7+ for expressive)"
  echo "  -c, --cfg <float>    Pacing control (default: 0.5, lower=faster)"
  echo "  -t, --temperature <float>   Sampling randomness (default: 0.8)"
  echo "  -l, --list           List available voices"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  ./tts_pt.sh Olá mundo"
  echo "  ./tts_pt.sh -o saudacao.wav Olá mundo"
  echo "  ./tts_pt.sh -e 0.7 -o expressivo.wav Isso é muito expressivo!"
  echo "  ./tts_pt.sh -v pt_male.flac -o clonado.wav Olá mundo"
}

OUTPUT="output.wav"
VOICE=""
EXAGGERATION=""
CFG_WEIGHT=""
TEMPERATURE=""
TEXT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -v|--voice)
      VOICE="$2"
      shift 2
      ;;
    -e|--exaggeration)
      EXAGGERATION="$2"
      shift 2
      ;;
    -c|--cfg)
      CFG_WEIGHT="$2"
      shift 2
      ;;
    -t|--temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -l|--list)
      curl -s localhost:8877/voices | jq -r '.voices[]'
      exit 0
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      TEXT="$*"
      break
      ;;
  esac
done

if [ -z "$TEXT" ]; then
  echo "Error: No text provided"
  show_help
  exit 1
fi

# Build JSON with jq, adding optional fields
JSON=$(jq -n --arg t "$TEXT" '{text: $t, lang: "pt"}')

if [ -n "$VOICE" ]; then
  JSON=$(echo "$JSON" | jq --arg v "$VOICE" '. + {voice: $v}')
fi

if [ -n "$EXAGGERATION" ]; then
  JSON=$(echo "$JSON" | jq --argjson e "$EXAGGERATION" '. + {exaggeration: $e}')
fi

if [ -n "$CFG_WEIGHT" ]; then
  JSON=$(echo "$JSON" | jq --argjson c "$CFG_WEIGHT" '. + {cfg_weight: $c}')
fi

if [ -n "$TEMPERATURE" ]; then
  JSON=$(echo "$JSON" | jq --argjson t "$TEMPERATURE" '. + {temperature: $t}')
fi

curl -s -X POST localhost:8877/tts \
  -H "Content-Type: application/json" \
  -d "$JSON" \
  -o "$OUTPUT"

echo "Saved: $OUTPUT"
