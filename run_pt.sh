#!/bin/bash
# Portuguese TTS via server

show_help() {
  echo "Usage: ./run_pt.sh [OPTIONS] <text>"
  echo ""
  echo "Options:"
  echo "  -v, --voice <name>   Use a specific voice (e.g., pt_male.flac)"
  echo "  -e, --exaggeration <float>  Emotion intensity (default: 0.5, try 0.7+ for expressive)"
  echo "  -c, --cfg <float>    Pacing control (default: 0.5, lower=faster)"
  echo "  -t, --temperature <float>   Sampling randomness (default: 0.8)"
  echo "  -s, --chunk-size <int>  Sentences per chunk (default: 3, higher=better prosody)"
  echo "  -l, --list           List available voices"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  ./run_pt.sh Olá mundo"
  echo "  ./run_pt.sh -v pt_male.flac Olá mundo"
  echo "  ./run_pt.sh -e 0.7 Isso é muito expressivo!"
  echo "  ./run_pt.sh -e 0.8 -c 0.3 Fala rápida e emocional"
  echo "  ./run_pt.sh -s 5 Texto longo com melhor prosódia"
  echo "  ./run_pt.sh --list"
}

VOICE=""
EXAGGERATION=""
CFG_WEIGHT=""
TEMPERATURE=""
CHUNK_SIZE=""
TEXT=""

while [[ $# -gt 0 ]]; do
  case $1 in
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
    -s|--chunk-size)
      CHUNK_SIZE="$2"
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

if [ -n "$CHUNK_SIZE" ]; then
  JSON=$(echo "$JSON" | jq --argjson s "$CHUNK_SIZE" '. + {chunk_size: $s}')
fi

curl -s -X POST localhost:8877/speak \
  -H "Content-Type: application/json" \
  -d "$JSON"
echo
