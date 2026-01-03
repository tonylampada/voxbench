#!/bin/bash
# Portuguese TTS via server

show_help() {
  echo "Usage: ./run_pt.sh [OPTIONS] <text>"
  echo ""
  echo "Options:"
  echo "  -v, --voice <name>   Use a specific voice (e.g., pt_male.flac)"
  echo "  -l, --list           List available voices"
  echo "  -h, --help           Show this help"
  echo ""
  echo "Examples:"
  echo "  ./run_pt.sh Olá mundo"
  echo "  ./run_pt.sh -v pt_male.flac Olá mundo"
  echo "  ./run_pt.sh --list"
}

VOICE=""
TEXT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--voice)
      VOICE="$2"
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

if [ -n "$VOICE" ]; then
  JSON=$(jq -n --arg t "$TEXT" --arg v "$VOICE" '{text: $t, lang: "pt", voice: $v}')
else
  JSON=$(jq -n --arg t "$TEXT" '{text: $t, lang: "pt"}')
fi

curl -s -X POST localhost:8877/speak \
  -H "Content-Type: application/json" \
  -d "$JSON"
echo
