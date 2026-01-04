#!/bin/bash
# Test script for Chatterbox TTS Server
# Works with both Docker and local (undocker) versions

set -e

BASE_URL="${TTS_URL:-http://localhost:8877}"
TEMP_DIR=$(mktemp -d)
PASSED=0
FAILED=0
WARNINGS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

log_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAILED=$((FAILED + 1))
}

log_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
    WARNINGS=$((WARNINGS + 1))
}

log_info() {
    echo -e "  INFO: $1"
}

# Check if server is running
echo "=============================================="
echo "Chatterbox TTS Server Test Suite"
echo "=============================================="
echo "Target: $BASE_URL"
echo ""

echo "--- Connectivity ---"
if curl -s --connect-timeout 5 "$BASE_URL/health" > /dev/null 2>&1; then
    log_pass "Server is reachable"
else
    log_fail "Server is not reachable at $BASE_URL"
    echo ""
    echo "Make sure the server is running:"
    echo "  Docker:  docker run -p 8877:8877 chatterbox-tts:latest"
    echo "  Local:   python server.py"
    exit 1
fi

# Test health endpoint
echo ""
echo "--- Health Endpoint ---"
HEALTH=$(curl -s "$BASE_URL/health")
if echo "$HEALTH" | jq -e '.status == "ok"' > /dev/null 2>&1; then
    log_pass "GET /health returns status ok"
else
    log_fail "GET /health did not return status ok"
fi

# Check models loaded
EN_LOADED=$(echo "$HEALTH" | jq -e '.models_loaded | contains(["en"])' 2>/dev/null || echo "false")
PT_LOADED=$(echo "$HEALTH" | jq -e '.models_loaded | contains(["pt"])' 2>/dev/null || echo "false")

if [ "$EN_LOADED" = "true" ]; then
    log_pass "English model loaded"
else
    log_fail "English model not loaded"
fi

if [ "$PT_LOADED" = "true" ]; then
    log_pass "Portuguese model loaded"
else
    log_fail "Portuguese model not loaded"
fi

# Test voices endpoint
echo ""
echo "--- Voices Endpoint ---"
VOICES=$(curl -s "$BASE_URL/voices")
VOICE_COUNT=$(echo "$VOICES" | jq '.voices | length' 2>/dev/null || echo "0")

if [ "$VOICE_COUNT" -gt 0 ]; then
    log_pass "GET /voices returns $VOICE_COUNT voices"
    FIRST_VOICE=$(echo "$VOICES" | jq -r '.voices[0]' 2>/dev/null)
    log_info "First voice: $FIRST_VOICE"
else
    log_warn "GET /voices returns no voices (voice cloning won't work)"
fi

# Test English TTS (POST)
echo ""
echo "--- English TTS ---"
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$TEMP_DIR/en_post.wav" \
    --max-time 120 \
    -X POST "$BASE_URL/tts" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello world. This is a test.", "lang": "en"}')

if [ "$HTTP_CODE" = "200" ]; then
    FILE_TYPE=$(file "$TEMP_DIR/en_post.wav" | grep -o "WAVE audio" || echo "")
    if [ -n "$FILE_TYPE" ]; then
        SIZE=$(ls -la "$TEMP_DIR/en_post.wav" | awk '{print $5}')
        log_pass "POST /tts (English) - $SIZE bytes"
    else
        log_fail "POST /tts (English) - response is not valid WAV"
    fi
else
    log_fail "POST /tts (English) - HTTP $HTTP_CODE"
fi

# Test English TTS (GET)
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$TEMP_DIR/en_get.wav" \
    --max-time 120 \
    -G "$BASE_URL/tts" \
    --data-urlencode "text=Testing GET endpoint." \
    --data-urlencode "lang=en")

if [ "$HTTP_CODE" = "200" ]; then
    FILE_TYPE=$(file "$TEMP_DIR/en_get.wav" | grep -o "WAVE audio" || echo "")
    if [ -n "$FILE_TYPE" ]; then
        log_pass "GET /tts (English)"
    else
        log_fail "GET /tts (English) - response is not valid WAV"
    fi
else
    log_fail "GET /tts (English) - HTTP $HTTP_CODE"
fi

# Test Portuguese TTS
echo ""
echo "--- Portuguese TTS ---"
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$TEMP_DIR/pt_post.wav" \
    --max-time 120 \
    -X POST "$BASE_URL/tts" \
    -H "Content-Type: application/json" \
    -d '{"text": "Olá mundo. Isto é um teste.", "lang": "pt"}')

if [ "$HTTP_CODE" = "200" ]; then
    FILE_TYPE=$(file "$TEMP_DIR/pt_post.wav" | grep -o "WAVE audio" || echo "")
    if [ -n "$FILE_TYPE" ]; then
        SIZE=$(ls -la "$TEMP_DIR/pt_post.wav" | awk '{print $5}')
        log_pass "POST /tts (Portuguese) - $SIZE bytes"
    else
        log_fail "POST /tts (Portuguese) - response is not valid WAV"
    fi
else
    log_fail "POST /tts (Portuguese) - HTTP $HTTP_CODE"
fi

# Test voice cloning (may fail in Docker if voices not mounted)
echo ""
echo "--- Voice Cloning ---"
if [ "$VOICE_COUNT" -gt 0 ]; then
    # Use de_female.flac which is >5 seconds (required by Chatterbox)
    VOICE="de_female.flac"
    # Fallback to first voice if de_female doesn't exist
    if ! echo "$VOICES" | jq -e '.voices | index("de_female.flac")' > /dev/null 2>&1; then
        VOICE=$(echo "$VOICES" | jq -r '.voices[0]')
        log_info "de_female.flac not available, using $VOICE (may fail if <5s)"
    fi
    HTTP_CODE=$(curl -s -w "%{http_code}" -o "$TEMP_DIR/voice_clone.wav" \
        --max-time 120 \
        -X POST "$BASE_URL/tts" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"Testing voice cloning.\", \"lang\": \"en\", \"voice\": \"$VOICE\"}")

    if [ "$HTTP_CODE" = "200" ]; then
        FILE_TYPE=$(file "$TEMP_DIR/voice_clone.wav" | grep -o "WAVE audio" || echo "")
        if [ -n "$FILE_TYPE" ]; then
            log_pass "Voice cloning with $VOICE"
        else
            log_warn "Voice cloning - response is not valid WAV"
        fi
    else
        log_warn "Voice cloning - HTTP $HTTP_CODE (may be expected in Docker)"
    fi
else
    log_warn "Voice cloning - skipped (no voices available)"
fi

# Test streaming endpoint
echo ""
echo "--- Streaming Endpoint ---"
HTTP_CODE=$(curl -s -w "%{http_code}" -o "$TEMP_DIR/stream.bin" \
    --max-time 180 \
    -X POST "$BASE_URL/tts/stream" \
    -H "Content-Type: application/json" \
    -d '{"text": "First sentence. Second sentence.", "lang": "en"}')

if [ "$HTTP_CODE" = "200" ]; then
    SIZE=$(ls -la "$TEMP_DIR/stream.bin" | awk '{print $5}')
    if [ "$SIZE" -gt 100 ]; then
        log_pass "POST /tts/stream - $SIZE bytes"
    else
        log_fail "POST /tts/stream - response too small ($SIZE bytes)"
    fi
else
    log_fail "POST /tts/stream - HTTP $HTTP_CODE"
fi

# Test invalid language
echo ""
echo "--- Error Handling ---"
HTTP_CODE=$(curl -s -w "%{http_code}" -o /dev/null \
    -X POST "$BASE_URL/tts" \
    -H "Content-Type: application/json" \
    -d '{"text": "Test", "lang": "invalid"}')

if [ "$HTTP_CODE" = "400" ]; then
    log_pass "Invalid language returns 400"
else
    log_fail "Invalid language should return 400, got $HTTP_CODE"
fi

# Summary
echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo -e "${GREEN}Passed${NC}:   $PASSED"
echo -e "${RED}Failed${NC}:   $FAILED"
echo -e "${YELLOW}Warnings${NC}: $WARNINGS"
echo ""

# Play audio option
if [ "$1" = "--play" ] || [ "$1" = "-p" ]; then
    echo "--- Playing Audio Samples ---"
    if command -v afplay &> /dev/null; then
        PLAYER="afplay"
    elif command -v aplay &> /dev/null; then
        PLAYER="aplay"
    elif command -v play &> /dev/null; then
        PLAYER="play"
    else
        echo "No audio player found (afplay/aplay/play)"
        exit $FAILED
    fi

    echo "Playing English sample..."
    $PLAYER "$TEMP_DIR/en_post.wav" 2>/dev/null || true

    echo "Playing Portuguese sample..."
    $PLAYER "$TEMP_DIR/pt_post.wav" 2>/dev/null || true

    if [ -f "$TEMP_DIR/voice_clone.wav" ]; then
        echo "Playing voice clone sample..."
        $PLAYER "$TEMP_DIR/voice_clone.wav" 2>/dev/null || true
    fi
fi

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
