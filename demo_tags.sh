#!/bin/bash
# Demo script for Chatterbox Turbo paralinguistic tags
# These only work with the English (Turbo) model

echo "=============================================="
echo "Chatterbox Turbo - Paralinguistic Tags Demo"
echo "=============================================="
echo ""

demo() {
    local name="$1"
    local text="$2"
    echo "[$name]"
    echo "  $text"
    echo ""
    ./run_en.sh "$text"
    sleep 1
}

echo "--- Sound Effects ---"
echo ""

demo "laugh" \
    "And then he said he was a professional chef [laugh] but he burned the toast!"

demo "chuckle" \
    "Well, I suppose that's one way to look at it [chuckle]."

demo "cough" \
    "Sorry about that [cough] [cough], I think I'm coming down with something."

demo "clear throat" \
    "[clear throat] Ladies and gentlemen, may I have your attention please."

demo "gasp" \
    "Wait, you're telling me the butler did it? [gasp] I never saw that coming!"

demo "groan" \
    "Ugh, another Monday morning meeting [groan]. Here we go again."

demo "sigh" \
    "I've been waiting here for three hours [sigh]. When will they call my number?"

demo "sniff" \
    "This flower smells absolutely divine [sniff]. You have to try it."

demo "shush" \
    "[shush] Keep your voice down, the baby just fell asleep."

demo "crying" \
    "[crying] I can't believe she's really gone. I'm going to miss her so much."

echo "--- Emotions and Styles ---"
echo ""

demo "angry" \
    "[angry] I specifically asked for no pickles! This is the third time this week!"

demo "happy" \
    "[happy] Oh my goodness, I got the job! This is the best day ever!"

demo "fear" \
    "[fear] Did you hear that noise? I think there's someone in the house."

demo "surprised" \
    "[surprised] You're pregnant? Oh wow, congratulations! When did you find out?"

demo "sarcastic" \
    "[sarcastic] Oh sure, because staying up until 3am was definitely a great idea."

demo "dramatic" \
    "[dramatic] In a world where nothing is as it seems, one hero must rise to face the ultimate challenge."

demo "whispering" \
    "[whispering] I heard that they're planning a surprise party for her birthday."

demo "narration" \
    "[narration] The old man sat by the window, watching the autumn leaves drift slowly to the ground."

demo "advertisement" \
    "[advertisement] Introducing the all-new SuperClean 3000! It cleans, it shines, it does your taxes!"

echo "--- Combined Examples ---"
echo ""

demo "conversation" \
    "So I walked into the meeting [clear throat] and said, excuse me, I have an announcement [dramatic]. I quit! [laugh] You should have seen their faces [gasp]."

demo "storytelling" \
    "[narration] It was a dark and stormy night. [whispering] She crept through the shadows [gasp] when suddenly [surprised] the lights came on!"

demo "emotional range" \
    "[happy] I thought everything was perfect, but then [sigh] reality set in [crying] and I realized [angry] I had been fooled all along!"

echo "=============================================="
echo "Demo Complete!"
echo "=============================================="
