#!/usr/bin/env python3
"""Smoke-test ElevenLabs TTS with a Tamil phrase."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

# Tamil phrase for the test
DEFAULT_TEXT = "என்ன புரியல? கொஞ்சம் தெளிவா சொல்லுங்க.நான் hear பண்ணிட்டேன்."

# Multilingual model works well for Tamil + English mix
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
# Rachel — override with ELEVENLABS_VOICE_ID in .env if you prefer another voice
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY is not set in .env")
        return 1

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID).strip()
    model_id = os.getenv("ELEVENLABS_MODEL_ID", DEFAULT_MODEL_ID).strip()
    text = os.getenv("ELEVENLABS_TEST_TEXT", DEFAULT_TEXT)
    out_path = project_root / "elevenlabs_test.mp3"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = json.dumps({"text": text, "model_id": model_id}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )

    print("Calling ElevenLabs TTS...")
    print(f"  voice:  {voice_id}")
    print(f"  model:  {model_id}")
    try:
        print(f"  text:   {text[:60]}{'...' if len(text) > 60 else ''}")
    except UnicodeEncodeError:
        print(f"  text:   ({len(text)} Tamil/English characters)")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            audio = response.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: ElevenLabs returned HTTP {e.code}")
        try:
            print(json.loads(body))
        except json.JSONDecodeError:
            print(body)
        return 1
    except urllib.error.URLError as e:
        print(f"ERROR: request failed: {e.reason}")
        return 1

    out_path.write_bytes(audio)
    size_kb = len(audio) / 1024
    print(f"OK — saved {size_kb:.1f} KB to {out_path}")
    print("Play it with your media player to confirm the voice sounds right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
