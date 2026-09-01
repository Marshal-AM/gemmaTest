#!/usr/bin/env python3
"""Smoke-test Sarvam Bulbul TTS with a Tamil phrase."""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_TEXT = "என்ன புரியல? கொஞ்சம் தெளிவா சொல்லுங்க.நான் hear பண்ணிட்டேன்."
DEFAULT_LANGUAGE = "ta-IN"
DEFAULT_MODEL = "bulbul:v3"
DEFAULT_SPEAKER = "ritu"
DEFAULT_CODEC = "wav"
API_URL = "https://api.sarvam.ai/text-to-speech"


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)

    api_key = os.getenv("SARVAM_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        print("ERROR: SARVAM_API_KEY is not set in .env")
        return 1

    text = os.getenv("SARVAM_TEST_TEXT", DEFAULT_TEXT)
    language_code = os.getenv("SARVAM_LANGUAGE_CODE", DEFAULT_LANGUAGE).strip()
    model = os.getenv("SARVAM_MODEL", DEFAULT_MODEL).strip()
    speaker = os.getenv("SARVAM_SPEAKER", DEFAULT_SPEAKER).strip()
    codec = os.getenv("SARVAM_OUTPUT_CODEC", DEFAULT_CODEC).strip()
    sample_rate = int(os.getenv("SARVAM_SAMPLE_RATE", "24000"))
    out_path = project_root / f"sarvam_test.{codec}"

    payload = {
        "text": text,
        "language_code": language_code,
        "model": model,
        "speaker": speaker,
        "output_audio_codec": codec,
        "speech_sample_rate": sample_rate,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print("Calling Sarvam TTS...")
    print(f"  model:    {model}")
    print(f"  speaker:  {speaker}")
    print(f"  language: {language_code}")
    try:
        print(f"  text:     {text[:60]}{'...' if len(text) > 60 else ''}")
    except UnicodeEncodeError:
        print(f"  text:     ({len(text)} Tamil/English characters)")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        print(f"ERROR: Sarvam returned HTTP {e.code}")
        try:
            print(json.dumps(json.loads(raw), indent=2))
        except json.JSONDecodeError:
            print(raw)
        return 1
    except urllib.error.URLError as e:
        print(f"ERROR: request failed: {e.reason}")
        return 1

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print("ERROR: Sarvam returned non-JSON response")
        print(body[:500])
        return 1

    audios = data.get("audios") or []
    if not audios:
        print("ERROR: Sarvam response had no audio data")
        print(json.dumps(data, indent=2))
        return 1

    audio = base64.b64decode("".join(audios))
    out_path.write_bytes(audio)
    size_kb = len(audio) / 1024
    request_id = data.get("request_id", "n/a")
    print(f"OK — request_id: {request_id}")
    print(f"OK — saved {size_kb:.1f} KB to {out_path}")
    print("Play the file to confirm Tamil TTS sounds right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
