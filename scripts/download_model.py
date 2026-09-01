#!/usr/bin/env python3
"""Download Gemma 4 E2B + MTP assistant to the local Hugging Face cache.

Run once on the VM after setup:
    source venv/bin/activate
    python scripts/download_model.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL_ID = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
ASSISTANT_MODEL_ID = os.getenv("GEMMA_ASSISTANT_MODEL_ID", "google/gemma-4-E2B-it-assistant")
DOWNLOAD_MTP = os.getenv("GEMMA_USE_MTP", "1").lower() in ("1", "true", "yes")


def main() -> int:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set. Copy .env.example to .env and add your token.")
        return 1

    from huggingface_hub import snapshot_download

    models = [MODEL_ID]
    if DOWNLOAD_MTP:
        models.append(ASSISTANT_MODEL_ID)

    for model_id in models:
        print(f"Downloading {model_id} ...")
        path = snapshot_download(repo_id=model_id, token=token)
        print(f"  cached at: {path}")

    print("\nDone. Set HF_LOCAL_FILES_ONLY=1 in .env and run: python bot.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
