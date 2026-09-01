#!/usr/bin/env python3
"""Download google/gemma-4-E2B-it to the local Hugging Face cache.

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


def main() -> int:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN is not set. Copy .env.example to .env and add your token.")
        return 1

    print(f"Downloading {MODEL_ID} ...")
    print("This is a one-time step (~several GB). Grab a coffee.")

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=MODEL_ID, token=token)
    print(f"Done. Model cached at:\n  {path}")
    print("\nNext: set HF_LOCAL_FILES_ONLY=1 in .env and run ./scripts/start.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
