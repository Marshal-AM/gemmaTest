#!/usr/bin/env bash
# Fix libnvrtc.so.13 / libtorchcodec_image.so errors on CUDA 12.9 VMs.
#
# vLLM pulls torchcodec for video decode. PyPI's default torchcodec wheel is
# built for CUDA 13, which breaks on cu129 systems. This bot only needs audio.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Removing CUDA 13 torchcodec wheel"
pip uninstall -y torchcodec 2>/dev/null || true

echo "==> Installing CPU torchcodec (audio-only bot does not need GPU video decode)"
if ! pip install torchcodec --index-url "https://download.pytorch.org/whl/cpu"; then
  echo "WARN: CPU torchcodec install failed; leaving torchcodec uninstalled."
  echo "      vLLM >= 0.25.1 will skip video decode if torchcodec is missing."
fi

python scripts/check_torchcodec.py
