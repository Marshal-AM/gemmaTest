#!/usr/bin/env bash
# One-time VM setup: venv, dependencies, .env template.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> Setting up gemmatest in $ROOT"

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.11+ first."
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "    Python $PYTHON_VERSION"

if [ ! -d venv ]; then
  echo "==> Creating virtualenv"
  python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

CUDA_INDEX="${TORCH_CUDA_INDEX:-cu129}"

echo "==> Installing PyTorch + torchvision + torchaudio (${CUDA_INDEX})"
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"

echo "==> Installing project dependencies"
pip install -r requirements.txt

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
  echo "    Edit .env and fill in your API keys before running."
else
  echo "==> .env already exists, skipping"
fi

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (HF_TOKEN, DAILY_API_KEY, DEEPGRAM_API_KEY)"
echo "  2. Download the model:  python scripts/download_model.py"
echo "  3. Check GPU:           python scripts/check_gpu.py"
echo "  4. Start the server:    ./scripts/start.sh"
