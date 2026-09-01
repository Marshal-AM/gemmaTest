#!/usr/bin/env bash
# Reinstall PyTorch + torchvision for your GPU's CUDA version.
# Usage:
#   ./scripts/install_torch.sh              # defaults to cu126
#   TORCH_CUDA_INDEX=cu128 ./scripts/install_torch.sh
set -euo pipefail

cd "$(dirname "$0")/.."
CUDA_INDEX="${TORCH_CUDA_INDEX:-cu126}"

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing torch + torchvision (${CUDA_INDEX})"
pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"

echo ""
python scripts/check_gpu.py
