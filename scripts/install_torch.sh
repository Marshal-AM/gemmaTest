#!/usr/bin/env bash
# Reinstall PyTorch + torchvision for your GPU's CUDA version.
#
# Usage:
#   ./scripts/install_torch.sh              # defaults to cu129 (RTX 50xx / Blackwell)
#   TORCH_CUDA_INDEX=cu130 ./scripts/install_torch.sh
#
# RTX 5080 / sm_120 needs cu129, cu130, or cu132 — NOT cu124/cu126/cu128.
set -euo pipefail

cd "$(dirname "$0")/.."
CUDA_INDEX="${TORCH_CUDA_INDEX:-cu129}"

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Removing old torch/torchvision (if any)"
pip uninstall -y torch torchvision 2>/dev/null || true

echo "==> Installing torch + torchvision (${CUDA_INDEX})"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"

echo ""
python scripts/check_gpu.py
