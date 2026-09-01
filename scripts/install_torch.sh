#!/usr/bin/env bash
# Reinstall PyTorch + torchvision + torchaudio for your GPU's CUDA version.
#
# All three MUST match the same CUDA index or vLLM/transformers will fail with:
#   libcudart.so.13: cannot open shared object file
#
# Usage:
#   ./scripts/install_torch.sh              # defaults to cu129 (RTX 50xx / Blackwell)
#   TORCH_CUDA_INDEX=cu130 ./scripts/install_torch.sh
set -euo pipefail

cd "$(dirname "$0")/.."
CUDA_INDEX="${TORCH_CUDA_INDEX:-cu129}"
PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Removing old torch stack"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo "==> Installing torch + torchvision + torchaudio (${CUDA_INDEX})"
pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX}"

echo ""
python scripts/check_gpu.py
python scripts/check_torchaudio.py
