#!/usr/bin/env bash
# One-shot fix for libcudart.so.13 / torchaudio CUDA mismatch errors.
set -euo pipefail

cd "$(dirname "$0")/.."
export TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu129}"

echo "==> Fixing CUDA stack for ${TORCH_CUDA_INDEX}"
chmod +x scripts/install_torch.sh scripts/install_vllm.sh
./scripts/install_torch.sh
./scripts/install_vllm.sh
