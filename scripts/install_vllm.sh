#!/usr/bin/env bash
# Install vLLM with the correct CUDA wheel (NOT default PyPI cu130).
#
# Default PyPI vLLM links against libcudart.so.13 (CUDA 13).
# RTX 5080 / most cloud GPUs need cu129 or cu128 instead.
#
# Usage:
#   ./scripts/install_vllm.sh
#   TORCH_CUDA_INDEX=cu129 ./scripts/install_vllm.sh
set -euo pipefail

cd "$(dirname "$0")/.."
CUDA_INDEX="${TORCH_CUDA_INDEX:-cu129}"
CUDA_VERSION="${CUDA_INDEX#cu}"  # e.g. cu129 -> 129

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Target CUDA index: ${CUDA_INDEX} (libcudart.so.${CUDA_VERSION:0:1}.${CUDA_VERSION:1})"
echo "==> Removing any existing vLLM install"
pip uninstall -y vllm 2>/dev/null || true

CPU_ARCH="$(uname -m)"
PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"

install_from_release_wheel() {
  local version
  version="$(curl -fsSL https://api.github.com/repos/vllm-project/vllm/releases/latest \
    | grep -oP '"tag_name":\s*"\K[^"]+' | sed 's/^v//')"
  if [ -z "${version}" ]; then
    return 1
  fi

  local wheel="https://github.com/vllm-project/vllm/releases/download/v${version}/vllm-${version}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"
  echo "==> Installing vLLM ${version}+cu${CUDA_VERSION} from GitHub release"
  echo "    ${wheel}"
  pip install "${wheel}" --extra-index-url "${PYTORCH_INDEX}"
}

install_from_pip_index() {
  echo "==> Fallback: pip install vllm via PyTorch ${CUDA_INDEX} index"
  pip install vllm --extra-index-url "${PYTORCH_INDEX}"
}

if ! install_from_release_wheel; then
  echo "WARN: GitHub cu${CUDA_VERSION} wheel not found, trying pip fallback..."
  install_from_pip_index
fi

echo "==> Installing OpenAI client + audio helpers for Gemma 4"
pip install openai librosa soundfile

echo ""
echo "==> Verifying vLLM import"
python scripts/check_vllm.py

echo ""
echo "Done. Start the inference server:"
echo "  ./scripts/start_vllm.sh"
echo ""
echo "Then in another terminal:"
echo "  GEMMA_BACKEND=vllm python bot.py"
