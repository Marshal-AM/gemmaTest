#!/usr/bin/env bash
# Install vLLM with the correct CUDA wheel (NOT default PyPI cu130).
#
# IMPORTANT: run install_torch.sh first (or let this script do it) so
# torch + torchvision + torchaudio all match cu129.
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

PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"
CPU_ARCH="$(uname -m)"

echo "==> Step 1/4: Reinstall matching torch stack (${CUDA_INDEX})"
pip uninstall -y vllm torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX}"

echo "==> Step 2/4: Install vLLM cu${CUDA_VERSION} wheel"
VLLM_VERSION="$(curl -fsSL https://api.github.com/repos/vllm-project/vllm/releases/latest \
  | grep -oP '"tag_name":\s*"\K[^"]+' | sed 's/^v//' || true)"

if [ -z "${VLLM_VERSION}" ]; then
  echo "ERROR: Could not detect latest vLLM release version"
  exit 1
fi

WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"
echo "    ${WHEEL_URL}"

if ! pip install "${WHEEL_URL}"; then
  echo "ERROR: Failed to install vLLM cu${CUDA_VERSION} wheel."
  echo "Try a different CUDA index, e.g.:"
  echo "  TORCH_CUDA_INDEX=cu128 ./scripts/install_vllm.sh"
  exit 1
fi

echo "==> Step 3/4: Re-pin torch stack (vLLM may have pulled wrong CUDA deps)"
pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX}" --force-reinstall

echo "==> Step 4/4: Install API client + audio helpers"
pip install openai librosa soundfile

echo ""
python scripts/check_gpu.py
python scripts/check_torchaudio.py
python scripts/check_vllm.py

echo ""
echo "Done. Start the inference server:"
echo "  ./scripts/start_vllm.sh"
echo ""
echo "Then in another terminal:"
echo "  GEMMA_BACKEND=vllm python bot.py"
