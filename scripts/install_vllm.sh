#!/usr/bin/env bash
# Install vLLM with the correct CUDA wheel (NOT default PyPI cu130).
#
# IMPORTANT: run install_torch.sh first (or let this script do it) so
# torch + torchvision + torchaudio all match cu129.
set -euo pipefail

cd "$(dirname "$0")/.."
CUDA_INDEX="${TORCH_CUDA_INDEX:-cu129}"
CUDA_VERSION="${CUDA_INDEX#cu}"  # e.g. cu129 -> 129
VLLM_MIN_VERSION="${VLLM_MIN_VERSION:-0.25.1}"

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"
CPU_ARCH="$(uname -m)"

echo "==> Step 1/5: Reinstall matching torch stack (${CUDA_INDEX})"
pip uninstall -y vllm torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX}"

echo "==> Step 2/5: Install vLLM cu${CUDA_VERSION} wheel (>= ${VLLM_MIN_VERSION})"
export VLLM_MIN_VERSION
VLLM_VERSION="$(python3 - <<'PY'
import json
import os
import re
import urllib.request

def parse_version(value: str) -> tuple[int, ...]:
    match = re.match(r"([0-9]+(?:\.[0-9]+)*)", value)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split("."))

min_version = parse_version(os.environ.get("VLLM_MIN_VERSION", "0.25.1"))
releases = json.load(
    urllib.request.urlopen(
        "https://api.github.com/repos/vllm-project/vllm/releases",
        timeout=30,
    )
)
for release in releases:
    if release.get("draft") or release.get("prerelease"):
        continue
    tag = release["tag_name"].lstrip("v")
    if parse_version(tag) >= min_version:
        print(tag)
        break
else:
    print(os.environ.get("VLLM_MIN_VERSION", "0.25.1"))
PY
)"

if [ -z "${VLLM_VERSION}" ]; then
  echo "ERROR: Could not detect a vLLM release >= ${VLLM_MIN_VERSION}"
  exit 1
fi
echo "    vLLM version: ${VLLM_VERSION}"

WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl"
echo "    ${WHEEL_URL}"

if ! pip install "${WHEEL_URL}"; then
  echo "ERROR: Failed to install vLLM cu${CUDA_VERSION} wheel."
  echo "Try a different CUDA index, e.g.:"
  echo "  TORCH_CUDA_INDEX=cu128 ./scripts/install_vllm.sh"
  exit 1
fi

echo "==> Step 3/5: Re-pin torch stack (vLLM may have pulled wrong CUDA deps)"
pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX}" --force-reinstall

echo "==> Step 4/5: Fix torchcodec CUDA 13 mismatch"
chmod +x scripts/fix_torchcodec.sh
./scripts/fix_torchcodec.sh

echo "==> Step 5/5: Install API client + audio helpers"
pip install openai librosa soundfile

echo ""
python scripts/check_gpu.py
python scripts/check_torchaudio.py
python scripts/check_torchcodec.py
python scripts/check_vllm.py

echo ""
echo "Done. Start the inference server:"
echo "  ./scripts/start_vllm.sh"
echo ""
echo "Then in another terminal:"
echo "  GEMMA_BACKEND=vllm python bot.py"
