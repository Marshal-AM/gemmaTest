#!/usr/bin/env bash
# Install vLLM with audio support for Gemma 4 E2B.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Installing vLLM with audio extras (this may take several minutes)"
pip install -U "vllm[audio]" openai

echo ""
echo "Done. Start the inference server:"
echo "  ./scripts/start_vllm.sh"
echo ""
echo "Then in another terminal:"
echo "  GEMMA_BACKEND=vllm python bot.py"
