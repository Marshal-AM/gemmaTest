#!/usr/bin/env bash
# Start vLLM with Gemma 4 E2B audio + all performance flags from reference/op.py
#
# Usage:
#   ./scripts/start_vllm.sh
#   GEMMA_MODEL_ID=google/gemma-4-E2B-it ./scripts/start_vllm.sh
#
# Requires: pip install "vllm[audio]"  (see scripts/install_vllm.sh)
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_ID="${GEMMA_MODEL_ID:-google/gemma-4-E2B-it}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

export VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-0}"
export VLLM_USE_V2_BLOCK_MANAGER="${VLLM_USE_V2_BLOCK_MANAGER:-1}"
export HF_TOKEN="${HF_TOKEN:-}"

echo "==> Starting vLLM for ${MODEL_ID} on port ${PORT}"
echo "    max_model_len=${MAX_MODEL_LEN}  gpu_memory_utilization=${GPU_MEM_UTIL}"

exec vllm serve "${MODEL_ID}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --limit-mm-per-prompt '{"image": 0, "audio": 1}'
