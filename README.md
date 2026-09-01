# Tamil Voice Agent (Gemma 4 + Daily + Sarvam)

Real-time voice bot that listens via Daily.co, reasons over speech with **google/gemma-4-E2B-it**, and responds in colloquial Tamil via **Sarvam Bulbul** TTS.

## Requirements

- Linux VM with **NVIDIA GPU** (recommended: 16 GB+ VRAM)
- Python 3.11+
- API keys: [Hugging Face](https://huggingface.co/settings/tokens), [Daily.co](https://dashboard.daily.co), [Sarvam AI](https://dashboard.sarvam.ai)

## Quick start (VM)

```bash
git clone <your-repo-url> gemmatest
cd gemmatest

chmod +x scripts/*.sh
./scripts/setup.sh          # venv + pip install (no model download)
```

Edit `.env` with your keys (created from `.env.example`):

```bash
nano .env
```

Download the Gemma model **once**:

```bash
source venv/bin/activate
python scripts/download_model.py
```

Start the server:

```bash
./scripts/start.sh
```

The agent creates a Daily room, joins it, and prints the URL. Open that link in your browser, allow microphone access, and start speaking.

Health check:

```bash
curl http://localhost:7860/health
# {"status":"ok","model_loaded":true,"room_url":"https://....daily.co/..."}
```

## GPU / CUDA troubleshooting

**Error: `no kernel image is available for execution on the device`**

Your PyTorch CUDA wheel doesn't match your GPU. This is not a model bug — reinstall PyTorch:

```bash
source venv/bin/activate
python scripts/check_gpu.py          # diagnose

# Try cu126 (most cloud GPUs), then cu128 (newer GPUs e.g. Blackwell):
TORCH_CUDA_INDEX=cu126 ./scripts/install_torch.sh
# or
TORCH_CUDA_INDEX=cu128 ./scripts/install_torch.sh

# RTX 5080 / 50xx (compute capability 12.0) — use cu129 or newer:
TORCH_CUDA_INDEX=cu129 ./scripts/install_torch.sh
# or
TORCH_CUDA_INDEX=cu130 ./scripts/install_torch.sh

./scripts/start.sh
```

| GPU generation | PyTorch CUDA index |
|----------------|-------------------|
| A100, L4, L40, RTX 30xx/40xx | `cu124` or `cu126` |
| H100, H200 | `cu126` or `cu128` |
| **RTX 5080 / 50xx (sm_120)** | **`cu129`, `cu130`, or `cu132`** |

The model loads **once at server startup** (not on every speech turn). If startup fails, fix PyTorch before joining the room.

### Inference speed

**Recommended: vLLM backend** (same optimization class as `reference/op.py`)

Terminal 1 — start vLLM (once per machine boot):
```bash
./scripts/install_vllm.sh      # one-time: pip install vllm[audio]
./scripts/start_vllm.sh        # prefix caching, chunked prefill, flash attn
```

Terminal 2 — start the voice bot:
```bash
GEMMA_BACKEND=vllm python bot.py
```

`reference/op.py` used **Ultravox + vLLM** internally. Our stack uses **Gemma 4 E2B + vLLM** the same way — the `VLLM_*` env vars and `enable-prefix-caching` / `enable-chunked-prefill` flags only apply when vLLM is the inference engine, not Hugging Face Transformers.

**Fallback: Transformers backend** (no separate server):
```bash
GEMMA_BACKEND=transformers python bot.py
```

Uses MTP assistant + bfloat16 + token streaming. Slower than vLLM but simpler to run.

| Variable | Default | Effect |
|----------|---------|--------|
| `GEMMA_BACKEND` | `vllm` | `vllm` or `transformers` |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | Matches reference/op.py |
| `VLLM_MAX_MODEL_LEN` | `4096` | Matches reference/op.py |
| `GEMMA_USE_MTP` | `1` | Transformers only — speculative decode |
| `GEMMA_MAX_NEW_TOKENS` | `64` | Shorter responses = faster |
| `GEMMA_FULL_PROMPT` | `0` | Compact system prompt |

**Error: `Triton Error [CUDA]: device kernel image is invalid`**

Basic CUDA works but Gemma inference fails during RoPE matmul. On RTX 5080 / sm_120 this is fixed automatically; if it persists:

```bash
# In .env:
PYTORCH_DISABLE_NATIVE_TRITON=1

python scripts/check_gpu.py   # should pass Test 2 (bmm outer-product)
python bot.py
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face token (accept Gemma license first) |
| `DAILY_API_KEY` | Yes | Daily.co API key |
| `SARVAM_API_KEY` | Yes | Sarvam AI API key |
| `SARVAM_SPEAKER` | No | Bulbul voice (default `kavitha`) |
| `SARVAM_MODEL` | No | `bulbul:v3` (default) or `bulbul:v2` |
| `SARVAM_PACE` | No | Speech pace (default `1.0`) |
| `HF_LOCAL_FILES_ONLY` | After download | Set `1` to use cached model only |
| `TORCH_CUDA_INDEX` | GPU errors | `cu129`+ for RTX 50xx — see GPU troubleshooting |
| `PYTORCH_DISABLE_NATIVE_TRITON` | Triton errors | `auto` (default), or `1` to force ATen fallback |
| `GEMMA_ATTN_IMPLEMENTATION` | No | `auto` (flash_attn if installed, else sdpa) |
| `GEMMA_DTYPE` | No | `bfloat16` (default, matches reference/op.py) |
| `GEMMA_TEMPERATURE` | No | Empty = greedy (fastest). Set `0.3` to match op.py |
| `GEMMA_TORCH_COMPILE` | No | `0` (default). Set `1` to try `torch.compile` |
| `GEMMA_USE_MTP` | No | `1` (default) — load assistant model for ~3× faster decode |
| `GEMMA_MAX_NEW_TOKENS` | No | Max response length (default `64` for voice) |
| `GEMMA_FULL_PROMPT` | No | `0` (default, fast) or `1` for long prompt with examples |
| `GEMMA_STREAM_TOKENS` | No | `1` (default) — stream tokens to TTS as they generate |
| `NGROK_AUTHTOKEN` | No | Public URL tunnel for widgets |
| `PORT` | No | Server port (default `7860`) |

## Project layout

```
bot.py                  # FastAPI server + Daily/Pipecat pipeline
gemma_llm_service.py    # Gemma 4 audio LLM (GPU, loaded once at startup)
scripts/
  setup.sh              # One-time dependency install
  install_torch.sh      # Reinstall PyTorch for your GPU's CUDA version
  check_gpu.py          # Verify GPU kernels work
  download_model.py     # One-time model download
  start.sh              # Run the server
requirements.txt
.env.example
```

## Notes

- The model is **not** downloaded during `setup.sh` — run `download_model.py` once.
- The model loads **once on GPU at server startup** (~1–2 min), not on every speech turn.
- `daily-python` (Daily transport) requires Linux/macOS — use a Linux GPU VM for production.
