# Tamil Voice Agent (Gemma 4 + Daily + Deepgram)

Real-time voice bot that listens via Daily.co, reasons over speech with **google/gemma-4-E2B-it**, and responds in colloquial Tamil via Deepgram TTS.

## Requirements

- Linux VM with **NVIDIA GPU** (recommended: 16 GB+ VRAM)
- Python 3.11+
- API keys: [Hugging Face](https://huggingface.co/settings/tokens), [Daily.co](https://dashboard.daily.co), [Deepgram](https://console.deepgram.com)

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
| `DEEPGRAM_API_KEY` | Yes | Deepgram API key |
| `HF_LOCAL_FILES_ONLY` | After download | Set `1` to use cached model only |
| `TORCH_CUDA_INDEX` | GPU errors | `cu129`+ for RTX 50xx — see GPU troubleshooting |
| `PYTORCH_DISABLE_NATIVE_TRITON` | Triton errors | `auto` (default), or `1` to force ATen fallback |
| `GEMMA_ATTN_IMPLEMENTATION` | No | `sdpa` (default), `eager`, or `flash_attention_2` |
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
