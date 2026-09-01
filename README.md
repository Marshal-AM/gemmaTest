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
# {"status":"ok","model_loaded":false,"room_url":"https://....daily.co/..."}
```

The room URL is also returned by `POST /start` if you need it programmatically.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Hugging Face token (accept Gemma license first) |
| `DAILY_API_KEY` | Yes | Daily.co API key |
| `DEEPGRAM_API_KEY` | Yes | Deepgram API key |
| `HF_LOCAL_FILES_ONLY` | After download | Set `1` to use cached model only |
| `PRELOAD_MODEL` | No | Set `true` to load model at server startup |
| `NGROK_AUTHTOKEN` | No | Public URL tunnel for widgets |
| `PORT` | No | Server port (default `7860`) |

## Project layout

```
bot.py                  # FastAPI server + Daily/Pipecat pipeline
gemma_llm_service.py    # Gemma 4 audio LLM (lazy-loaded)
scripts/
  setup.sh              # One-time dependency install
  download_model.py     # One-time model download
  start.sh              # Run the server
requirements.txt
.env.example
```

## Notes

- The model is **not** downloaded during `setup.sh` or server startup by default.
- First speech turn triggers model load into GPU memory (~30–60 s depending on hardware).
- Set `PRELOAD_MODEL=true` in `.env` to load the model when the server starts instead.
- `daily-python` (Daily transport) requires Linux/macOS — use a Linux GPU VM for production.
