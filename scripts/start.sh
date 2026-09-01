#!/usr/bin/env bash
# Start the Tamil voice agent server.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d venv ]; then
  echo "ERROR: venv not found. Run ./scripts/setup.sh first."
  exit 1
fi

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Run ./scripts/setup.sh or copy .env.example to .env."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "Starting bot on port ${PORT:-7860} ..."
exec python bot.py
