#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=True)

from pytorch_runtime import configure_inference_env

configure_inference_env()

# Monkeypatch for HTTPMethod import compatibility
if sys.version_info < (3, 11):
    import http
    if not hasattr(http, 'HTTPMethod'):
        from enum import Enum
        class HTTPMethod(Enum):
            GET = "GET"
            POST = "POST"
            PUT = "PUT"
            DELETE = "DELETE"
            PATCH = "PATCH"
            HEAD = "HEAD"
            OPTIONS = "OPTIONS"
        http.HTTPMethod = HTTPMethod

import aiohttp
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pyngrok import ngrok
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transcriptions.language import Language
from pipecat.transports.daily.transport import DailyParams, DailyTransport, DailyTranscriptionSettings

from gemma_llm_service import GemmaAudioLLMService
from sarvam_tts_service import TamilSarvamTTSService

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
PORT = int(os.getenv("PORT", "7860"))
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip().strip('"').strip("'")
SARVAM_SPEAKER = os.getenv("SARVAM_SPEAKER", "kavitha")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "bulbul:v3")
SARVAM_PACE = float(os.getenv("SARVAM_PACE", "1.0"))

# NOTE: This bot requires GPU resources to run efficiently.
# The Gemma 4 E2B model is compute-intensive and performs best with GPU acceleration.

# Simple persona for the cab booking voice agent.
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Your name is Malar and you work at XYZ, a cab company where people book cabs. "
    "Help callers with bookings, pickup, drop-off, fares, and ride status. "
    "Keep responses short and clear for a phone call.",
)


def setup_ngrok_proxy(port: int = PORT):
    """Expose the agent server publicly so a local widget can reach it."""
    ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN") or os.getenv("NGROK_AUTH_TOKEN")
    if not ngrok_auth_token:
        logger.warning("NGROK_AUTHTOKEN not set — server will only be reachable locally")
        return None

    ngrok.set_auth_token(ngrok_auth_token)
    tunnel = ngrok.connect(port, "http")
    proxy_url = tunnel.public_url
    logger.info(f"Ngrok tunnel created: {proxy_url}")
    return proxy_url


async def create_daily_room() -> tuple[str, str]:
    """Create a Daily.co room and bot token."""
    if not DAILY_API_KEY:
        raise ValueError("DAILY_API_KEY must be set")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "exp": int(time.time()) + 3600,
                    "enable_chat": False,
                    "enable_emoji_reactions": False,
                }
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to create Daily room: {response.status} - {error_text}")

            room_data = await response.json()
            room_url = room_data.get("url")
            room_name = room_data.get("name")
            if not room_url or not room_name:
                raise Exception("Invalid room data from Daily API")

        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers={
                "Authorization": f"Bearer {DAILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": True,
                }
            },
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to create Daily token: {response.status} - {error_text}")

            token_data = await response.json()
            token = token_data.get("token")
            if not token:
                raise Exception("Invalid token data from Daily API")

    logger.info(f"Created Daily room: {room_url}")
    return room_url, token


_active_room_url: str | None = None
_bot_task: asyncio.Task | None = None


def print_join_banner(room_url: str) -> None:
    """Print the room URL prominently so you can open it in a browser."""
    banner = f"""
{'=' * 72}
  Tamil Voice Agent is waiting in the Daily room.

  Open this URL in your browser and allow microphone access:
  {room_url}

  The agent will hear you when you speak in the room.
{'=' * 72}
"""
    print(banner, flush=True)
    logger.info(f"Join the voice room: {room_url}")


_gemma_llm: GemmaAudioLLMService | None = None


def get_gemma_llm() -> GemmaAudioLLMService:
    """Return the shared Gemma service (created lazily, model loads on first speech)."""
    global _gemma_llm
    if _gemma_llm is None:
        _gemma_llm = GemmaAudioLLMService(
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=int(os.getenv("GEMMA_MAX_NEW_TOKENS", "128")),
            max_conversation_turns=int(os.getenv("GEMMA_MAX_CONVERSATION_TURNS", "10")),
            hf_token=os.getenv("HF_TOKEN"),
        )
    return _gemma_llm


async def run_bot(room_url: str, token: str):
    """Run the Gemma + Sarvam bot inside a Daily.co room."""
    transport = None
    try:
        if not SARVAM_API_KEY:
            raise ValueError("SARVAM_API_KEY must be set")

        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=float(os.getenv("VAD_STOP_SECS", "0.35")),
                    min_volume=0.4,
                )
            ),
        )

        transport = DailyTransport(
            room_url,
            token,
            "Tamil Voice Agent",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=False,
                audio_in_passthrough=True,
                transcription_enabled=True,
                transcription_settings=DailyTranscriptionSettings(
                    language=os.getenv("DAILY_TRANSCRIPTION_LANGUAGE", "ta"),
                    model=os.getenv("DAILY_TRANSCRIPTION_MODEL", "nova-2-general"),
                    punctuate=True,
                ),
            ),
        )

        async with aiohttp.ClientSession() as http_session:
            tts = TamilSarvamTTSService(
                api_key=SARVAM_API_KEY,
                aiohttp_session=http_session,
                sample_rate=16000,
                settings=TamilSarvamTTSService.Settings(
                    voice=SARVAM_SPEAKER,
                    model=SARVAM_MODEL,
                    language=Language.EN_IN,
                    pace=SARVAM_PACE,
                ),
            )

            pipeline = Pipeline(
                [
                    transport.input(),
                    vad,
                    get_gemma_llm(),
                    tts,
                    transport.output(),
                ]
            )

            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    audio_in_sample_rate=16000,
                    audio_out_sample_rate=16000,
                    enable_metrics=True,
                    enable_usage_metrics=True,
                ),
            )

            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.info(f"Participant joined Daily room: {participant}")
                await transport.capture_participant_transcription(participant["id"])

            @transport.event_handler("on_participant_joined")
            async def on_participant_joined(transport, participant):
                logger.info(f"Participant joined: {participant}")
                await transport.capture_participant_transcription(participant["id"])

            @transport.event_handler("on_participant_left")
            async def on_participant_left(transport, participant, reason):
                logger.info(f"Participant left: {participant}, reason: {reason}")

            runner = PipelineRunner()
            await runner.run(task)
    finally:
        if transport:
            await transport.cleanup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Gemma on GPU once, then create a Daily room and join it."""
    global _active_room_url, _bot_task

    logger.info("Loading Gemma model (one-time, may take 1-2 minutes)...")
    await get_gemma_llm().preload_async()
    logger.info("Gemma model loaded — starting voice session")

    room_url, token = await create_daily_room()
    _active_room_url = room_url
    print_join_banner(room_url)
    _bot_task = asyncio.create_task(run_bot(room_url, token))
    yield
    if _bot_task and not _bot_task.done():
        _bot_task.cancel()
        try:
            await _bot_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/start")
async def start_session():
    """Return the active room URL (room is created automatically on server startup)."""
    if _active_room_url:
        return JSONResponse(content={"room_url": _active_room_url})
    return JSONResponse(status_code=503, content={"error": "Agent room not ready yet"})


@app.get("/health")
async def health_check():
    llm = _gemma_llm
    return {
        "status": "ok",
        "model_loaded": llm.is_model_loaded if llm else False,
        "room_url": _active_room_url,
    }


if __name__ == "__main__":
    proxy_url = setup_ngrok_proxy(PORT)
    if proxy_url:
        logger.info(f"Widget server URL: {proxy_url}")
        logger.info(f"Point widget SERVER_URL to: {proxy_url}")
    else:
        logger.info(f"Local server URL: http://localhost:{PORT}")

    logger.info(f"Starting agent server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
