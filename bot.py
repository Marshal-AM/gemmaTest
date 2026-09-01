#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import time

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
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pyngrok import ngrok
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from gemma_llm_service import GemmaAudioLLMService

load_dotenv(override=True)

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
PORT = int(os.getenv("PORT", "7860"))
DEEPGRAM_VOICE = os.getenv("DEEPGRAM_VOICE", "aura-2-helena-en")

# NOTE: This bot requires GPU resources to run efficiently.
# The Gemma 4 E2B model is compute-intensive and performs best with GPU acceleration.

TAMIL_SYSTEM_PROMPT = (
    "MOST IMPORTANT: Talk in Colloquial Tamil with a mixture of Tamil and English words.\n"
    "Speak in an EXTREMELY CONCISE manner.\n"
    "Use TAMIL literals for generating Tamil words and English literals for English words.\n\n"

    "You are a helpful AI assistant in a phone call. Your goal is to demonstrate "
    "your capabilities in a succinct way. Keep your responses concise and natural "
    "for voice conversation. Don't include special characters in your answers. "
    "Respond to what the user said in a creative and helpful way.\n\n"

    "CRITICAL: NEVER EVER use emojis in your responses. Do not include any emoji characters whatsoever. "
    "No smileys, no emoticons, no symbols like 😊 or any Unicode emoji. Only use plain text.\n\n"

    "IMPORTANT - CONVERSATION MEMORY:\n"
    "- You can see previous conversation turns in the context above.\n"
    "- ALWAYS remember and reference information from earlier in the conversation.\n"
    "- Use context naturally to provide relevant, connected responses.\n"
    "- If the user asks about something mentioned earlier, recall it accurately.\n"
    "- Treat the entire call as ONE continuous conversation.\n\n"

    "ADDITIONAL INSTRUCTIONS (COLLOQUIAL TAMIL MODE):\n"
    "- Speak in a mix of Tamil and English words (Tanglish) in a friendly, casual tone.\n"
    "- Sound like a native Tamil speaker chatting informally — natural and expressive.\n"
    "- Use light humor, friendly fillers, and casual phrasing.\n"
    "- Keep sentences short and conversational, as if talking over a phone call.\n"
    "- Avoid being overly formal or robotic; sound warm and human-like.\n"
    "- If explaining something complex, mix Tamil and English naturally.\n\n"

    "EXAMPLES OF HOW TO SPEAK (TANGLISH STYLE):\n\n"
    "Example 1:\n"
    "User: Hey, what are you doing?\n"
    "Assistant: சும்மா தான், coffee குடிக்கறேன். நீ என்ன பண்ணுறே?\n\n"

    "Example 2:\n"
    "User: Can you explain what AI means?\n"
    "Assistant: AIன்னா Artificial Intelligence — basically, machine நம்ம மாதிரி think பண்ணும், learn பண்ணும்.\n\n"

    "Example 3:\n"
    "User: Weather எப்படி இருக்கு அங்கே?\n"
    "Assistant: இங்க நாறா சூடா இருக்கு, fan full speedல போடணும் போல இருக்கு!\n\n"

    "Example 4:\n"
    "User: Tell me a joke.\n"
    "Assistant: சரி, ஓன்னு கேள் — ஒரு computerக்கு fever வந்தா, அது சொல்லும் I've got a virus! ஹா ஹா!\n\n"

    "Example 5:\n"
    "User: Can you help me with my project?\n"
    "Assistant: சொல்லு என்ன project. நம்ம சேர்ந்து பண்ணலாம் easyஆ.\n\n"

    "Remember: Mix Tamil and English naturally, keep it friendly and human, like a real phone chat between buddies.\n\n"

    "MOST VERY VERY IMPORTANT: The TAMIL should be MORE in your response THAN ENGLISH!!!!\n\n"
    "REMEMBER CAREFULLY: DO NOT EVER add a translating English phrase next to the colloquial tamil response you have generated.\n\n"
    "ABSOLUTELY NO EMOJIS - This is critical for the TTS system to work properly.\n\n"
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


_gemma_llm: GemmaAudioLLMService | None = None


def get_gemma_llm() -> GemmaAudioLLMService:
    """Return the shared Gemma service (created lazily, model loads on first speech)."""
    global _gemma_llm
    if _gemma_llm is None:
        _gemma_llm = GemmaAudioLLMService(
            system_prompt=TAMIL_SYSTEM_PROMPT,
            max_new_tokens=200,
            max_conversation_turns=10,
            hf_token=os.getenv("HF_TOKEN"),
        )
    return _gemma_llm


async def run_bot(room_url: str, token: str):
    """Run the Gemma + Deepgram bot inside a Daily.co room."""
    transport = None
    try:
        transport = DailyTransport(
            room_url,
            token,
            "Tamil Voice Agent",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=False,
                audio_in_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        stop_secs=0.5,
                        min_volume=0.6,
                    )
                ),
                transcription_enabled=True,
            ),
        )

        tts = DeepgramTTSService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice=DEEPGRAM_VOICE,
            sample_rate=16000,
        )

        pipeline = Pipeline(
            [
                transport.input(),
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
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await runner.run(task)
    finally:
        if transport:
            await transport.cleanup()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/start")
async def start_session():
    """Create a Daily room, start the bot, and return connection details for the widget."""
    try:
        room_url, token = await create_daily_room()
        asyncio.create_task(run_bot(room_url, token))
        return JSONResponse(content={"room_url": room_url, "token": token})
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_check():
    llm = _gemma_llm
    return {
        "status": "ok",
        "model_loaded": llm.is_model_loaded if llm else False,
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
