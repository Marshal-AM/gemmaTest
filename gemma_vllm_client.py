"""Async OpenAI-compatible client for a vLLM-served Gemma 4 E2B audio model."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import wave
from collections.abc import AsyncGenerator

import aiohttp
import numpy as np
from loguru import logger
from openai import AsyncOpenAI

SAMPLE_RATE = 16000


def configure_vllm_env() -> None:
    """vLLM performance flags from reference/op.py."""
    os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "1")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
    os.environ.setdefault("VLLM_USE_V2_BLOCK_MANAGER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def float32_audio_to_wav_data_url(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Encode mono float32 audio as a base64 WAV data URL for vLLM."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


class GemmaVLLMClient:
    """Talk to `vllm serve google/gemma-4-E2B-it` via the OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_id: str,
        api_key: str = "EMPTY",
        max_tokens: int = 64,
        temperature: float = 0.3,
        timeout_s: float = 120.0,
    ) -> None:
        configure_vllm_env()
        self._base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")).rstrip(
            "/"
        )
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key,
            timeout=timeout_s,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def wait_until_ready(self, timeout_s: float = 300.0, poll_s: float = 2.0) -> None:
        """Block until the vLLM server responds."""
        health_url = self._base_url.replace("/v1", "") + "/health"
        deadline = time.monotonic() + timeout_s
        last_error = "unknown"

        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            logger.info(f"vLLM server ready at {self._base_url}")
                            return
                        last_error = f"HTTP {resp.status}"
                except Exception as e:
                    last_error = str(e)
                await asyncio.sleep(poll_s)

        raise RuntimeError(
            f"vLLM server not ready at {self._base_url} ({last_error}).\n"
            "Start it with: ./scripts/start_vllm.sh"
        )

    def _build_messages(self, system_prompt: str, audio: np.ndarray) -> list[dict]:
        audio_url = float32_audio_to_wav_data_url(audio)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                    },
                    {"type": "text", "text": "Respond briefly in Tamil."},
                ],
            },
        ]

    async def generate_stream(
        self, *, system_prompt: str, audio: np.ndarray
    ) -> AsyncGenerator[str, None]:
        """Stream decoded text tokens from vLLM."""
        messages = self._build_messages(system_prompt, audio)
        t0 = time.perf_counter()
        first = True

        stream = await self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            if first:
                logger.info(f"vLLM TTFB: {time.perf_counter() - t0:.2f}s")
                first = False
            yield delta

        logger.info(f"vLLM total: {time.perf_counter() - t0:.2f}s")
