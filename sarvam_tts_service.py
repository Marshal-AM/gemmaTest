"""Sarvam TTS service with corrected API fields and WAV sample-rate handling.

Pipecat's built-in SarvamHttpTTSService sends ``sample_rate`` and
``target_language_code``, but the Sarvam REST API expects ``speech_sample_rate``
and ``language_code``. When those are wrong, Sarvam returns 24 kHz audio while
frames are tagged 16 kHz — playback sounds slow and low-pitched.
"""

from __future__ import annotations

import base64
import io
import wave
from collections.abc import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.sarvam.tts import SarvamHttpTTSService
from pipecat.utils.tracing.service_decorators import traced_tts


def _parse_wav_pcm(audio_bytes: bytes) -> tuple[bytes, int]:
    """Extract mono 16-bit PCM and the true sample rate from a WAV blob."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit audio")
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono audio, got {wf.getnchannels()} channels")
        return wf.readframes(wf.getnframes()), wf.getframerate()


class TamilSarvamTTSService(SarvamHttpTTSService):
    """Sarvam HTTP TTS with API payload + WAV metadata fixes for Tamil voice bots."""

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        try:
            payload: dict = {
                "text": text,
                "language_code": self._settings.language,
                "speaker": self._settings.voice,
                "speech_sample_rate": self.sample_rate,
                "model": self._settings.model,
                "pace": self._settings.pace if self._settings.pace is not None else 1.0,
                "output_audio_codec": "wav",
            }

            if self._config.supports_pitch:
                payload["pitch"] = (
                    self._settings.pitch if self._settings.pitch is not None else 0.0
                )
            if self._config.supports_loudness:
                payload["loudness"] = (
                    self._settings.loudness if self._settings.loudness is not None else 1.0
                )
            if self._config.supports_temperature:
                payload["temperature"] = (
                    self._settings.temperature if self._settings.temperature is not None else 0.6
                )

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
                **sdk_headers(),
            }

            async with self._session.post(
                f"{self._base_url}/text-to-speech", json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Sarvam API error: {error_text}")
                    return
                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            audios = response_data.get("audios") or []
            if not audios:
                yield ErrorFrame(error="No audio data received from Sarvam")
                return

            wav_bytes = base64.b64decode(audios[0])
            pcm, actual_rate = _parse_wav_pcm(wav_bytes)

            if actual_rate != self.sample_rate:
                logger.debug(
                    f"Sarvam returned {actual_rate} Hz audio (pipeline target {self.sample_rate} Hz)"
                )

            yield TTSAudioRawFrame(
                audio=pcm,
                sample_rate=actual_rate,
                num_channels=1,
                context_id=context_id,
            )
        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
