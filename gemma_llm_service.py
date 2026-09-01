#
# Gemma 4 E2B audio LLM service for Pipecat voice pipelines.
#

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

MODEL_ID = "google/gemma-4-E2B-it"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 15
MIN_AUDIO_SECONDS = 0.3
FALLBACK_RESPONSE = "மன்னிக்கவும், ஏதோ technical issue வந்துச்சு. மறுபடியும் சொல்லுங்க?"


def _get_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to .env (see .env.example). "
            "Get a token at https://huggingface.co/settings/tokens"
        )
    return token


def _extract_response_text(parsed) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
    return str(parsed).strip()


class GemmaAudioLLMService(LLMService):
    """Audio-in / text-out LLM using google/gemma-4-E2B-it via Hugging Face Transformers.

    The model is loaded lazily on the first inference (or when preload() is called).
    Run scripts/download_model.py once on the VM before starting the server.
    """

    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        max_conversation_turns: int = 10,
        model_id: str = MODEL_ID,
        hf_token: Optional[str] = None,
        **kwargs,
    ):
        self._system_prompt = system_prompt or "You are a helpful assistant."
        super().__init__(
            settings=LLMSettings(
                model=model_id,
                system_instruction=self._system_prompt,
                temperature=None,
                max_tokens=max_new_tokens,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
            ),
            **kwargs,
        )
        self._max_new_tokens = max_new_tokens
        self._max_conversation_turns = max_conversation_turns
        self._model_id = model_id
        self._hf_token = hf_token
        self._conversation_history: list[dict[str, str]] = []
        self._error_count = 0
        self._user_speaking = False
        self._audio_frames: list[AudioRawFrame] = []
        self._buffer_start_idx = 0
        self._is_processing = False
        self._generation_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = None
        self._processor = None
        self._model_load_lock = threading.Lock()
        self._load_error: str | None = None
        logger.info(
            f"GemmaAudioLLMService ready (GPU load happens once at server startup; max "
            f"{max_conversation_turns} conversation turns)"
        )

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def preload(self) -> None:
        """Load model weights into GPU memory once."""
        self._ensure_model_loaded()

    def _verify_cuda_gpu(self) -> str:
        """Confirm PyTorch can execute kernels on the attached GPU."""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. This agent requires a GPU.\n"
                "Run: nvidia-smi\n"
                "Then: python scripts/check_gpu.py"
            )

        device_name = torch.cuda.get_device_name(0)
        try:
            probe = torch.zeros(1, device="cuda")
            probe.add_(1)
            torch.cuda.synchronize()
        except RuntimeError as e:
            raise RuntimeError(
                f"PyTorch cannot run on GPU '{device_name}': {e}\n"
                "Your PyTorch CUDA build does not match this GPU. Fix:\n"
                "  TORCH_CUDA_INDEX=cu126 ./scripts/install_torch.sh\n"
                "  TORCH_CUDA_INDEX=cu128 ./scripts/install_torch.sh\n"
                "Then: python scripts/check_gpu.py"
            ) from e

        return device_name

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        if self._load_error is not None:
            raise RuntimeError(f"Gemma model failed to load: {self._load_error}")

        with self._model_load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise RuntimeError(f"Gemma model failed to load: {self._load_error}")

            from transformers import AutoModelForMultimodalLM, AutoProcessor

            token = self._hf_token or _get_hf_token()
            local_only = os.getenv("HF_LOCAL_FILES_ONLY", "0").lower() in ("1", "true", "yes")

            try:
                device_name = self._verify_cuda_gpu()
                logger.info(f"Loading Gemma on GPU ({device_name}): {self._model_id}")

                self._processor = AutoProcessor.from_pretrained(
                    self._model_id,
                    token=token,
                    local_files_only=local_only,
                )
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    self._model_id,
                    dtype="auto",
                    device_map="auto",
                    token=token,
                    local_files_only=local_only,
                )
                logger.info(f"Gemma ready on GPU ({device_name})")
            except Exception as e:
                self._processor = None
                self._model = None
                self._load_error = str(e)
                raise

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Model is preloaded at server startup in bot.py lifespan.

    def _manage_conversation_history(self) -> None:
        if len(self._conversation_history) > self._max_conversation_turns:
            removed_count = len(self._conversation_history) - self._max_conversation_turns
            self._conversation_history = self._conversation_history[-self._max_conversation_turns :]
            logger.info(
                f"Trimmed conversation history: removed {removed_count} old turns, "
                f"kept {len(self._conversation_history)} recent turns"
            )

    def _build_enhanced_system_prompt(self) -> str:
        if not self._conversation_history:
            logger.info("No previous conversation history, using base system prompt")
            return self._system_prompt

        context_text = "Previous conversation context:\n"
        for i, turn in enumerate(self._conversation_history, 1):
            context_text += f"\nTurn {i}:\n"
            context_text += f"User said: {turn['user']}\n"
            context_text += f"You responded: {turn['assistant']}\n"

        logger.info(f"Built context with {len(self._conversation_history)} previous turns")
        return (
            f"{self._system_prompt}\n\n{context_text}\n\n"
            "Now respond to the current user's speech:"
        )

    def _frames_to_audio_float32(self, frames: list[AudioRawFrame]) -> Optional[np.ndarray]:
        audio_arrays: list[np.ndarray] = []

        for frame in frames:
            if not frame.audio:
                continue
            if isinstance(frame.audio, bytes):
                arr = np.frombuffer(frame.audio, dtype=np.int16)
                if arr.size > 0:
                    audio_arrays.append(arr)
            elif isinstance(frame.audio, np.ndarray):
                if frame.audio.size > 0:
                    if frame.audio.dtype != np.int16:
                        audio_arrays.append(frame.audio.astype(np.int16))
                    else:
                        audio_arrays.append(frame.audio)

        if not audio_arrays:
            return None

        audio_int16 = np.concatenate(audio_arrays)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if len(audio_float32) == 0:
            return None
        if np.any(np.isnan(audio_float32)) or np.any(np.isinf(audio_float32)):
            logger.warning("Invalid audio data (NaN/Inf), skipping processing")
            return None

        audio_float32 = np.clip(audio_float32, -1.0, 1.0)

        max_audio_length = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)
        min_audio_length = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)

        if len(audio_float32) > max_audio_length:
            logger.warning(
                f"Audio too long ({len(audio_float32) / SAMPLE_RATE:.1f}s), "
                f"truncating to {MAX_AUDIO_SECONDS:.1f}s"
            )
            audio_float32 = audio_float32[:max_audio_length]

        if len(audio_float32) < min_audio_length:
            logger.warning(
                f"Audio too short ({len(audio_float32) / SAMPLE_RATE:.2f}s), "
                f"padding to {MIN_AUDIO_SECONDS:.2f}s"
            )
            padding = np.zeros(min_audio_length - len(audio_float32), dtype=np.float32)
            audio_float32 = np.concatenate([audio_float32, padding])

        return audio_float32

    def _build_messages(self, audio_float32: np.ndarray) -> list[dict]:
        return [
            {"role": "system", "content": self._build_enhanced_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Listen to my speech and respond following your instructions.",
                    },
                    {"type": "audio", "audio": audio_float32},
                ],
            },
        ]

    def _generate_blocking(self, audio_float32: np.ndarray) -> str:
        self._ensure_model_loaded()
        assert self._processor is not None and self._model is not None

        messages = self._build_messages(audio_float32)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        raw_response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        parsed = self._processor.parse_response(raw_response, prefix=inputs["input_ids"])
        return _extract_response_text(parsed)

    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        if self._is_processing:
            logger.warning("Already processing audio, skipping")
            return

        frames = list(self._audio_frames[self._buffer_start_idx :])
        self._audio_frames.clear()
        self._buffer_start_idx = 0

        if not frames:
            logger.warning("No audio frames to process")
            return

        audio_float32 = self._frames_to_audio_float32(frames)
        if audio_float32 is None:
            logger.warning("No valid audio data found in frames")
            return

        logger.info(
            f"Processing audio: {len(audio_float32) / SAMPLE_RATE:.2f} seconds, "
            f"{len(audio_float32)} samples"
        )

        async with self._generation_lock:
            self._is_processing = True
            try:
                await self.start_ttfb_metrics()
                await self.start_processing_metrics()
                yield LLMFullResponseStartFrame()

                loop = asyncio.get_running_loop()
                full_response = await loop.run_in_executor(
                    self._executor, self._generate_blocking, audio_float32
                )
                await self.stop_ttfb_metrics()

                if full_response:
                    yield LLMTextFrame(text=full_response)

                await self.stop_processing_metrics()
                yield LLMFullResponseEndFrame()

                if full_response.strip():
                    self._conversation_history.append(
                        {
                            "user": "[Audio input received]",
                            "assistant": full_response.strip(),
                        }
                    )
                    logger.info(
                        f"Stored conversation turn. Total history: "
                        f"{len(self._conversation_history)} turns"
                    )
                    logger.info(f"Assistant response: {full_response[:100]}...")
                    self._manage_conversation_history()

                self._error_count = 0

            except Exception as e:
                logger.error(f"Error generating text from Gemma model: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                self._error_count += 1

                if self._error_count >= 3:
                    logger.warning(
                        f"Too many consecutive errors ({self._error_count}), "
                        "resetting conversation history"
                    )
                    self._conversation_history = []
                    self._error_count = 0

                yield LLMFullResponseStartFrame()
                yield LLMTextFrame(text=FALLBACK_RESPONSE)
                yield LLMFullResponseEndFrame()
            finally:
                self._is_processing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            # Keep audio already received (includes the chunk that triggered VAD).
            self._buffer_start_idx = max(0, len(self._audio_frames) - 1)
            logger.info("VAD: user started speaking — buffering audio")
            await self.push_frame(frame, direction)
        elif isinstance(frame, (AudioRawFrame, InputAudioRawFrame)):
            self._audio_frames.append(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            logger.info(
                f"VAD: user stopped speaking — processing "
                f"{len(self._audio_frames) - self._buffer_start_idx} audio frames"
            )
            await self.push_frame(frame, direction)
            await self.process_generator(self._process_audio_buffer())
        elif isinstance(frame, InterruptionFrame):
            self._user_speaking = False
            self._audio_frames.clear()
            self._buffer_start_idx = 0
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
