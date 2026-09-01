#
# Gemma 4 E2B audio LLM service for Pipecat voice pipelines.
#

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Callable, Optional

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

from gemma_vllm_client import GemmaVLLMClient, configure_vllm_env
from pytorch_runtime import configure_inference_env, configure_pytorch_runtime, configure_torch_backends

configure_inference_env()

MODEL_ID = "google/gemma-4-E2B-it"
ASSISTANT_MODEL_ID = "google/gemma-4-E2B-it-assistant"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = float(os.getenv("GEMMA_MAX_AUDIO_SECONDS", "8"))
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
    """Audio-in / text-out LLM using google/gemma-4-E2B-it.

    Backends (set ``GEMMA_BACKEND``):
      - ``vllm`` (recommended): vLLM server with prefix caching, chunked prefill,
        async scheduling — same class of optimizations as reference/op.py.
      - ``transformers``: in-process Hugging Face Transformers + optional MTP.
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
        self._assistant_model = None
        self._processor = None
        self._model_load_lock = threading.Lock()
        self._load_error: str | None = None
        self._backend = os.getenv("GEMMA_BACKEND", "vllm").lower()
        self._vllm_client: GemmaVLLMClient | None = None
        self._vllm_ready = False
        self._use_mtp = os.getenv("GEMMA_USE_MTP", "1").lower() in ("1", "true", "yes")
        self._stream_tokens = os.getenv("GEMMA_STREAM_TOKENS", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        temp_raw = os.getenv("GEMMA_TEMPERATURE", "")
        self._temperature = float(temp_raw) if temp_raw else None
        self._dtype = os.getenv("GEMMA_DTYPE", "bfloat16")
        self._torch_compile = os.getenv("GEMMA_TORCH_COMPILE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        logger.info(
            f"GemmaAudioLLMService ready (backend={self._backend}, max "
            f"{max_conversation_turns} turns, max_new_tokens={max_new_tokens}, "
            f"mtp={self._use_mtp if self._backend == 'transformers' else 'n/a'}, "
            f"stream={self._stream_tokens})"
        )

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def is_model_loaded(self) -> bool:
        if self._backend == "vllm":
            return self._vllm_ready
        return self._model is not None

    def preload(self) -> None:
        """Load model weights into GPU memory once and run a warmup pass."""
        if self._backend == "vllm":
            asyncio.run(self._preload_vllm())
            return
        self._ensure_model_loaded()
        self._warmup()

    async def preload_async(self) -> None:
        """Async preload for FastAPI lifespan."""
        if self._backend == "vllm":
            await self._preload_vllm()
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._preload_transformers)

    def _preload_transformers(self) -> None:
        self._ensure_model_loaded()
        self._warmup()

    async def _preload_vllm(self) -> None:
        configure_vllm_env()
        temp_raw = os.getenv("GEMMA_TEMPERATURE", "0.3")
        temperature = float(temp_raw) if temp_raw else 0.3
        self._vllm_client = GemmaVLLMClient(
            model_id=self._model_id,
            max_tokens=self._max_new_tokens,
            temperature=temperature,
        )
        await self._vllm_client.wait_until_ready()
        self._vllm_ready = True
        logger.info(f"vLLM backend ready at {self._vllm_client.base_url}")

    def _verify_cuda_gpu(self) -> str:
        """Confirm PyTorch can execute kernels on the attached GPU."""
        import torch

        if configure_pytorch_runtime():
            logger.info(
                "Disabled PyTorch native Triton ops (using ATen fallback kernels)"
            )

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
                "  RTX 5080/50xx: TORCH_CUDA_INDEX=cu129 ./scripts/install_torch.sh\n"
                "  Other GPUs:    TORCH_CUDA_INDEX=cu126 ./scripts/install_torch.sh\n"
                "Then: python scripts/check_gpu.py"
            ) from e

        return device_name

    def _resolve_torch_dtype(self):
        import torch

        dtype_name = self._dtype.lower()
        if dtype_name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype_name in ("fp16", "float16"):
            return torch.float16
        if dtype_name in ("fp32", "float32"):
            return torch.float32
        return "auto"

    def _resolve_attn_implementation(self) -> str:
        preferred = os.getenv("GEMMA_ATTN_IMPLEMENTATION", "auto").lower()
        if preferred != "auto":
            return preferred

        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            return "sdpa"

    def _load_model_with_attn(self, model_cls, model_id: str, token: str, local_only: bool):
        import torch

        dtype = self._resolve_torch_dtype()
        attn_impl = self._resolve_attn_implementation()
        load_kwargs = {
            "dtype": dtype,
            "device_map": "auto",
            "token": token,
            "local_files_only": local_only,
        }

        try:
            logger.info(f"Loading {model_id} (attn={attn_impl}, dtype={dtype})")
            return model_cls.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                **load_kwargs,
            )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                logger.warning(f"flash_attention_2 failed ({e}); falling back to sdpa")
                return model_cls.from_pretrained(
                    model_id,
                    attn_implementation="sdpa",
                    **load_kwargs,
                )
            raise

    def _maybe_compile(self, model, label: str):
        if not self._torch_compile:
            return model

        import torch

        logger.info(f"torch.compile enabled for {label}")
        return torch.compile(model, mode="reduce-overhead")

    def _ensure_model_loaded(self) -> None:
        if self._backend == "vllm":
            if not self._vllm_ready:
                raise RuntimeError(
                    "vLLM backend is not ready. Start ./scripts/start_vllm.sh first."
                )
            return

        if self._model is not None:
            return

        if self._load_error is not None:
            raise RuntimeError(f"Gemma model failed to load: {self._load_error}")

        with self._model_load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise RuntimeError(f"Gemma model failed to load: {self._load_error}")

            from transformers import AutoModelForMultimodalLM, AutoProcessor, Gemma4AssistantForCausalLM

            token = self._hf_token or _get_hf_token()
            local_only = os.getenv("HF_LOCAL_FILES_ONLY", "0").lower() in ("1", "true", "yes")
            assistant_id = os.getenv("GEMMA_ASSISTANT_MODEL_ID", ASSISTANT_MODEL_ID)

            try:
                device_name = self._verify_cuda_gpu()
                configure_torch_backends()
                logger.info(f"Loading Gemma on GPU ({device_name}): {self._model_id}")

                self._processor = AutoProcessor.from_pretrained(
                    self._model_id,
                    token=token,
                    local_files_only=local_only,
                )
                self._model = self._load_model_with_attn(
                    AutoModelForMultimodalLM,
                    self._model_id,
                    token,
                    local_only,
                )
                self._model.eval()
                self._model = self._maybe_compile(self._model, "gemma-main")

                if self._use_mtp:
                    logger.info(f"Loading Gemma MTP assistant: {assistant_id}")
                    dtype = self._resolve_torch_dtype()
                    self._assistant_model = Gemma4AssistantForCausalLM.from_pretrained(
                        assistant_id,
                        dtype=dtype,
                        device_map="auto",
                        token=token,
                        local_files_only=local_only,
                    )
                    self._assistant_model.eval()
                    self._assistant_model.generation_config.num_assistant_tokens_schedule = (
                        "heuristic"
                    )
                    self._assistant_model = self._maybe_compile(
                        self._assistant_model, "gemma-assistant"
                    )
                    logger.info("MTP speculative decoding enabled (up to ~3x faster decode)")

                logger.info(f"Gemma ready on GPU ({device_name})")
            except Exception as e:
                self._processor = None
                self._model = None
                self._assistant_model = None
                self._load_error = str(e)
                raise

    def _warmup(self) -> None:
        """Prime CUDA kernels with a tiny silent-audio inference."""
        import torch

        logger.info("Running Gemma warmup inference...")
        t0 = time.perf_counter()
        silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        with torch.inference_mode():
            self._generate_blocking(silence)
        logger.info(f"Gemma warmup complete in {time.perf_counter() - t0:.2f}s")

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
            return self._system_prompt

        lines = [self._system_prompt, "", "Recent conversation:"]
        for turn in self._conversation_history[-3:]:
            lines.append(f"- You: {turn['assistant']}")

        return "\n".join(lines)

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
                    {"type": "text", "text": "Respond briefly in Tamil."},
                    {"type": "audio", "audio": audio_float32},
                ],
            },
        ]

    def _prepare_inputs(self, audio_float32: np.ndarray):
        import torch

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
        return inputs, inputs["input_ids"].shape[-1]

    def _build_generate_kwargs(self, inputs) -> dict:
        kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "use_cache": True,
        }
        if self._temperature is not None and self._temperature > 0:
            # reference/op.py uses temperature=0.3 for Ultravox
            kwargs["do_sample"] = True
            kwargs["temperature"] = self._temperature
        else:
            kwargs["do_sample"] = False

        if self._assistant_model is not None:
            kwargs["assistant_model"] = self._assistant_model
        return kwargs

    def _decode_response(self, outputs, input_len: int, prefix) -> str:
        assert self._processor is not None
        raw_response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        parsed = self._processor.parse_response(raw_response, prefix=prefix)
        return _extract_response_text(parsed)

    def _generate_streaming(
        self, audio_float32: np.ndarray, on_token: Callable[[str], None] | None = None
    ) -> str:
        import torch
        from transformers import TextIteratorStreamer

        self._ensure_model_loaded()
        assert self._processor is not None and self._model is not None

        t0 = time.perf_counter()
        inputs, input_len = self._prepare_inputs(audio_float32)
        prefill_s = time.perf_counter() - t0

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )
        gen_kwargs = {**inputs, **self._build_generate_kwargs(inputs), "streamer": streamer}

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        parts: list[str] = []
        for token_text in streamer:
            parts.append(token_text)
            if on_token:
                on_token(token_text)
        thread.join()

        decode_s = time.perf_counter() - t0 - prefill_s
        logger.info(
            f"Gemma inference: prefill={prefill_s:.2f}s decode={decode_s:.2f}s "
            f"tokens≈{len(parts)}"
        )
        return "".join(parts).strip()

    def _generate_blocking(self, audio_float32: np.ndarray) -> str:
        import torch

        self._ensure_model_loaded()
        assert self._processor is not None and self._model is not None

        t0 = time.perf_counter()
        inputs, input_len = self._prepare_inputs(audio_float32)
        prefill_s = time.perf_counter() - t0

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **self._build_generate_kwargs(inputs))

        decode_s = time.perf_counter() - t0 - prefill_s
        logger.info(f"Gemma inference: prefill={prefill_s:.2f}s decode={decode_s:.2f}s")

        return self._decode_response(outputs, input_len, inputs["input_ids"])

    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        """Process collected audio with Gemma — mirrors reference/op.py streaming flow."""
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

                full_response = ""
                ttfb_stopped = False

                if self._backend == "vllm":
                    assert self._vllm_client is not None
                    system_prompt = self._build_enhanced_system_prompt()
                    async for token_text in self._vllm_client.generate_stream(
                        system_prompt=system_prompt,
                        audio=audio_float32,
                    ):
                        if not ttfb_stopped:
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                        full_response += token_text
                        yield LLMTextFrame(text=token_text)
                else:
                    loop = asyncio.get_running_loop()
                    token_queue: asyncio.Queue[str | None] = asyncio.Queue()
                    result_box: dict[str, str] = {}

                    def on_token(token_text: str) -> None:
                        loop.call_soon_threadsafe(token_queue.put_nowait, token_text)

                    def run_inference() -> None:
                        try:
                            if self._stream_tokens:
                                result_box["text"] = self._generate_streaming(
                                    audio_float32, on_token=on_token
                                )
                            else:
                                result_box["text"] = self._generate_blocking(audio_float32)
                        finally:
                            loop.call_soon_threadsafe(token_queue.put_nowait, None)

                    inference_task = loop.run_in_executor(self._executor, run_inference)
                    streamed_any = False
                    while True:
                        token_text = await token_queue.get()
                        if token_text is None:
                            break
                        if not ttfb_stopped:
                            await self.stop_ttfb_metrics()
                            ttfb_stopped = True
                        streamed_any = True
                        full_response += token_text
                        yield LLMTextFrame(text=token_text)

                    await inference_task
                    if not full_response:
                        full_response = result_box.get("text", "")
                    if full_response and not streamed_any:
                        yield LLMTextFrame(text=full_response)

                if not ttfb_stopped:
                    await self.stop_ttfb_metrics()

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
