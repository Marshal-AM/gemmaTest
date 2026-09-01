"""PyTorch / CUDA runtime tweaks for Gemma inference."""

from __future__ import annotations

import os

_configured = False
_triton_disabled = False
_inference_env_configured = False


def configure_inference_env() -> None:
    """Set process-wide env vars for low-latency GPU inference.

    Mirrors the intent of reference/op.py performance flags, adapted for
    Hugging Face Transformers (not vLLM).
    """
    global _inference_env_configured
    if _inference_env_configured:
        return

    # General CUDA / PyTorch throughput (safe for HF Transformers).
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Reduce allocator churn on long-running voice sessions.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _inference_env_configured = True


def configure_torch_backends() -> None:
    """Enable fast math paths once CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def configure_pytorch_runtime() -> bool:
    """Disable native Triton ops when they are unsupported on this GPU.

    PyTorch 2.12+ routes some bmm paths (used by Gemma RoPE) through Triton
    kernels that may fail on Blackwell with:
      RuntimeError: Triton Error [CUDA]: device kernel image is invalid

    Disabling falls back to standard ATen CUDA kernels (slightly slower, works).

    Control via PYTORCH_DISABLE_NATIVE_TRITON:
      - auto (default): disable on compute capability >= 12.0 (sm_120+)
      - 1 / true / yes: always disable
      - 0 / false / no: never disable

    Returns True if native Triton ops were disabled.
    """
    global _configured, _triton_disabled
    if _configured:
        return _triton_disabled

    import torch

    configure_inference_env()
    configure_torch_backends()

    mode = os.getenv("PYTORCH_DISABLE_NATIVE_TRITON", "auto").lower()
    if mode in ("1", "true", "yes"):
        should_disable = True
    elif mode in ("0", "false", "no"):
        should_disable = False
    else:
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        should_disable = cap[0] >= 12

    if should_disable:
        try:
            import torch.backends.python_native as pn

            pn.triton.enabled = False
            _triton_disabled = True
        except Exception:
            # Older PyTorch builds may not expose python_native; ignore.
            _triton_disabled = False

    _configured = True
    return _triton_disabled


def test_bmm_outer_product_kernel() -> None:
    """Exercise the bmm outer-product path used by rotary embeddings."""
    import torch

    configure_pytorch_runtime()
    a = torch.ones(2, 8, 1, device="cuda", dtype=torch.float16)
    b = torch.ones(2, 1, 128, device="cuda", dtype=torch.float16)
    out = a @ b
    torch.cuda.synchronize()
    if out.shape != (2, 8, 128):
        raise RuntimeError(f"Unexpected bmm outer-product shape: {out.shape}")
