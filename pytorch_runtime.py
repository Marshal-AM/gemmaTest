"""PyTorch runtime tweaks for newer GPUs (e.g. RTX 5080 / sm_120)."""

from __future__ import annotations

import os

_configured = False
_triton_disabled = False


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
