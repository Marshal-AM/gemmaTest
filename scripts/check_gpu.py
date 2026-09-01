#!/usr/bin/env python3
"""Verify PyTorch can run on this VM's GPU."""

import sys
from pathlib import Path

# Allow importing pytorch_runtime from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed")
        return 1

    from pytorch_runtime import configure_pytorch_runtime, test_bmm_outer_product_kernel

    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.version.cuda or 'not built with CUDA'}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU visible to PyTorch.")
        print("  - Is an NVIDIA GPU attached? Run: nvidia-smi")
        print("  - Reinstall PyTorch: TORCH_CUDA_INDEX=cu129 ./scripts/install_torch.sh")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU:      {name}")
    print(f"SM:       {cap[0]}.{cap[1]}")

    if configure_pytorch_runtime():
        print("Triton:   native ops disabled (ATen fallback — required on some RTX 50xx GPUs)")

    try:
        x = torch.zeros(1, device="cuda")
        x += 1
        torch.cuda.synchronize()
        print("Test 1:   OK — basic GPU kernels")
    except RuntimeError as e:
        print(f"ERROR:    Basic GPU kernel test failed: {e}")
        _print_torch_fix(cap)
        return 1

    try:
        test_bmm_outer_product_kernel()
        print("Test 2:   OK — bmm outer-product (Gemma RoPE path)")
    except RuntimeError as e:
        print(f"ERROR:    bmm outer-product test failed: {e}")
        print()
        print("This is the path that fails during Gemma inference on some GPUs.")
        print("Try setting in .env:")
        print("  PYTORCH_DISABLE_NATIVE_TRITON=1")
        print("Then restart the server.")
        _print_torch_fix(cap)
        return 1

    return 0


def _print_torch_fix(cap: tuple[int, int]) -> None:
    print()
    print("Your PyTorch CUDA wheel may not match this GPU.")
    if cap[0] >= 12:
        print("RTX 50xx / Blackwell (sm_120) needs cu129, cu130, or cu132:")
        print("  TORCH_CUDA_INDEX=cu129 ./scripts/install_torch.sh")
        print("  TORCH_CUDA_INDEX=cu130 ./scripts/install_torch.sh")
    else:
        print("Try:")
        print("  TORCH_CUDA_INDEX=cu126 ./scripts/install_torch.sh")
        print("  TORCH_CUDA_INDEX=cu128 ./scripts/install_torch.sh")
    print()
    print("Then restart: python bot.py")


if __name__ == "__main__":
    sys.exit(main())
