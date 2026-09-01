#!/usr/bin/env python3
"""Verify PyTorch can run on this VM's GPU."""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed")
        return 1

    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.version.cuda or 'not built with CUDA'}")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU visible to PyTorch.")
        print("  - Is an NVIDIA GPU attached? Run: nvidia-smi")
        print("  - Reinstall PyTorch: TORCH_CUDA_INDEX=cu126 ./scripts/install_torch.sh")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU:      {name}")
    print(f"SM:       {cap[0]}.{cap[1]}")

    try:
        x = torch.zeros(1, device="cuda")
        x += 1
        torch.cuda.synchronize()
        print("Test:     OK — GPU kernels work with this PyTorch build")
    except RuntimeError as e:
        print(f"ERROR:    GPU kernel test failed: {e}")
        print()
        print("Your PyTorch CUDA wheel does not match this GPU. Try:")
        print("  TORCH_CUDA_INDEX=cu126 ./scripts/install_torch.sh")
        print("  TORCH_CUDA_INDEX=cu128 ./scripts/install_torch.sh")
        print()
        print("Then restart: ./scripts/start.sh")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
