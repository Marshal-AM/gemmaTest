#!/usr/bin/env python3
"""Verify torchaudio matches the PyTorch CUDA build."""

import sys


def main() -> int:
    try:
        import torch
        import torchaudio
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Fix: TORCH_CUDA_INDEX=cu129 ./scripts/install_torch.sh")
        return 1
    except OSError as e:
        if "libcudart" in str(e) or "Could not load this library" in str(e):
            print(f"ERROR: torchaudio CUDA mismatch: {e}")
            print()
            print("torchaudio was built for a different CUDA than torch.")
            print("Fix:")
            print("  pip uninstall -y torch torchvision torchaudio vllm")
            print("  TORCH_CUDA_INDEX=cu129 ./scripts/install_torch.sh")
            print("  TORCH_CUDA_INDEX=cu129 ./scripts/install_vllm.sh")
            return 1
        raise

    print(f"torch:      {torch.__version__} (cuda {torch.version.cuda})")
    print(f"torchaudio: {torchaudio.__version__}")
    print("OK — torchaudio loads with matching CUDA runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
