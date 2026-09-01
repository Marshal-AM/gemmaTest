#!/usr/bin/env python3
"""Verify vLLM is installed for the correct CUDA version."""

import sys


def main() -> int:
    try:
        import vllm
    except ImportError as e:
        print(f"ERROR: vllm is not installed: {e}")
        print("Fix: TORCH_CUDA_INDEX=cu129 ./scripts/install_vllm.sh")
        return 1
    except OSError as e:
        err = str(e)
        if "libcudart" in err or "libnvrtc" in err or "torchcodec" in err:
            print(f"ERROR: CUDA runtime mismatch: {e}")
            print()
            print("PyPI vLLM/torchcodec default to CUDA 13 (libcudart.so.13 / libnvrtc.so.13).")
            print("Your GPU VM likely needs cu129:")
            print("  TORCH_CUDA_INDEX=cu129 ./scripts/fix_cuda_stack.sh")
            print("  ./scripts/fix_torchcodec.sh")
            return 1
        raise

    print(f"vLLM:     {vllm.__version__}")

    try:
        import torch

        print(f"PyTorch:  {torch.__version__}")
        print(f"CUDA:     {torch.version.cuda or 'n/a'}")
        if torch.cuda.is_available():
            print(f"GPU:      {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    print("OK — vLLM imports successfully with matching CUDA runtime")
    return 0


if __name__ == "__main__":
    sys.exit(main())
