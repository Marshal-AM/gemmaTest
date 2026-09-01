#!/usr/bin/env python3
"""Verify vLLM is installed for the correct CUDA version."""

import sys
from pathlib import Path


def _parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
    return tuple(parts or (0,))


def main() -> int:
    venv_root = Path(__file__).resolve().parent.parent / "venv"
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

    vllm_file = Path(vllm.__file__).resolve()
    if venv_root.exists() and venv_root not in vllm_file.parents:
        print(f"WARN: vLLM is loaded from outside the project venv: {vllm_file}")
        print("      Run: TORCH_CUDA_INDEX=cu129 ./scripts/install_vllm.sh")

    version = getattr(vllm, "__version__", "0")
    print(f"vLLM:     {version}")
    if _parse_version(version) < _parse_version("0.28.0"):
        print("ERROR: Gemma 4 requires vLLM >= 0.28.0 (fixes head_dim crash with transformers 5.15+).")
        print("Fix: VLLM_MIN_VERSION=0.28.0 ./scripts/install_vllm.sh")
        return 1

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
