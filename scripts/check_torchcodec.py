#!/usr/bin/env python3
"""Verify torchcodec does not require CUDA 13 libs (libnvrtc.so.13)."""

import sys


def main() -> int:
    try:
        import torchcodec  # noqa: F401
    except ImportError:
        print("torchcodec: not installed (OK for audio-only vLLM >= 0.25.1)")
        return 0
    except (OSError, RuntimeError) as e:
        err = str(e)
        if "libnvrtc" in err or "libcudart" in err or "libtorchcodec" in err:
            print(f"ERROR: torchcodec CUDA mismatch: {e}")
            print()
            print("Fix:")
            print("  ./scripts/fix_torchcodec.sh")
            return 1
        raise

    print("torchcodec: loads without CUDA 13 runtime errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
