#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu.py — Check GPU availability and configuration

Detects available GPU hardware and reports whether TensorFlow
can use it for acceleration.

Usage:
    ./pt.py gpu
"""

import argparse


def check_gpu():
    """Detect and report GPU availability and configuration."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print(f"\n{'─'*50}")
            print(f"  GPU Status: 🚀 Acceleration available")
            print(f"{'─'*50}")
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                name    = details.get("device_name", f"GPU:{i}")
                print(f"  Device {i}  : {name}")
            print(f"  Count     : {len(gpus)} device(s)")
            print(f"  TF version: {tf.__version__}")
            print(f"{'─'*50}\n")
        else:
            import platform
            print(f"\n{'─'*50}")
            print(f"  GPU Status: ℹ  No GPU detected — running on CPU")
            print(f"{'─'*50}")
            print(f"  TF version: {tf.__version__}")
            print(f"  Platform  : {platform.system()} {platform.machine()}")
            print()
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                print("  For Apple Silicon GPU support, run:")
                print("    pip install tensorflow-macos tensorflow-metal")
            else:
                print("  For NVIDIA GPU support, install CUDA + cuDNN and run:")
                print("    pip install tensorflow[and-cuda]")
            print(f"{'─'*50}\n")

    except Exception as e:
        print(f"\n{'─'*50}")
        print(f"  GPU Status: ⚠  Check failed")
        print(f"{'─'*50}")
        print(f"  Error: {e}")
        print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check GPU availability and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.parse_args()
    check_gpu()


if __name__ == "__main__":
    main()
