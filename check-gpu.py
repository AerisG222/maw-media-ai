#!/usr/bin/env python3
"""
check_gpu.py
-------------
Reports if a GPU is available and can be used for face scanning (ONNX Runtime CUDA provider).
"""

import sys

try:
    import onnxruntime as ort
except ImportError:
    print(
        "onnxruntime is not installed. Please install it with 'pip install onnxruntime' or 'pip install onnxruntime-gpu'."
    )
    sys.exit(1)

providers = ort.get_available_providers()

print("ONNX Runtime available providers:", providers)

if "CUDAExecutionProvider" in providers:
    print("✅ GPU is available and ONNX Runtime will use CUDA for face scanning.")
else:
    print("❌ GPU is NOT available. ONNX Runtime will use CPU only for face scanning.")

# Optional: also check PyTorch CUDA (for completeness)
try:
    import torch

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch detected {torch.cuda.device_count()} CUDA device(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  - {torch.cuda.get_device_name(i)}")
except ImportError:
    pass
