#!/usr/bin/env python3
# artifact/scripts/validate_environment.py
import sys
import torch
import tensorflow as tf
import sklearn

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    return True

def check_frameworks():
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ TensorFlow: {tf.__version__}")
    print(f"✅ Scikit-Learn: {sklearn.__version__}")
    return True

if __name__ == "__main__":
    print("🔍 M-TRACE Artifact Environment Validation")
    gpu_ok = check_gpu()
    fw_ok = check_frameworks()
    sys.exit(0 if (gpu_ok and fw_ok) else 1)