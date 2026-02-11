#!/usr/bin/env python3
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit/libdevice"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Show all logs

import tensorflow as tf

print("XLA configuration:")
print(f"  XLA_FLAGS env: {os.environ.get('XLA_FLAGS')}")
print(f"  libdevice file exists: {os.path.exists('/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc')}")

# Force XLA compilation
@tf.function(jit_compile=True)
def test(x):
    return tf.matmul(x, x)

try:
    result = test(tf.random.normal((128, 128)))
    print("\n✅ XLA compilation SUCCESSFUL - libdevice accessible")
except Exception as e:
    print(f"\n❌ XLA compilation FAILED: {type(e).__name__}")
    print(f"   {str(e)[:200]}")