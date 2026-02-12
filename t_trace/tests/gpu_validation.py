import torch
import tensorflow as tf
import subprocess
import sys

print("="*60)
print("GPU COMPUTE VALIDATION FOR M-TRACE EXPERIMENTS")
print("="*60)

# 1. NVIDIA Driver & CUDA
print("\n[1] NVIDIA Driver & CUDA:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    print(result.stdout if result.returncode == 0 else "❌ nvidia-smi failed")
except Exception as e:
    print(f"❌ Error: {e}")

# 2. PyTorch GPU Detection
print("\n[2] PyTorch GPU Detection:")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    # Test tensor computation
    x = torch.randn(10000, 10000).cuda()
    y = torch.randn(10000, 10000).cuda()
    z = torch.matmul(x, y)
    print(f"  ✓ GPU computation test PASSED (tensor shape: {z.shape})")
else:
    print("  ❌ CUDA NOT AVAILABLE - PyTorch won't use GPU!")

# 3. TensorFlow GPU Detection
print("\n[3] TensorFlow GPU Detection:")
print(f"  TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPUs detected: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"    - {gpu}")
    # Test computation
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
    print(f"  ✓ GPU computation test PASSED (tensor shape: {c.shape})")
else:
    print("  ❌ NO GPUS DETECTED - TensorFlow won't use GPU!")

# 4. Memory Check (critical for RTX 4080 Super's 16GB)
print("\n[4] GPU Memory Availability:")
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Total GPU memory: {total_mem:.2f} GB")
    print(f"  Required for M-TRACE experiments: ~8-12 GB (depending on model size)")
    if total_mem >= 12:
        print("  ✓ Sufficient memory for large transformer models")
    else:
        print("  ⚠ Limited memory - use smaller batch sizes")

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)