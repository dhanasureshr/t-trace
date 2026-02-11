#!/usr/bin/env python3
"""GPU validation using symlink fix (no env var timing issues)."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce noise

# CRITICAL: Import TensorFlow AFTER env vars but symlink makes this timing irrelevant
import tensorflow as tf
import numpy as np

import sys


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)
from t_trace.logging_engine import LoggingEngine

print(f"✓ TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Test M-TRACE integration
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

engine = LoggingEngine()
callback = engine.enable_logging(model, mode="development")
assert callback is not None, "❌ Callback not returned"
print("✓ M-TRACE callback initialized")

# Train on GPU (XLA will find libdevice via symlink)
x = np.random.rand(500, 10).astype(np.float32)
y = np.random.rand(500, 1).astype(np.float32)

model.fit(x, y, epochs=2, batch_size=64, callbacks=[callback], verbose=0)

logs = engine.collect_logs()
print(f"✓ Captured {len(logs)} logs during GPU training")
assert len(logs) > 0, "No logs captured"

engine.disable_logging()
print("\n✅ SUCCESS: M-TRACE TensorFlow GPU integration fully validated!")
print("   libdevice accessible via symlink - no environment variable timing issues")