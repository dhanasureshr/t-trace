#!/usr/bin/env python3
"""
End-to-end M-TRACE demonstration: MNIST CNN training with safe TensorFlow imports.
Uses only `import tensorflow as tf` to avoid keras.datasets/utils import issues.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow noise
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Use GPU

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

print("=" * 70)
print("M-TRACE TensorFlow Demonstration: MNIST CNN Training (Safe Imports)")
print("=" * 70)

# Load MNIST using TensorFlow's built-in dataset loader
print("\n✓ Loading MNIST dataset via tf.keras.datasets...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert labels using TensorFlow's to_categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
print(f"  Train shape: {x_train.shape} | Test shape: {x_test.shape}")
print(f"  Label shape: {y_train.shape}")

# Build CNN model with explicit layer names for clear logging
print("\n✓ Building CNN model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name="conv2d_1"),
    tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_1"),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
    tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_2"),
    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(64, activation='relu', name="dense_1"),
    tf.keras.layers.Dense(10, activation='softmax', name="output")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Enable M-TRACE logging in development mode
print("\n✓ Enabling M-TRACE logging (development mode)...")
engine = LoggingEngine()
callback = engine.enable_logging(model, mode="development")
run_id = engine.get_run_id()
print(f"  Run ID: {run_id[:8]}...")
print(f"  Logs directory: mtrace_logs/development/")
# After enabling logging but BEFORE model.fit():
callback = engine.enable_logging(model, mode="development")

# Train model with M-TRACE callback
print("\n✓ Training model (2 epochs) with M-TRACE instrumentation...")
history = model.fit(
    x_train, y_train,
    epochs=2,
    batch_size=128,
    validation_split=0.1,
    callbacks=[callback],
    verbose=1
)

# Disable logging to trigger final flush to disk
print("\n✓ Flushing logs to storage...")
engine.disable_logging()

# Verify logs were saved to Parquet
import glob
import os
log_pattern = f"mtrace_logs/development/*{run_id[:8]}*.parquet"
log_files = glob.glob(log_pattern)

if log_files:
    print(f"\n✅ SUCCESS: Logs saved to {len(log_files)} Parquet file(s):")
    for f in sorted(log_files, key=os.path.getmtime, reverse=True)[:3]:  # Show latest 3
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"   - {os.path.basename(f)} ({size_mb:.2f} MB)")
    
    # Quick log inspection
    try:
        import pandas as pd
        latest_log = sorted(log_files, key=os.path.getmtime, reverse=True)[0]
        df = pd.read_parquet(latest_log)
        print(f"\n📊 Log summary:")
        print(f"   Total entries: {len(df)}")
        print(f"   Unique layers: {df['layer_name'].nunique()}")
        print(f"   Layer names: {df['layer_name'].drop_duplicates().tolist()}")
        print(f"   Event types: {df['event_type'].drop_duplicates().tolist()}")
    except ImportError:
        print("\n⚠️  pandas/pyarrow not installed - skipping log inspection")
        print("   Install with: pip install pandas pyarrow")
else:
    print(f"\n❌ WARNING: No Parquet files found matching pattern: {log_pattern}")
    print("   Check mtrace_logs/development/ directory permissions")

print("\n" + "=" * 70)
print("NEXT STEP: Visualize logs in Dash dashboard")
print("=" * 70)
print("Run in a new terminal:")
print("  python -m t_trace.analysis_engine.dashboard")
print("\nThen open: http://localhost:8050")
print("=" * 70)