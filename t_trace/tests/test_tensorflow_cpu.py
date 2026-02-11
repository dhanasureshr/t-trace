# tests/test_tensorflow_cpu.py
#!/usr/bin/env python3
"""CPU-only validation test for M-TRACE TensorFlow integration."""
import os
# CRITICAL: Set BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # Force CPU execution
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0" # Disable XLA compilation
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # Suppress TensorFlow noise

import sys
import os


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)
import tensorflow as tf
import numpy as np
from t_trace.logging_engine import LoggingEngine

print(f"✓ TensorFlow {tf.__version__} running on CPU\n")

def test_tensorflow_callback():
    # Build simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Enable M-TRACE logging
    engine = LoggingEngine()
    callback = engine.enable_logging(model, mode="development")
    
    # VALIDATION 1: Callback must be returned (critical for Section 2.1.2 compliance)
    if callback is None:
        print("❌ FAILED: enable_logging() returned None instead of callback")
        print("   FIX REQUIRED: Update LoggingEngine.enable_logging() to return callback for TensorFlow")
        print("   Current implementation likely discards the callback return value")
        exit(1)
    
    if not isinstance(callback, tf.keras.callbacks.Callback):
        print(f"❌ FAILED: Returned object type is {type(callback)}, not tf.keras.callbacks.Callback")
        exit(1)
    
    print("✓ Callback properly returned from enable_logging() [Section 2.1.2 compliant]")
    
    # Train with callback (standard Keras workflow)
    x_train = np.random.rand(100, 5).astype(np.float32)
    y_train = np.random.rand(100, 1).astype(np.float32)
    model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=32,
        callbacks=[callback],  # Standard Keras integration pattern
        verbose=0
    )
    
    # VALIDATION 2: Logs must be captured
    logs = engine.collect_logs()
    if len(logs) == 0:
        print("❌ FAILED: No logs captured during training")
        exit(1)
    
    print(f"✓ Successfully captured {len(logs)} log entries during training")
    print(f"  Sample log keys: {list(logs[0].keys())[:5]}")
    
    # VALIDATION 3: Verify expected log structure
    assert any(log.get("layer_name") == "model_output" for log in logs), \
        "Missing model output logs"
    assert any(log.get("event_type") == "forward" for log in logs), \
        "Missing forward pass logs"
    
    engine.disable_logging()
    print("\n✅ TensorFlow callback integration validated successfully!")
    print("   M-TRACE is Section 2.1.2 compliant (Keras callback pattern)")
    return True

if __name__ == "__main__":
    try:
        test_tensorflow_callback()
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)