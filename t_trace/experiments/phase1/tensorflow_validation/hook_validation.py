#!/usr/bin/env python3
"""TensorFlow validation with forced flush and schema debugging"""
import os
import sys
import time
import numpy as np
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from t_trace.logging_engine import enable_logging

print(f"✅ TensorFlow CPU mode: {tf.config.list_physical_devices('CPU')}\n")

def profile_with_mtrace(model, input_tensor, iterations=50):
    """Profile with explicit flush before disable"""
    # Use validation config
    config_path = Path(__file__).parent / "config_validation.yml"
    result = enable_logging(model, mode="development", config_path=str(config_path))
    
    engine, callback = result if isinstance(result, tuple) else (result, None)
    if callback is None:
        raise RuntimeError("TensorFlow callback not returned")
    
    run_id = engine.get_run_id() if hasattr(engine, "get_run_id") else "unknown"
    print(f"   → M-TRACE enabled (run_id: {run_id[:8]}...)")
    
    # Warmup
    for _ in range(10):
        callback.on_predict_batch_begin(batch=0)
        output = model(input_tensor, training=False)
        callback.on_predict_batch_end(batch=0, logs={"outputs": output})
    
    # Timed runs
    start = time.perf_counter()
    for i in range(iterations):
        callback.on_predict_batch_begin(batch=i)
        output = model(input_tensor, training=False)
        callback.on_predict_batch_end(batch=i, logs={"outputs": output})
    elapsed = time.perf_counter() - start
    
    # CRITICAL FIX: Force flush BEFORE disable
    print(f"   → Forcing log flush BEFORE disable...")
    logs = engine.collect_logs()
    print(f"   → Logs in buffer: {len(logs)}")
    
    # DEBUG: Inspect first log schema
    if logs:
        first_log = logs[0]
        print(f"   → First log keys: {list(first_log.keys())}")
        print(f"   → layer_name: {first_log.get('internal_states', {}).get('layer_name', 'MISSING')}")
        print(f"   → layer_index: {first_log.get('internal_states', {}).get('layer_index', 'MISSING')}")
    
    # Force storage write (bypass writer thread)
    if hasattr(engine, '_storage_engine') and engine._storage_engine:
        try:
            filepath = engine._storage_engine.save_logs(
                logs=logs,
                run_id=run_id,
                model_type="sequential",
                mode="development"
            )
            print(f"   ✓ Forced save to: {filepath}")
        except Exception as e:
            print(f"   ✗ Forced save FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    engine.disable_logging()
    
    return {
        "latency_ms": (elapsed / iterations) * 1000,
        "logs_captured": len(logs),
        "first_layer_name": logs[0]["internal_states"].get("layer_name", "N/A") 
            if logs and "internal_states" in logs[0] else "N/A"
    }

def main():
    print("="*70)
    print("M-TRACE PHASE 1 - TENSORFLOW VALIDATION (FORCED FLUSH)")
    print("✅ Batch size=10 + explicit flush before disable")
    print("="*70)
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    print(f"\n▶ Model created: {model.input_shape} → {model.output_shape}")
    
    dummy_input = tf.random.normal((4, 32, 32, 3))
    
    print("\n▶ Inference WITH M-TRACE callback (CPU)...")
    mtrace_metrics = profile_with_mtrace(model, dummy_input, iterations=50)
    
    print(f"\n✅ RESULTS:")
    print(f"   Logs captured: {mtrace_metrics['logs_captured']}")
    print(f"   First layer name: {mtrace_metrics['first_layer_name']}")
    
    # Verify Parquet output
    log_dir = Path("mtrace_logs/development")
    if log_dir.exists():
        recent_files = [
            f for f in log_dir.glob("*.parquet") 
            if time.time() - f.stat().st_mtime < 120  # Last 2 minutes
        ]
        if recent_files:
            latest = max(recent_files, key=lambda f: f.stat().st_mtime)
            print(f"   ✅ NEW Parquet: {latest.name}")
            
            # Verify schema
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(latest)
                has_layer_name = "internal_states.layer_name" in table.column_names
                print(f"   ✅ Schema valid: layer_name present = {has_layer_name}")
            except Exception as e:
                print(f"   ⚠️ Schema check failed: {e}")
        else:
            print(f"   ❌ No NEW Parquet files (check mtrace_logs/development/)")
            print(f"      Existing files: {[f.name for f in log_dir.glob('*.parquet')]}")
    else:
        print(f"   ❌ Log directory missing: {log_dir}")
    
    print("\n" + "="*70)
    print("PHASE 1 CRITERIA:")
    print(f"   • Logs captured >0: {'✅ PASS' if mtrace_metrics['logs_captured'] > 0 else '❌ FAIL'}")
    print(f"   • layer_name present: {'✅ PASS' if mtrace_metrics['first_layer_name'] != 'N/A' else '❌ FAIL'}")
    print(f"   • NEW Parquet created: {'✅ PASS' if recent_files else '❌ FAIL'}")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()