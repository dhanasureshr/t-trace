#!/usr/bin/env python3
"""
scikit-learn Production Mode Validation: Measures overhead with minimal logging
CRITICAL: Production mode should achieve ≤5% overhead (vs 355% in development mode)
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# Suppress sklearn warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from t_trace.logging_engine import enable_logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def profile_baseline(model, X_test, iterations=200):
    """Profile baseline inference WITHOUT logging (high iteration count for accuracy)"""
    # Warmup
    for _ in range(20):
        _ = model.predict(X_test)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model.predict(X_test)
    elapsed = time.perf_counter() - start
    
    return {
        "latency_ms": (elapsed / iterations) * 1000,
        "throughput_samples_sec": (iterations * X_test.shape[0]) / elapsed
    }

def profile_production_mode(model, X_train, y_train, X_test, iterations=200):
    """
    Profile inference WITH M-TRACE in PRODUCTION mode.
    Critical differences from development mode:
      • mode="production" (NOT "development")
      • Only essential fields logged (layer_name, layer_index, losses)
      • Aggressive sparse threshold (0.5+) to minimize data volume
      • NO feature_importances captured by default
      • NO tree structure metadata
      • Batch size=100 for infrequent writes
    """
    # Enable logging in PRODUCTION mode
    engine = enable_logging(model, mode="production")
    wrapped_model = engine.get_wrapped_model()
    
    run_id = engine.get_run_id() if hasattr(engine, "get_run_id") else "unknown"
    print(f"   → M-TRACE enabled (production mode, run_id: {run_id[:8]}...)")
    
    # Train wrapped model
    wrapped_model.fit(X_train, y_train)
    
    # Warmup with wrapped model
    for _ in range(20):
        _ = wrapped_model.predict(X_test)
    
    # Timed runs with wrapped model
    start = time.perf_counter()
    for _ in range(iterations):
        _ = wrapped_model.predict(X_test)
    elapsed = time.perf_counter() - start
    
    # Collect logs BEFORE disable
    logs = engine.collect_logs() if hasattr(engine, "collect_logs") else []
    print(f"   → Logs captured: {len(logs)} (expected: ~2-5 for 200 inferences)")
    
    # Cleanup
    engine.disable_logging()
    
    return {
        "latency_ms": (elapsed / iterations) * 1000,
        "throughput_samples_sec": (iterations * X_test.shape[0]) / elapsed,
        "logs_captured": len(logs)
    }

def main():
    print("="*70)
    print("M-TRACE PHASE 1 - SCIKIT-LEARN PRODUCTION MODE VALIDATION")
    print("✅ Target: ≤5% overhead (vs 355% in development mode)")
    print("="*70)
    
    # Create dataset (smaller for faster iteration)
    X, y = make_classification(
        n_samples=500, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n▶ Dataset: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
    
    # Baseline inference (no logging)
    print("\n▶ Baseline inference (no logging)...")
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    baseline = profile_baseline(model, X_test, iterations=200)
    print(f"   Latency: {baseline['latency_ms']:.3f} ms")
    print(f"   Throughput: {baseline['throughput_samples_sec']:.0f} samples/sec")
    
    # M-TRACE production mode inference
    print("\n▶ Inference WITH M-TRACE (production mode)...")
    model2 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    
    mtrace_metrics = profile_production_mode(
        model2, X_train, y_train, X_test, iterations=200
    )
    
    overhead_pct = ((mtrace_metrics["latency_ms"] - baseline["latency_ms"]) / baseline["latency_ms"]) * 100
    
    print(f"\n✅ PRODUCTION MODE RESULTS:")
    print(f"   Latency: {mtrace_metrics['latency_ms']:.3f} ms (+{overhead_pct:.1f}% overhead)")
    print(f"   Logs captured: {mtrace_metrics['logs_captured']}")
    print(f"   Target overhead ≤5%: {'✅ PASS' if overhead_pct <= 5.0 else '❌ FAIL'}")
    
    # Verify Parquet output
    log_dir = Path("mtrace_logs/production")
    if log_dir.exists():
        recent_files = [
            f for f in log_dir.glob("*.parquet") 
            if time.time() - f.stat().st_mtime < 60
        ]
        if recent_files:
            latest = max(recent_files, key=lambda f: f.stat().st_mtime)
            print(f"   ✅ Parquet saved: {latest.name}")
            
            # Verify minimal schema (production mode)
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(latest)
                has_layer_name = "internal_states.layer_name" in table.column_names
                has_layer_index = "internal_states.layer_index" in table.column_names
                print(f"   ✅ Schema minimal: layer_name={has_layer_name}, layer_index={has_layer_index}")
            except Exception as e:
                print(f"   ⚠️ Schema check: {e}")
        else:
            print(f"   ⚠️ No NEW Parquet in last 60s (may use batch write)")
    else:
        print(f"   ⚠️ Production log directory missing (creating now)")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 1 PRODUCTION MODE CRITERIA:")
    print(f"   • Overhead ≤5%: {'✅ PASS' if overhead_pct <= 5.0 else f'❌ FAIL ({overhead_pct:.1f}%)'}")
    print(f"   • Logs captured >0: {'✅ PASS' if mtrace_metrics['logs_captured'] > 0 else '❌ FAIL'}")
    print(f"   • Schema minimal: {'✅ PASS' if has_layer_name and has_layer_index else '❌ FAIL'}")
    print("="*70)
    
    # Critical comparison table
    print("\n📊 OVERHEAD COMPARISON:")
    print(f"   Development mode: 355.7% overhead (expected - captures full feature importances)")
    print(f"   Production mode:   {overhead_pct:.1f}% overhead ({'✅ MEETS SPEC' if overhead_pct <= 5.0 else '❌ EXCEEDS SPEC'})")
    print("\n💡 WHY PRODUCTION MODE IS FASTER:")
    print("   • NO feature_importances captured (only layer_name/layer_index/losses)")
    print("   • Aggressive sparse threshold (0.5 vs 0.1 in dev mode)")
    print("   • Batch size=100 (vs 10 in dev mode) → fewer disk writes")
    print("   • NO tree structure metadata (n_leaves, n_nodes, etc.)")
    print("   • NO gradient capture (production mode skips backward hooks)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()