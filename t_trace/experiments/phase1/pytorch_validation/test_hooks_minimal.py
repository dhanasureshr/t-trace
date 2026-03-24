#!/usr/bin/env python3
"""
Minimal PyTorch Hook Validation - MEMORY SAFE (10 iterations)
Based on M-TRACE Implementation v4
FIXED: Device handling, API calls, cleanup
"""
import os
import sys
import time
import json
import torch
import torchvision
import pynvml
from pathlib import Path

# === FIX 1: Correct path for your project structure ===
# Script location: t_trace/experiments/phase2/exp1/
# Project root: ~/Documents/Ai/mtrace/t-trace/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Import AFTER path setup
from t_trace.logging_engine import enable_logging

# Initialize monitoring
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_vram_gb():
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / 1024**3

def profile_minimal():
    print("="*70)
    print("M-TRACE PHASE 1 - MINIMAL HOOK VALIDATION")
    # Safe GPU name retrieval (handles both str and bytes)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f"GPU: {gpu_name}")
    print("⚠️  Using ONLY 10 iterations to prevent OOM")
    print("="*70)
    
    # === FIX 5: Verify config.yml ===
    config_path = Path("config.yml")
    if not config_path.exists():
        print(f"⚠️  config.yml not found at {config_path.resolve()}")
        print(f"   LoggingEngine will use defaults")
    else:
        print(f"✓ config.yml found at {config_path.resolve()}")
    
    # === FIX 2: Explicit device handling (CRITICAL) ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n▶ Device: {device}")
    
    # Load model
    model = torchvision.models.resnet50(weights=None).to(device).eval()
    dummy_input = torch.randn(8, 3, 224, 224).to(device)  # Small batch
    print(f"   ✓ Model and input on device: {device}")
    
    # Baseline inference
    print("\n▶ Baseline inference (no logging)...")
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    baseline_vram = get_vram_gb()
    print(f"   VRAM baseline: {baseline_vram:.2f} GB")
    
    # Enable M-TRACE logging
    print("\n▶ Enabling M-TRACE logging (development mode)...")
    engine = enable_logging(model, mode="development")
    # === FIX 3: Use public API ===
    print(f"   ✓ Logging enabled (run_id: {engine.get_run_id()[:8]}...)")
    
    # Single inference to trigger hooks
    print("\n▶ First inference WITH logging...")
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    first_latency = (time.perf_counter() - start) * 1000
    after_vram = get_vram_gb()
    print(f"   Latency: {first_latency:.2f} ms")
    print(f"   VRAM after: {after_vram:.2f} GB (+{after_vram - baseline_vram:.2f} GB)")
    
    # Small batch of inferences (10 total)
    print("\n▶ Running 10 iterations WITH logging...")
    start = time.perf_counter()
    for i in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
        if i % 5 == 0:
            vram = get_vram_gb()
            print(f"   Iter {i}/10 | VRAM: {vram:.2f} GB")
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    # Collect logs
    logs = engine.collect_logs() if hasattr(engine, "collect_logs") else []
    print(f"\n✅ Logs captured: {len(logs)}")
    
    # === FIX 3: Safe log access ===
    if logs and len(logs) > 0:
        print(f"   Sample log keys: {list(logs[0].keys())[:4]}")
        first_log = logs[0]
        if "internal_states" in first_log:
            print(f"   First log layer: {first_log['internal_states'].get('layer_name', 'N/A')}")
        else:
            print(f"   ⚠️  No 'internal_states' in log")
    else:
        print(f"   ⚠️  No logs captured")
    
    # Check log files on disk
    log_dir = Path("mtrace_logs/development")
    if log_dir.exists():
        parquet_files = list(log_dir.glob("*.parquet"))
        print(f"✅ Parquet files on disk: {len(parquet_files)}")
        if parquet_files:
            print(f"   Latest: {parquet_files[-1].name}")
    else:
        print(f"⚠️  Log directory missing: {log_dir}")
    
    # === FIX 4: Proper cleanup (use public API) ===
    print("\n▶ Disabling logging (auto-flush)...")
    if hasattr(engine, "disable_logging"):
        engine.disable_logging()
        time.sleep(1.0)  # Allow async writer to complete
    
    pynvml.nvmlShutdown()
    print("\n" + "="*70)
    print("✅ MINIMAL VALIDATION COMPLETE")
    print("   → If no crash occurred, hooks are attaching successfully")
    print("   → Next: Check mtrace_logs/development/ for Parquet files")
    print("="*70)

if __name__ == "__main__":
    try:
        profile_minimal()
    except Exception as e:
        print(f"\n❌ CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Run from TERMINAL (not VS Code debugger)")
        print("   2. Check mtrace_logs/development/ exists")
        print("   3. Verify config.yml has sparse_logging.enabled: true")
        pynvml.nvmlShutdown()