"""Minimal test to verify StorageEngine integration."""
import torch
import torch.nn as nn
import os
from pathlib import Path
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)


from t_trace.logging_engine import enable_logging

# Simple test model
model = nn.Sequential(nn.Linear(10, 5))

# Enable logging
engine = enable_logging(model, mode="development")
print(f"✓ Logging enabled (run_id: {engine.get_run_id()})")

# Execute forward pass
x = torch.randn(4, 10, requires_grad=True)
output = model(x)
output.sum().backward()

# Force immediate flush (bypasses async writer)
engine._flush_buffer()

# Check for Parquet files
log_dir = Path("mtrace_logs/development")
parquet_files = list(log_dir.glob("*.parquet")) if log_dir.exists() else []

print(f"\nStorage directory: {log_dir.absolute()}")
print(f"Parquet files found: {len(parquet_files)}")

if parquet_files:
    latest = sorted(parquet_files, key=lambda f: f.stat().st_mtime)[-1]
    print(f"✓ Latest log: {latest.name} ({latest.stat().st_size / 1024:.2f} KB)")
    print("\n✅ SUCCESS: Logs are being written to storage!")
else:
    print("❌ FAILURE: No Parquet files created")
    print("\nDebugging checklist:")
    print("1. Check t_trace/logging_engine/base.py:_flush_buffer() contains ACTUAL storage call")
    print("2. Verify no 'TODO' or commented-out code in _flush_buffer()")
    print("3. Check logs for 'Storage write FAILED' errors")
    print("4. Ensure mtrace_logs/ directory is writable:")
    print(f"   $ ls -la {log_dir.parent.absolute()}")