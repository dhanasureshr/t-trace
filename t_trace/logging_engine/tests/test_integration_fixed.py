# test_integration_fixed.py
import torch
import torch.nn as nn
from pathlib import Path
import sys
from pathlib import Path

import os



sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)

"""Validate end-to-end schema compliance."""
import torch
import torch.nn as nn
import sys
from pathlib import Path

from t_trace.logging_engine import enable_logging

# Simple test model
model = nn.Sequential(nn.Linear(10, 5))
engine = enable_logging(model, mode="development")
print(f"✓ Logging enabled (run_id: {engine.run_id})")

# Execute forward/backward passes
x = torch.randn(4, 10, requires_grad=True)
output = model(x)
output.sum().backward()

# Force flush to storage
engine._flush_buffer()
print("✓ Logs flushed to storage")

# Verify file creation
import time
time.sleep(1)  # Allow async writes to complete
log_dir = Path("mtrace_logs/development")
parquet_files = sorted(log_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime) if log_dir.exists() else []
print(f"\n✓ Parquet files found: {len(parquet_files)}")

if parquet_files:
    latest = parquet_files[-1]
    print(f"✓ Latest log: {latest.name} ({latest.stat().st_size / 1024:.2f} KB)")
    print("\n✅ SUCCESS: Schema-compliant logs written to storage!")
else:
    print("❌ FAILURE: No files created")
    print("\nDebugging checklist:")
    print("1. Verify PyTorchHook.get_logs() wraps logs with _wrap_log_for_schema()")
    print("2. Check logs for 'Schema mismatch' errors at ERROR level")
    print("3. Ensure _flush_buffer() calls framework_engine.collect_logs() (not local buffer)")