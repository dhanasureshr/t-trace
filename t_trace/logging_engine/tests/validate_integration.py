"""Validate LoggingEngine → StorageEngine integration."""
import torch
import torch.nn as nn
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
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Enable logging in development mode
model = TestModel()
engine = enable_logging(model, mode="development")
print(f"✓ Logging enabled (run_id: {engine.get_run_id()})")

# Execute forward pass
x = torch.randn(4, 10)
output = model(x)
print(f"✓ Forward pass: input {x.shape} → output {output.shape}")

# Execute backward pass (development mode only)
output.sum().backward()
print(f"✓ Backward pass completed")

# Explicit flush to storage (triggers _flush_buffer)
engine._flush_buffer()
print("✓ Logs flushed to storage")

# Verify Parquet file exists
log_dir = Path("mtrace_logs/development")
parquet_files = sorted(
    log_dir.glob("*.parquet"), 
    key=lambda f: f.stat().st_mtime
) if log_dir.exists() else []

if parquet_files:
    latest = parquet_files[-1]
    print(f"✓ Log file created: {latest.name} ({latest.stat().st_size / 1024:.2f} KB)")
    print(f"\n✅ FULL INTEGRATION WORKING: LoggingEngine → StorageEngine")
    print(f"   Logs stored at: {latest.absolute()}")
else:
    print(f"❌ No Parquet files found in {log_dir.absolute()}")
    print("   Possible causes:")
    print("   1. StorageEngine integration not properly implemented in _flush_buffer()")
    print("   2. Permission issues writing to mtrace_logs/")
    print("   3. Schema validation failure during write")