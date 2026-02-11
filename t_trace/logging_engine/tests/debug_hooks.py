"""Debug script to verify hook attachment and log collection."""
import torch
import torch.nn as nn
import sys
from pathlib import Path

import os



sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)



from t_trace.logging_engine import enable_logging

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)

# Enable logging
model = TestModel()
engine = enable_logging(model, mode="development")
print(f"✓ Logging enabled (run_id: {engine.run_id})")

# Verify hooks were attached
fw_engine = engine._framework_engine
print(f"✓ Hooks attached to {len(fw_engine.hooks)} layers:")
for hook in fw_engine.hooks:
    print(f"  - {hook.layer_name} (index {hook.layer_index})")

# Execute forward/backward passes
x = torch.randn(4, 10, requires_grad=True)
output = model(x)
output.sum().backward()

# Check log collection BEFORE flush
logs = fw_engine.collect_logs()
print(f"\n✓ Collected {len(logs)} logs from framework engine")
if logs:
    print(f"  Sample log keys: {list(logs[0].keys())}")
    print(f"  First log layer: {logs[0].get('layer_name', 'N/A')}")

# Force flush to storage
engine._flush_buffer()

# Verify file creation
import time
time.sleep(1)  # Allow async writes to complete
log_dir = Path("mtrace_logs/development")
parquet_files = sorted(log_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime) if log_dir.exists() else []
print(f"\n✓ Parquet files in {log_dir}: {len(parquet_files)}")
if parquet_files:
    latest = parquet_files[-1]
    print(f"  Latest: {latest.name} ({latest.stat().st_size / 1024:.2f} KB)")
    print("\n✅ SUCCESS: Logs captured AND persisted to storage!")
else:
    print("❌ FAILURE: No files created - check _flush_buffer() implementation")