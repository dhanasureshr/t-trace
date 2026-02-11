import sys
from pathlib import Path
import os



sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)

from t_trace.analysis_engine.data_loader import DataLoader

loader = DataLoader(storage_config={"directory": "mtrace_logs"})
runs = loader.list_runs()

print(f"Found {len(runs)} runs:")
for run in runs[:3]:  # Show first 3
    print(f"  - {run['run_id']} | {run['model_type']} | {run['timestamp']} | {run['size_bytes']/1024:.1f} KB")