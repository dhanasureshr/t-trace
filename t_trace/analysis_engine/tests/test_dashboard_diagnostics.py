"""Test dashboard with comprehensive diagnostics."""
import logging
import sys
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
# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

from t_trace.analysis_engine.data_loader import DataLoader

# Test DataLoader directly first
print("=== Testing DataLoader Directly ===")
loader = DataLoader(storage_config={"directory": "mtrace_logs"})
print(f"Storage directory: {loader.storage_dir}")
print(f"Development dir exists: {(loader.storage_dir / 'development').exists()}")
print(f"Production dir exists: {(loader.storage_dir / 'production').exists()}")

runs = loader.list_runs()
print(f"\nFound {len(runs)} runs:")
for run in runs:
    print(f"  - {run['run_id'][:8]} | {run['model_type']} | {run['size_bytes']/1024:.1f} KB")

if not runs:
    print("\n⚠️  NO RUNS FOUND - Check:")
    print(f"  1. Directory exists: {loader.storage_dir}")
    print(f"  2. Files exist: {list(loader.storage_dir.glob('**/*.parquet'))}")
    import os
    print(f"  3. Current working directory: {os.getcwd()}")
else:
    print("\n✅ DataLoader working correctly - starting dashboard...")

# Start dashboard
from t_trace.analysis_engine import enable_analysis
enable_analysis(host="127.0.0.1", port=8050, debug=True)