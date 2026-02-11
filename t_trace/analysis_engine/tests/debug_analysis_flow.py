"""Debug script to reproduce exact dashboard analysis flow."""
import sys
import logging
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
from t_trace.analysis_engine.visualizations import Visualizations
import traceback

print("=== Loading DataLoader ===")
loader = DataLoader(storage_config={"directory": "mtrace_logs"})

print(f"\n=== Available Runs ===")
runs = loader.list_runs()
for run in runs[:3]:
    print(f"  - {run['run_id'][:8]} | {run['model_type']} | {run['size_bytes']/1024:.1f} KB")

if not runs:
    print("❌ No runs found - check storage directory")
    sys.exit(1)

# Use first run for testing
selected_run_id = runs[0]["run_id"]
print(f"\n=== Loading Logs for Run: {selected_run_id[:8]} ===")

df = loader.load_run_logs(selected_run_id)
if df is None or df.empty:
    print("❌ No logs loaded")
    sys.exit(1)

print(f"✓ Loaded {len(df)} logs")
print(f"\n=== DataFrame Info ===")
print(f"Columns: {df.columns.tolist()}")
print(f"\n=== Sample Log Structure ===")
sample_log = df.iloc[0].to_dict()
print(f"Keys: {list(sample_log.keys())}")

# Try to extract internal_states
if "internal_states" in sample_log:
    print(f"\ninternal_states keys: {list(sample_log['internal_states'].keys())}")
    print(f"Sample output: {sample_log['internal_states'].get('output', 'N/A')}")
    print(f"Sample losses: {sample_log['internal_states'].get('losses', 'N/A')}")

print("\n=== Testing Visualizations ===")

try:
    print("1. Creating loss curve...")
    viz = Visualizations()
    loss_fig = viz.create_loss_curve(df)
    print(f"   ✓ Loss curve created: {type(loss_fig)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

try:
    print("\n2. Creating layer activations chart...")
    activations_fig = viz.create_layer_activations_chart(df)
    print(f"   ✓ Activations chart created: {type(activations_fig)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

try:
    print("\n3. Creating gradient norm chart...")
    gradient_fig = viz.create_gradient_norm_chart(df)
    print(f"   ✓ Gradient chart created: {type(gradient_fig)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n=== Analysis Complete ===")