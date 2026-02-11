# test_directory_creation.py
from pathlib import Path

import os

import sys


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)
from t_trace.storage_engine import get_storage_engine

# Initialize storage engine
engine = get_storage_engine(backend="local", config={"storage_dir": "test_logs"})

# Create test log (minimal schema-compliant structure)
test_log = [{
    "model_metadata": {
        "model_type": "test",
        "framework": "pytorch",
        "timestamp": 1234567890000,
        "run_id": "test_run",
        "mode": "development",
        "model_architecture": {"num_layers": 1, "layer_types": ["linear"], "connections": ["sequential"]},
        "hyperparameters": {"learning_rate": 0.001, "batch_size": 32, "optimizer": "sgd", "other_params": {}},
        "layer_metadata": {"layer_type": "linear", "activation_function": "relu", "num_parameters": 1000}
    },
    "internal_states": {
        "layer_name": "layer_0",
        "layer_index": 0,
        "attention_weights": [0.5, 0.3, 0.2],
        "feature_maps": [],
        "node_splits": [],
        "gradients": [],
        "losses": 0.25,
        "feature_importance": [],
        "decision_paths": []
    },
    "event_type": "forward"
}]

# Save logs (should create directory + file)
filepath = engine.save_logs(test_log, run_id="test_run", model_type="test", mode="development")
print(f"✓ File created: {filepath}")
print(f"✓ Directory exists: {Path(filepath).parent.exists()}")
print(f"✓ File exists: {Path(filepath).exists()}")