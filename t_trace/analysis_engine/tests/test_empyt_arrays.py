"""Test visualizations with empty arrays."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)

"""Test visualizations with empty arrays - FIXED VERSION."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from t_trace.analysis_engine.visualizations import Visualizations

# Test 1: Empty attention weights (NumPy array)
print("Test 1: Empty attention weights (NumPy array)")
fig = Visualizations.create_attention_heatmap(np.array([]))
assert fig is not None, "Should not crash on empty NumPy array"
print("✓ Passed")

# Test 2: Empty list
print("Test 2: Empty list")
fig = Visualizations.create_feature_importance_chart([])
assert fig is not None, "Should not crash on empty list"
print("✓ Passed")

# Test 3: DataFrame with empty activations
print("Test 3: DataFrame with empty activations")
df = pd.DataFrame([{
    "model_metadata": {"timestamp": 1234567890000},
    "internal_states": {
        "layer_name": "layer_0",
        "output": np.array([])  # Empty NumPy array
    },
    "event_type": "forward"
}])
fig = Visualizations.create_layer_activations_chart(df)
assert fig is not None, "Should not crash on empty activations"
print("✓ Passed")

# Test 4: DataFrame with None loss
print("Test 4: DataFrame with None loss")
df = pd.DataFrame([{
    "model_metadata": {"timestamp": 1234567890000},
    "internal_states": {"layer_name": "layer_0", "losses": None},
    "event_type": "forward"
}])
fig = Visualizations.create_loss_curve(df)
assert fig is not None, "Should not crash on None loss"
print("✓ Passed")

print("\n✅ All empty array tests passed with explicit checks!")