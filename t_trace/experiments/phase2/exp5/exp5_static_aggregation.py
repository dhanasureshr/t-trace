import torch
import shap
import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from t_trace.logging_engine import enable_logging


def run_static_task_experiment():
    print(">>> Running Experiment 5: Static Aggregation Boundary Test")
    
    # 1. Load Simple Tabular Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Inject Spurious Correlation (e.g., feature 0 is now perfectly predictive but wrong context)
    X[:, 0] = y * 100 
    
    # 2. Train Shallow Model (No deep reasoning required)
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    model.fit(X, y)
    
    # 3. Setup M-TRACE (Development Mode)
    # Note: Overhead will be visible here, but utility should be low
    engine = enable_logging(model, mode="development")
    _ = model.predict(X[:10]) # Trigger logging
    logs = engine.collect_logs()
    
    # 4. Setup SHAP
    explainer = shap.Explainer(model.predict, X[:100])
    start_shap = time.time()
    shap_values = explainer(X[:10])
    time_shap = time.time() - start_shap
    
    # 5. Simulate Diagnostic Utility
    # Heuristic: If the top feature in SHAP matches the injected spurious feature, 
    # SHAP solved it instantly. M-TRACE requires traversing layers.
    
    shap_top_feature = np.abs(shap_values.values).mean(axis=0).argmax()
    mtrace_diagnosis_time = len(logs) * 0.5 # Simulated time to scan layers (seconds)
    
    results = {
        "task": "Tabular Static",
        "shap_time_sec": time_shap,
        "mtrace_scan_time_sec": mtrace_diagnosis_time,
        "shap_found_root_cause": (shap_top_feature == 0),
        "temporal_redundancy_index": calculate_tri(shap_time=time_shap, mtrace_time=mtrace_diagnosis_time)
    }
    
    return results

def calculate_tri(shap_time, mtrace_time):
    # If M-TRACE takes longer and provides same info, TRI is high (bad)
    if mtrace_time > shap_time:
        return (mtrace_time - shap_time) / mtrace_time
    return 0.0

if __name__ == "__main__":
    res = run_static_task_experiment()
    print(f"Results: {res}")
    # Save to JSON for LaTeX table generation