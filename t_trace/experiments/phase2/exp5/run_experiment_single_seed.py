"""
Single-seed version of Experiment 5 for statistical rigor.
Call this script with --seed argument for each of the 5 seeds.

Usage:
    python run_experiment_single_seed.py --seed 42
    python run_experiment_single_seed.py --seed 123
    # ... repeat for all 5 seeds
"""

import os
import sys
import time
import json
import uuid
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import tracemalloc
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    from t_trace.logging_engine import enable_logging
    from t_trace.storage_engine import get_storage_engine
except ImportError as e:
    print(f"ERROR: Could not import M-TRACE modules.\n{e}")
    sys.exit(1)

from t_trace.experiments.phase2.exp5.statistical_analysis import StatisticalRigor

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "dataset": "breast_cancer",  # Options: "breast_cancer", "digits"
    "model_type": "mlp",  # Options: "mlp", "random_forest"
    "spurious_feature_idx": 0,
    "spurious_correlation_strength": 1.0,  # 1.0 = perfect correlation
    "shap_samples": 100,  # Number of samples for SHAP baseline
    "shap_test_samples": 5,  # Number of samples to explain with SHAP
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_tri(mtrace_info_gain: float, shap_info_gain: float, 
                  mtrace_cost: float, shap_cost: float) -> float:
    """
    Calculate Temporal Redundancy Index (TRI).
    
    Formula: TRI = 1 - ((I_mtrace - I_shap) / Cost_mtrace)
    
    Where:
    - I = Information Gain (accuracy of human diagnosis)
    - Cost = Cognitive load (time to analyze) + Computational overhead
    
    Interpretation:
    - TRI ≈ 1.0: Temporal data is REDUNDANT (static task)
    - TRI ≈ 0.0: Temporal data provides EQUAL value
    - TRI < 0.0: Temporal data provides NEGATIVE value (overhead not justified)
    
    For Experiment 5 (static tasks), we EXPECT TRI ≈ 1.0
    """
    if mtrace_cost <= 0:
        return 1.0  # Avoid division by zero
    
    # Information gain difference (should be ~0 for static tasks)
    info_gain_diff = mtrace_info_gain - shap_info_gain
    
    # TRI calculation
    tri = 1.0 - (info_gain_diff / mtrace_cost)
    
    # Clamp to [0, 1] for interpretability
    tri = max(0.0, min(1.0, tri))
    
    return tri

def inject_spurious_correlation(X: np.ndarray, y: np.ndarray, 
                                feature_idx: int, strength: float = 1.0) -> np.ndarray:
    """
    Inject spurious correlation into dataset for debugging validation.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_idx: Index of feature to make spurious
        strength: Correlation strength (1.0 = perfect correlation)
    
    Returns:
        X with injected spurious correlation
    """
    X_spurious = X.copy()
    
    # Make feature_idx perfectly predictive of label
    X_spurious[:, feature_idx] = y * 100.0 * strength
    
    return X_spurious

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_single_seed_experiment(
    seed: int,
    dataset: str = "breast_cancer",
    model_type: str = "mlp",
    logs_dir: str = "mtrace_logs"
) -> Dict[str, Any]:
    """
    Run Experiment 5: Static Aggregation Boundary Test with REAL SHAP baseline.
    
    This validates the boundary condition where M-TRACE should show NO advantage
    over post-hoc tools (static tasks with no temporal reasoning).
    
    Args:
        seed: Random seed for reproducibility
        dataset: Dataset to use ("breast_cancer" or "digits")
        model_type: Model type ("mlp" or "random_forest")
        logs_dir: Directory for M-TRACE logs
    
    Returns:
        Dictionary with all metrics for this seed
    """
    print(">>> Running Experiment 5 (v2): Static Aggregation Boundary Test & Overhead Analysis")
    print("-" * 80)
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # ========================================================================
    # 1. DATASET & ERROR INJECTION
    # ========================================================================
    print("\n[Step 1] Loading Dataset...")
    
    if dataset == "breast_cancer":
        data = load_breast_cancer()
        task_description = "Binary Classification (Benign/Malignant)"
    elif dataset == "digits":
        data = load_digits()
        task_description = "Multi-class Classification (Digits 0-9)"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    X, y = data.data, data.target
    
    # Inject Spurious Correlation for debugging validation
    print(f"  Injecting spurious correlation at feature {CONFIG['spurious_feature_idx']}...")
    X = inject_spurious_correlation(
        X, y, 
        feature_idx=CONFIG["spurious_feature_idx"],
        strength=CONFIG["spurious_correlation_strength"]
    )
    
    print(f"  Dataset: {dataset.replace('_', ' ').title()}")
    print(f"  Task: {task_description}")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # ========================================================================
    # 2. DEFINE MODEL ARCHITECTURE
    # ========================================================================
    print("\n[Step 2] Initializing Model...")
    
    if model_type == "mlp":
        base_model_architecture = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=200,
            random_state=seed,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1
        )
        model_description = "MLP (2 hidden layers: 16, 8 neurons)"
    elif model_type == "random_forest":
        base_model_architecture = RandomForestClassifier(
            n_estimators=10,
            max_depth=4,
            random_state=seed,
            n_jobs=-1
        )
        model_description = "Random Forest (10 trees, max_depth=4)"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"  Model: {model_description}")
    
    # ========================================================================
    # 3. GROUP A: M-TRACE ANALYSIS
    # ========================================================================
    print("\n[Group A] Running M-TRACE Logging...")
    
    # Start Memory Tracking
    tracemalloc.start()
    
    exp_run_id = f"exp5_static_{uuid.uuid4().hex[:8]}"
    
    # Enable M-TRACE logging
    engine = enable_logging(base_model_architecture, mode="development", config_path=None)
    
    # Get wrapped model
    if hasattr(engine, 'get_wrapped_model'):
        wrapped_model = engine.get_wrapped_model()
    else:
        raise RuntimeError("Failed to retrieve wrapped model.")
    
    # Train wrapped model (captures fit logs)
    print("  Training WRAPPED model (captures fit logs)...")
    train_start = time.perf_counter()
    wrapped_model.fit(X, y)
    mtrace_train_time = time.perf_counter() - train_start
    
    # Run inference (captures predict logs)
    print("  Running inference on WRAPPED model (captures predict logs)...")
    infer_start = time.perf_counter()
    preds = wrapped_model.predict(X[:10])
    mtrace_infer_time = time.perf_counter() - infer_start
    
    # Collect logs
    logs = engine.collect_logs()
    engine.disable_logging()
    
    # Get Memory Usage
    current, peak = tracemalloc.get_traced_memory()
    mtrace_peak_memory_kb = peak / 1024
    tracemalloc.stop()
    
    # Save logs to disk
    storage_dir = Path(logs_dir) / "experiment_5"
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    storage_engine = get_storage_engine(backend="local", config={"storage_dir": str(storage_dir)})
    storage_engine.initialize()
    
    save_start = time.perf_counter()
    filepath = storage_engine.save_logs(
        logs=logs,
        run_id=exp_run_id,
        model_type=model_type,
        mode="development"
    )
    mtrace_save_time = time.perf_counter() - save_start
    
    # Parse logs
    parse_start = time.perf_counter()
    table = pq.read_table(filepath)
    df_mtrace = table.to_pandas()
    mtrace_parse_time = time.perf_counter() - parse_start
    
    # M-TRACE Diagnostic Utility (Did it find the spurious feature?)
    mtrace_found_cause = False
    if not df_mtrace.empty:
        try:
            # Check if spurious feature is ranked highest in feature_importance
            fi_col = df_mtrace['internal_states'].apply(
                lambda x: x.get('feature_importance', []) if isinstance(x, dict) else []
            )
            if len(fi_col) > 0 and len(fi_col.iloc[0]) > 0:
                if np.argmax(fi_col.iloc[0]) == CONFIG["spurious_feature_idx"]:
                    mtrace_found_cause = True
        except Exception as e:
            print(f"  Warning: Could not parse feature importance: {e}")
    
    # Calculate M-TRACE Information Gain (proxy: found root cause)
    mtrace_info_gain = 1.0 if mtrace_found_cause else 0.0
    
    # M-TRACE Total Cost (time + complexity)
    mtrace_total_time = mtrace_infer_time + mtrace_save_time + mtrace_parse_time
    mtrace_cost = mtrace_total_time * 1000  # Convert to ms
    
    print(f"  M-TRACE Training Time: {mtrace_train_time:.2f}s")
    print(f"  M-TRACE Inference Time: {mtrace_infer_time*1000:.2f}ms")
    print(f"  M-TRACE Save Time: {mtrace_save_time*1000:.2f}ms")
    print(f"  M-TRACE Parse Time: {mtrace_parse_time*1000:.2f}ms")
    print(f"  M-TRACE Peak Memory: {mtrace_peak_memory_kb:.2f} KB")
    print(f"  M-TRACE Found Root Cause: {mtrace_found_cause}")
    
    # ========================================================================
    # 4. GROUP B: SHAP ANALYSIS (REAL IMPLEMENTATION)
    # ========================================================================
    print("\n[Group B] Running SHAP Baseline...")
    
    tracemalloc.start()
    
    # Use the SAME wrapped model to ensure fair comparison
    # SHAP requires a callable predictor function
    def model_predict(X_input):
        return wrapped_model.predict(X_input)
    
    # Initialize SHAP explainer
    print("  Initializing SHAP explainer...")
    try:
        if model_type == "random_forest":
            explainer = shap.TreeExplainer(wrapped_model)
        else:
            # For MLP, use KernelExplainer (more general but slower)
            # Use a subset of training data for background
            background = shap.sample(X, CONFIG["shap_samples"])
            explainer = shap.KernelExplainer(model_predict, background)
    except Exception as e:
        print(f"  Warning: SHAP initialization failed: {e}")
        explainer = None
    
    # Run SHAP on test samples
    shap_values = None
    if explainer is not None:
        print(f"  Computing SHAP values for {CONFIG['shap_test_samples']} samples...")
        shap_start = time.perf_counter()
        try:
            shap_values = explainer.shap_values(X[:CONFIG["shap_test_samples"]])
        except Exception as e:
            print(f"  Warning: SHAP computation failed: {e}")
            shap_values = None
        shap_total_time = time.perf_counter() - shap_start
    else:
        shap_total_time = 0.0
    
    # Get Memory Usage
    current, peak = tracemalloc.get_traced_memory()
    shap_peak_memory_kb = peak / 1024
    tracemalloc.stop()
    
    # SHAP Diagnostic Utility (Did it find the spurious feature?)
    shap_found_cause = False
    if shap_values is not None:
        try:
            # For binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[0]).mean(axis=0)
            else:
                shap_vals = np.abs(shap_values).mean(axis=0)
            
            if np.argmax(shap_vals) == CONFIG["spurious_feature_idx"]:
                shap_found_cause = True
        except Exception as e:
            print(f"  Warning: Could not parse SHAP values: {e}")
    
    # Calculate SHAP Information Gain (proxy: found root cause)
    shap_info_gain = 1.0 if shap_found_cause else 0.0
    
    # SHAP Total Cost
    shap_cost = shap_total_time * 1000  # Convert to ms
    
    print(f"  SHAP Total Time: {shap_total_time*1000:.2f}ms")
    print(f"  SHAP Peak Memory: {shap_peak_memory_kb:.2f} KB")
    print(f"  SHAP Found Root Cause: {shap_found_cause}")
    
    # ========================================================================
    # 5. COMPARATIVE METRICS (TRI CALCULATION)
    # ========================================================================
    print("\n[Step 5] Calculating Comparative Metrics...")
    
    # Calculate Temporal Redundancy Index (TRI)
    tri = calculate_tri(
        mtrace_info_gain=mtrace_info_gain,
        shap_info_gain=shap_info_gain,
        mtrace_cost=mtrace_cost,
        shap_cost=shap_cost
    )
    
    # Cost Ratio (M-TRACE overhead vs SHAP)
    cost_ratio = mtrace_cost / max(1.0, shap_cost)
    
    # Determine conclusion based on TRI
    if tri > 0.8:
        conclusion = "TEMPORAL DATA REDUNDANT (Expected for static tasks)"
    elif tri > 0.5:
        conclusion = "TEMPORAL DATA MARGINALLY USEFUL"
    else:
        conclusion = "TEMPORAL DATA VALUABLE (Unexpected for static task)"
    
    print(f"  M-TRACE Information Gain: {mtrace_info_gain:.4f}")
    print(f"  SHAP Information Gain: {shap_info_gain:.4f}")
    print(f"  M-TRACE Cost: {mtrace_cost:.2f}ms")
    print(f"  SHAP Cost: {shap_cost:.2f}ms")
    print(f"  Temporal Redundancy Index (TRI): {tri:.4f}")
    print(f"  Cost Ratio (M-TRACE/SHAP): {cost_ratio:.2f}")
    print(f"  Conclusion: {conclusion}")
    
    # ========================================================================
    # 6. RETURN RESULTS
    # ========================================================================
    results = {
        "seed": seed,
        "experiment_id": exp_run_id,
        "timestamp": time.time(),
        "task_type": f"Static Aggregation ({dataset.replace('_', ' ').title()})",
        "model_type": model_type,
        "metrics": {
            "mtrace": {
                "train_time_sec": round(mtrace_train_time, 4),
                "inference_latency_ms": round(mtrace_infer_time * 1000, 2),
                "disk_write_time_ms": round(mtrace_save_time * 1000, 2),
                "parse_load_time_ms": round(mtrace_parse_time * 1000, 2),
                "total_cost_ms": round(mtrace_cost, 2),
                "peak_memory_kb": round(mtrace_peak_memory_kb, 2),
                "found_root_cause": mtrace_found_cause,
                "information_gain": mtrace_info_gain,
                "logs_captured_count": len(logs)
            },
            "shap": {
                "total_analysis_time_ms": round(shap_total_time * 1000, 2),
                "peak_memory_kb": round(shap_peak_memory_kb, 2),
                "found_root_cause": shap_found_cause,
                "information_gain": shap_info_gain,
                "samples_explained": CONFIG["shap_test_samples"]
            },
            "comparative": {
                "temporal_redundancy_index": round(tri, 4),
                "cost_ratio": round(cost_ratio, 2),
                "information_gain_diff": round(mtrace_info_gain - shap_info_gain, 4),
                "conclusion": conclusion
            }
        },
        "hypothesis": "For static aggregation tasks, M-TRACE temporal data should be redundant (TRI ≈ 1.0)",
        "validation_status": "PASS" if tri > 0.8 else "NEEDS_INVESTIGATION"
    }
    
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 5 with single seed")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for this run")
    parser.add_argument("--dataset", type=str, default="breast_cancer",
                       choices=["breast_cancer", "digits"], help="Dataset to use")
    parser.add_argument("--model-type", type=str, default="mlp",
                       choices=["mlp", "random_forest"], help="Model type")
    parser.add_argument("--logs-dir", type=str, default="mtrace_logs",
                       help="Directory for M-TRACE logs")
    parser.add_argument("--results-dir", type=str,
                       default="t_trace/experiments/phase2/exp5/results",
                       help="Directory to save per-seed results (base, not /raw)")
    args = parser.parse_args()
    
    # Update config from args
    CONFIG["dataset"] = args.dataset
    CONFIG["model_type"] = args.model_type
    
    # Run experiment
    results = run_single_seed_experiment(
        seed=args.seed,
        dataset=args.dataset,
        model_type=args.model_type,
        logs_dir=args.logs_dir
    )
    
    # Extract key metrics for statistical rigor
    tri_score = results["metrics"]["comparative"]["temporal_redundancy_index"]
    mtrace_cost = results["metrics"]["mtrace"]["total_cost_ms"]
    shap_cost = results["metrics"]["shap"]["total_analysis_time_ms"]
    mtrace_gain = results["metrics"]["mtrace"]["information_gain"]
    shap_gain = results["metrics"]["shap"]["information_gain"]
    cost_ratio = results["metrics"]["comparative"]["cost_ratio"]
    
    # Save results using StatisticalRigor
    stats = StatisticalRigor(results_dir=Path(args.results_dir))
    stats.save_seed_result(
        seed=args.seed,
        tri_score=tri_score,
        mtrace_cost_ms=mtrace_cost,
        shap_cost_ms=shap_cost,
        mtrace_info_gain=mtrace_gain,
        shap_info_gain=shap_gain,
        cost_ratio=cost_ratio,
        additional_metrics=results
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ SEED {args.seed} COMPLETE")
    print(f"{'='*70}")
    print(f"Temporal Redundancy Index (TRI): {tri_score:.4f}")
    print(f"M-TRACE Cost: {mtrace_cost:.2f}ms")
    print(f"SHAP Cost: {shap_cost:.2f}ms")
    print(f"Cost Ratio (M-TRACE/SHAP): {cost_ratio:.2f}")
    print(f"Validation Status: {results['validation_status']}")
    print(f"Conclusion: {results['metrics']['comparative']['conclusion']}")
    print(f"Results saved to: {args.results_dir}/raw/experiment5_seed{args.seed}_results.json")
    print(f"{'='*70}\n")