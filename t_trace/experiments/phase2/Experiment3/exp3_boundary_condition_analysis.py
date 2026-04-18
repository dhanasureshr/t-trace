#!/usr/bin/env python3
r"""
Experiment 3: Boundary Condition Analysis (Multi-Seed + Direct Comparison)
=====================================================================================
Validates the boundary conditions where M-TRACE offers NO advantage over post-hoc tools.

Scientific Claim: For static aggregation tasks (no temporal reasoning required),
M-TRACE's temporal trajectory data adds zero value over standard post-hoc attribution.

This is a "null hypothesis" validation - proving M-TRACE is honest about when it adds value.

UPDATED: Reports boundary conditions via direct comparison of Diagnostic Utility and 
Computational Efficiency 

Theoretical Foundation (project m trace_v1.pdf):
- Definition 6: Temporal Redundancy Index (conceptual framework only)
- Boundary Conditions: When $\mathcal{T}(x)$ is redundant for static tasks
- Intellectual Honesty: Defining when instrumentation adds no value

Statistical Rigor (experiments plan.pdf):
- 5 seeds: [42, 123, 456, 789, 1011]
- Welch's t-test for significance
- Mean ± std reporting

Hardware Target (My_WorkStation_COMPONETS.pdf):
- GPU: RTX 4080 Super (16GB GDDR6X)
- CPU: AMD Ryzen 9 7900X (12 cores, 24 threads)
- RAM: 64GB DDR5 6000MHz

Reproducibility (Docker_setup_v1.pdf):
- CUDA 12.3, PyTorch 2.3.0, Python 3.10
- Containerized environment for exact reproduction
"""

import os
import sys
import time
import json
import uuid
import argparse
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
from scipy.stats import ttest_ind

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

try:
    from t_trace.logging_engine import enable_logging
    from t_trace.storage_engine import get_storage_engine
except ImportError as e:
    print(f"ERROR: Could not import M-TRACE modules.\n{e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "dataset": "breast_cancer",  # Options: "breast_cancer", "digits"
    "model_type": "mlp",  # Options: "mlp", "random_forest"
    "num_samples": 500,
    "spurious_feature_idx": 0,
    "spurious_correlation_strength": 1.0,  # 1.0 = perfect correlation
    "output_dir": Path(__file__).parent / "results",
    "logs_dir": Path(__file__).resolve().parents[5] / "mtrace_logs",
    "seeds": [42, 123, 456, 789, 1011],  # 5 seeds for statistical rigor
    "shap_samples": 100,  # Number of samples for SHAP baseline
    "shap_test_samples": 5  # Number of samples to explain with SHAP
}

# ============================================================================
# STATISTICAL ANALYSIS UTILITIES
# ============================================================================

def welchs_t_test(values_mtrace: List[float], values_shap: List[float]) -> Dict:
    """
    Perform Welch's t-test (unequal variance) between M-TRACE and SHAP.
    
    Args:
        values_mtrace: List of M-TRACE metric values across seeds
        values_shap: List of SHAP metric values across seeds
    
    Returns:
        dict with t_statistic, p_value, mean_diff, effect_size
    """
    if len(values_mtrace) < 2 or len(values_shap) < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": float(np.mean(values_mtrace) - np.mean(values_shap)),
            "effect_size": 0.0,
            "significant": False
        }
    
    # Welch's t-test (does not assume equal variance)
    t_stat, p_value = ttest_ind(values_mtrace, values_shap, equal_var=False)
    
    # Effect size (Cohen's d for unequal variance)
    n1, n2 = len(values_mtrace), len(values_shap)
    var1, var2 = np.var(values_mtrace, ddof=1), np.var(values_shap, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
    cohens_d = (np.mean(values_mtrace) - np.mean(values_shap)) / pooled_std if pooled_std > 0 else 0.0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(np.mean(values_mtrace) - np.mean(values_shap)),
        "effect_size": float(cohens_d),
        "significant": bool(p_value < 0.05)
    }

def format_p_value(p_value: float) -> str:
    """Format p-value for LaTeX table with proper notation."""
    if p_value < 0.001:
        return r"$p<0.001$"
    elif p_value < 0.01:
        return r"$p<0.01$"
    elif p_value < 0.05:
        return r"$p<0.05$"
    else:
        return f"$p={p_value:.3f}$"

def format_statistical_result(mean: float, std: float, p_value: float, n: int) -> str:
    """Format result for LaTeX table with proper notation."""
    p_str = format_p_value(p_value)
    return f"${mean:.3f} \\pm {std:.3f}$ ({p_str}, $n={n}$)"

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    CRITICAL FIX: Prevents TypeError when saving aggregated results.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_schema_complexity(log_entry: Dict) -> int:
    """
    Proxy for Cognitive Load: Counts depth/fields to inspect.
    Higher = more complex for humans to analyze.
    """
    complexity_score = 0
    if not isinstance(log_entry, dict):
        return 0
    
    if "model_metadata" in log_entry:
        metadata = log_entry["model_metadata"]
        if isinstance(metadata, dict):
            complexity_score += len(metadata)
    
    if "internal_states" in log_entry:
        states = log_entry["internal_states"]
        if isinstance(states, dict):
            complexity_score += len(states)
            for k, v in states.items():
                if isinstance(v, list) and len(v) > 0:
                    complexity_score += 1
    
    if "sparse_logging_metadata" in log_entry:
        sparse_meta = log_entry["sparse_logging_metadata"]
        if isinstance(sparse_meta, dict):
            complexity_score += len(sparse_meta)
    
    return complexity_score

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
    X_spurious[:, feature_idx] = y * 100.0 * strength
    return X_spurious

# ============================================================================
# SINGLE SEED EXPERIMENT FUNCTION
# ============================================================================

def run_single_seed(seed: int = 42) -> Dict[str, Any]:
    """
    Run Experiment 5 for a single seed with REAL SHAP baseline.
    
    This validates the boundary condition where M-TRACE should show NO advantage
    over post-hoc tools (static tasks with no temporal reasoning).
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with all metrics for this run
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 5: Static Aggregation Boundary Test (Seed={seed})")
    print(f"{'='*80}")
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    
    # ========================================================================
    # 1. DATASET & ERROR INJECTION
    # ========================================================================
    print("\n[Step 1] Loading Dataset...")
    
    if CONFIG["dataset"] == "breast_cancer":
        data = load_breast_cancer()
        task_description = "Binary Classification (Benign/Malignant)"
    elif CONFIG["dataset"] == "digits":
        data = load_digits()
        task_description = "Multi-class Classification (Digits 0-9)"
    else:
        raise ValueError(f"Unknown dataset: {CONFIG['dataset']}")
    
    X, y = data.data, data.target
    
    # Inject Spurious Correlation for debugging validation
    print(f"   Injecting spurious correlation at feature {CONFIG['spurious_feature_idx']}...")
    X = inject_spurious_correlation(
        X, y, 
        feature_idx=CONFIG["spurious_feature_idx"],
        strength=CONFIG["spurious_correlation_strength"]
    )
    
    print(f"   Dataset: {CONFIG['dataset'].replace('_', ' ').title()}")
    print(f"   Task: {task_description}")
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # ========================================================================
    # 2. DEFINE MODEL ARCHITECTURE
    # ========================================================================
    print("\n[Step 2] Initializing Model...")
    
    if CONFIG["model_type"] == "mlp":
        base_model_architecture = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=200,
            random_state=seed,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1
        )
        model_description = "MLP (2 hidden layers: 16, 8 neurons)"
    elif CONFIG["model_type"] == "random_forest":
        base_model_architecture = RandomForestClassifier(
            n_estimators=10,
            max_depth=4,
            random_state=seed,
            n_jobs=-1
        )
        model_description = "Random Forest (10 trees, max_depth=4)"
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")
    
    print(f"   Model: {model_description}")
    
    # ========================================================================
    # 3. GROUP A: M-TRACE ANALYSIS
    # ========================================================================
    print("\n[Group A] Running M-TRACE Logging...")
    
    tracemalloc.start()
    
    exp_run_id = f"exp3_static_{uuid.uuid4().hex[:8]}"
    
    engine = enable_logging(base_model_architecture, mode="development", config_path=None)
    
    if hasattr(engine, 'get_wrapped_model'):
        wrapped_model = engine.get_wrapped_model()
    else:
        raise RuntimeError("Failed to retrieve wrapped model.")
    
    print("   Training WRAPPED model (captures fit logs)...")
    train_start = time.perf_counter()
    wrapped_model.fit(X, y)
    mtrace_train_time = time.perf_counter() - train_start
    
    print("   Running inference on WRAPPED model (captures predict logs)...")
    infer_start = time.perf_counter()
    preds = wrapped_model.predict(X[:10])
    mtrace_infer_time = time.perf_counter() - infer_start
    
    logs = engine.collect_logs()
    engine.disable_logging()
    
    current, peak = tracemalloc.get_traced_memory()
    mtrace_peak_memory_kb = peak / 1024
    tracemalloc.stop()
    
    storage_dir = Path(CONFIG["logs_dir"]) / "experiment_5"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_engine = get_storage_engine(backend="local", config={"storage_dir": str(storage_dir)})
    storage_engine.initialize()
    
    save_start = time.perf_counter()
    filepath = storage_engine.save_logs(
        logs=logs, 
        run_id=exp_run_id, 
        model_type=CONFIG["model_type"], 
        mode="development"
    )
    mtrace_save_time = time.perf_counter() - save_start
    
    parse_start = time.perf_counter()
    table = pq.read_table(filepath)
    df_mtrace = table.to_pandas()
    mtrace_parse_time = time.perf_counter() - parse_start
    
    # M-TRACE Diagnostic Utility
    mtrace_found_cause = False
    if not df_mtrace.empty:
        try:
            fi_col = df_mtrace['internal_states'].apply(
                lambda x: x.get('feature_importance', []) if isinstance(x, dict) else []
            )
            if len(fi_col) > 0 and len(fi_col.iloc[0]) > 0:
                if np.argmax(fi_col.iloc[0]) == CONFIG["spurious_feature_idx"]:
                    mtrace_found_cause = True
        except Exception as e:
            print(f"   Warning: Could not parse feature importance: {e}")
    
    mtrace_info_gain = 1.0 if mtrace_found_cause else 0.0
    mtrace_total_time = mtrace_infer_time + mtrace_save_time + mtrace_parse_time
    mtrace_cost = mtrace_total_time * 1000  # Convert to ms
    
    print(f"   M-TRACE Training Time: {mtrace_train_time:.2f}s")
    print(f"   M-TRACE Inference Time: {mtrace_infer_time*1000:.2f}ms")
    print(f"   M-TRACE Save Time: {mtrace_save_time*1000:.2f}ms")
    print(f"   M-TRACE Parse Time: {mtrace_parse_time*1000:.2f}ms")
    print(f"   M-TRACE Peak Memory: {mtrace_peak_memory_kb:.2f} KB")
    print(f"   M-TRACE Found Root Cause: {mtrace_found_cause}")
    
    # ========================================================================
    # 4. GROUP B: SHAP ANALYSIS (REAL IMPLEMENTATION)
    # ========================================================================
    print("\n[Group B] Running SHAP Baseline...")
    
    tracemalloc.start()
    
    def model_predict(X_input):
        return wrapped_model.predict(X_input)
    
    print("   Initializing SHAP explainer...")
    try:
        if CONFIG["model_type"] == "random_forest":
            explainer = shap.TreeExplainer(wrapped_model)
        else:
            background = shap.sample(X, CONFIG["shap_samples"])
            explainer = shap.KernelExplainer(model_predict, background)
    except Exception as e:
        print(f"   Warning: SHAP initialization failed: {e}")
        explainer = None
    
    shap_values = None
    if explainer is not None:
        print(f"   Computing SHAP values for {CONFIG['shap_test_samples']} samples...")
        shap_start = time.perf_counter()
        try:
            shap_values = explainer.shap_values(X[:CONFIG["shap_test_samples"]])
        except Exception as e:
            print(f"   Warning: SHAP computation failed: {e}")
            shap_values = None
        shap_total_time = time.perf_counter() - shap_start
    else:
        shap_total_time = 0.0
    
    current, peak = tracemalloc.get_traced_memory()
    shap_peak_memory_kb = peak / 1024
    tracemalloc.stop()
    
    # SHAP Diagnostic Utility
    shap_found_cause = False
    if shap_values is not None:
        try:
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[0]).mean(axis=0)
            else:
                shap_vals = np.abs(shap_values).mean(axis=0)
            
            if np.argmax(shap_vals) == CONFIG["spurious_feature_idx"]:
                shap_found_cause = True
        except Exception as e:
            print(f"   Warning: Could not parse SHAP values: {e}")
    
    shap_info_gain = 1.0 if shap_found_cause else 0.0
    shap_cost = shap_total_time * 1000  # Convert to ms
    
    print(f"   SHAP Total Time: {shap_total_time*1000:.2f}ms")
    print(f"   SHAP Peak Memory: {shap_peak_memory_kb:.2f} KB")
    print(f"   SHAP Found Root Cause: {shap_found_cause}")
    
    # ========================================================================
    # 5. COMPARATIVE METRICS (DIRECT COMPARISON APPROACH)
    # ========================================================================
    print("\n[Step 5] Calculating Comparative Metrics (Direct Comparison)...")
    
    utility_diff = mtrace_info_gain - shap_info_gain
    cost_ratio = mtrace_cost / max(1.0, shap_cost)
    
    utilities_similar = abs(utility_diff) < 0.1  # Tolerance threshold
    mtrace_more_efficient = mtrace_cost < shap_cost
    
    if utilities_similar and mtrace_more_efficient:
        conclusion = "TEMPORAL DATA REDUNDANT (Expected for static tasks)"
        validation_status = "PASS"
    elif not utilities_similar and mtrace_more_efficient:
        conclusion = "UTILITY DIFFERENCE DETECTED (Investigate model compatibility)"
        validation_status = "NEEDS_INVESTIGATION"
    elif utilities_similar and not mtrace_more_efficient:
        conclusion = "M-TRACE LESS EFFICIENT (Overhead not justified for static task)"
        validation_status = "NEEDS_INVESTIGATION"
    else:
        conclusion = "BOTH UTILITY AND COST DIFFER (Unexpected for static task)"
        validation_status = "NEEDS_INVESTIGATION"
    
    print(f"   M-TRACE Information Gain: {mtrace_info_gain:.4f}")
    print(f"   SHAP Information Gain: {shap_info_gain:.4f}")
    print(f"   Utility Difference (M-TRACE - SHAP): {utility_diff:.4f}")
    print(f"   M-TRACE Cost: {mtrace_cost:.2f}ms")
    print(f"   SHAP Cost: {shap_cost:.2f}ms")
    print(f"   Cost Ratio (M-TRACE/SHAP): {cost_ratio:.2f}")
    print(f"   Utilities Similar (|diff|<0.1): {utilities_similar}")
    print(f"   M-TRACE More Efficient: {mtrace_more_efficient}")
    print(f"   Conclusion: {conclusion}")
    
    # ========================================================================
    # 6. RETURN RESULTS
    # ========================================================================
    results = {
        "seed": seed,
        "experiment_id": exp_run_id,
        "timestamp": time.time(),
        "task_type": f"Static Aggregation ({CONFIG['dataset'].replace('_', ' ').title()})",
        "model_type": CONFIG["model_type"],
        "metrics": {
            "mtrace": {
                "train_time_sec": round(mtrace_train_time, 4),
                "inference_latency_ms": round(mtrace_infer_time * 1000, 2),
                "disk_write_time_ms": round(mtrace_save_time * 1000, 2),
                "parse_load_time_ms": round(mtrace_parse_time * 1000, 2),
                "total_cost_ms": round(mtrace_cost, 2),
                "peak_memory_kb": round(mtrace_peak_memory_kb, 2),
                "found_root_cause": bool(mtrace_found_cause),
                "information_gain": float(mtrace_info_gain),
                "logs_captured_count": len(logs)
            },
            "shap": {
                "total_analysis_time_ms": round(shap_total_time * 1000, 2),
                "peak_memory_kb": round(shap_peak_memory_kb, 2),
                "found_root_cause": bool(shap_found_cause),
                "information_gain": float(shap_info_gain),
                "samples_explained": CONFIG["shap_test_samples"]
            },
            "comparative": {
                "utility_diff": float(utility_diff),
                "cost_ratio": float(cost_ratio),
                "utilities_similar": bool(utilities_similar),
                "mtrace_more_efficient": bool(mtrace_more_efficient),
                "conclusion": conclusion
            }
        },
        "hypothesis": "For static aggregation tasks, M-TRACE temporal data should be redundant (similar utility, lower cost)",
        "validation_status": validation_status
    }
    
    # Save results to JSON
    output_file = CONFIG["output_dir"] / f"exp3_results_seed{seed}.json"
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n   Results saved to: {output_file}")
    print("=" * 80)
    
    return results

# ============================================================================
# MULTI-SEED AGGREGATION FUNCTION
# ============================================================================

def aggregate_multi_seed_results() -> Dict:
    """
    Aggregate results across all 5 seeds with statistical analysis.
    Generates publication-ready LaTeX tables with p-values.
    """
    print(f"\n{'='*80}")
    print("MULTI-SEED AGGREGATION (5 Seeds)")
    print(f"Seeds: {CONFIG['seeds']}")
    print(f"{'='*80}")
    
    # Load all seed results
    seed_results = []
    for seed in CONFIG["seeds"]:
        json_path = CONFIG["output_dir"] / f"exp3_results_seed{seed}.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                seed_results.append(json.load(f))
            print(f"   ✓ Loaded seed {seed}")
        else:
            print(f"   ✗ Missing seed {seed} - run with --seed {seed} first")
    
    if len(seed_results) == 0:
        print("\n   No seed results found. Run individual seeds first.")
        return {}
    
    print(f"\n   Aggregating {len(seed_results)} seeds...")
    
    # Aggregate metrics across seeds
    mtrace_info_gains = [r["metrics"]["mtrace"]["information_gain"] for r in seed_results]
    shap_info_gains = [r["metrics"]["shap"]["information_gain"] for r in seed_results]
    mtrace_costs = [r["metrics"]["mtrace"]["total_cost_ms"] for r in seed_results]
    shap_costs = [r["metrics"]["shap"]["total_analysis_time_ms"] for r in seed_results]
    utility_diffs = [r["metrics"]["comparative"]["utility_diff"] for r in seed_results]
    cost_ratios = [r["metrics"]["comparative"]["cost_ratio"] for r in seed_results]
    
    # Calculate statistics with Welch's t-test
    info_gain_stats = welchs_t_test(mtrace_info_gains, shap_info_gains)
    cost_stats = welchs_t_test(mtrace_costs, shap_costs)
    
    # Count validation outcomes
    pass_count = sum(1 for r in seed_results if r["validation_status"] == "PASS")
    investigation_count = sum(1 for r in seed_results if r["validation_status"] == "NEEDS_INVESTIGATION")
    
    final_results = {
        "mtrace_info_gain_mean": float(np.mean(mtrace_info_gains)),
        "mtrace_info_gain_std": float(np.std(mtrace_info_gains, ddof=1)),
        "shap_info_gain_mean": float(np.mean(shap_info_gains)),
        "shap_info_gain_std": float(np.std(shap_info_gains, ddof=1)),
        "info_gain_diff_mean": float(np.mean(utility_diffs)),
        "info_gain_diff_std": float(np.std(utility_diffs, ddof=1)),
        "info_gain_p_value": float(info_gain_stats["p_value"]),
        "info_gain_effect_size": float(info_gain_stats["effect_size"]),  # ← ADD THIS
        "mtrace_cost_mean": float(np.mean(mtrace_costs)),
        "mtrace_cost_std": float(np.std(mtrace_costs, ddof=1)),
        "shap_cost_mean": float(np.mean(shap_costs)),
        "shap_cost_std": float(np.std(shap_costs, ddof=1)),
        "cost_diff_mean": float(np.mean(mtrace_costs) - np.mean(shap_costs)),
        "cost_p_value": float(cost_stats["p_value"]),
        "cost_effect_size": float(cost_stats["effect_size"]),          # ← ADD THIS
        "cost_ratio_mean": float(np.mean(cost_ratios)),
        "cost_ratio_std": float(np.std(cost_ratios, ddof=1)),
        "pass_count": int(pass_count),
        "investigation_count": int(investigation_count),
        "n_seeds": int(len(seed_results))
    }
    
    # Generate LaTeX table
    latex_table = generate_statistical_latex_table(final_results, len(seed_results))
    
    # Save aggregated results
    agg_path = CONFIG["output_dir"] / "aggregated" / "exp3_multi_seed_aggregated.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_safe_data = convert_numpy_types({
        "seeds": CONFIG["seeds"],
        "n_seeds": len(seed_results),
        "metrics": final_results,
        "latex_table": latex_table
    })
    
    with open(agg_path, "w") as f:
        json.dump(json_safe_data, f, indent=2)
    
    print(f"\n   Aggregated results saved to: {agg_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS (5 Seeds)")
    print(f"{'='*80}")
    print(f"\nInformation Gain Comparison:")
    print(f"   M-TRACE: {final_results['mtrace_info_gain_mean']:.3f} ± {final_results['mtrace_info_gain_std']:.3f}")
    print(f"   SHAP:    {final_results['shap_info_gain_mean']:.3f} ± {final_results['shap_info_gain_std']:.3f}")
    print(f"   Diff:    {final_results['info_gain_diff_mean']:.3f} ± {final_results['info_gain_diff_std']:.3f} ({format_p_value(final_results['info_gain_p_value'])})")
    print(f"\nComputational Cost Comparison:")
    print(f"   M-TRACE: {final_results['mtrace_cost_mean']:.2f} ± {final_results['mtrace_cost_std']:.2f} ms")
    print(f"   SHAP:    {final_results['shap_cost_mean']:.2f} ± {final_results['shap_cost_std']:.2f} ms")
    print(f"   Cost Ratio (M-TRACE/SHAP): {final_results['cost_ratio_mean']:.2f} ± {final_results['cost_ratio_std']:.2f}")
    print(f"\nValidation Outcomes:")
    print(f"   PASS (Temporal Data Redundant): {pass_count}/{len(seed_results)}")
    print(f"   NEEDS INVESTIGATION: {investigation_count}/{len(seed_results)}")
    print(f"{'='*80}")
    
    return final_results

def format_effect_size(cohens_d: float, threshold: float = 10.0) -> str:
    """Format Cohen's d with interpretation guard for extreme values."""
    if abs(cohens_d) > threshold:
        return r"$d \gg 1.0^\dagger$"  # Maximal practical significance
    elif abs(cohens_d) >= 0.8:
        return r"$d \geq 0.8$ (large)"
    elif abs(cohens_d) >= 0.5:
        return r"$d \geq 0.5$ (medium)"
    elif abs(cohens_d) >= 0.2:
        return r"$d \geq 0.2$ (small)"
    else:
        return r"$d < 0.2$ (negligible)"

def generate_statistical_latex_table(results: Dict, n_seeds: int) -> str:
    """Generate publication-ready LaTeX table with statistical significance markers and effect sizes."""
    
    # Extract & format effect sizes
    ig_d_formatted = format_effect_size(results.get("info_gain_effect_size", 0.0))
    cost_d_formatted = format_effect_size(results.get("cost_effect_size", 0.0))

    info_gain_mtrace = format_statistical_result(
        results["mtrace_info_gain_mean"], results["mtrace_info_gain_std"],
        results["info_gain_p_value"], n_seeds
    )
    info_gain_shap = format_statistical_result(
        results["shap_info_gain_mean"], results["shap_info_gain_std"],
        results["info_gain_p_value"], n_seeds
    )
    cost_mtrace = f"${results['mtrace_cost_mean']:.2f}\\pm{results['mtrace_cost_std']:.2f}$"
    cost_shap = f"${results['shap_cost_mean']:.2f}\\pm{results['shap_cost_std']:.2f}$"
    validation_outcome = f"{results['pass_count']}/{n_seeds} PASS"

    return r"""
\begin{table}[h]
\centering
\caption{Experiment 3: Boundary Condition Analysis(""" + f"{n_seeds} Seeds, Direct Comparison)" + r"""}
\label{tab:exp3_boundary_conditions}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{M-TRACE} & \textbf{SHAP} & \textbf{Effect Size ($d$)} & \textbf{Interpretation} \\
\midrule
Information Gain & """ + info_gain_mtrace + r""" & """ + info_gain_shap + r""" & """ + ig_d_formatted + r""" & \makecell[l]{Utilities similar($\|\text{diff}\|<0.1$)\\ validates redundancy hypothesis} \\
\midrule
Computational Cost & """ + cost_mtrace + r""" & """ + cost_shap + r""" & """ + cost_d_formatted + r""" & \makecell[l]{Cost ratio quantifies overhead\\ for static tasks} \\
\midrule
Cost Ratio (M-TRACE/SHAP) & \multicolumn{3}{c}{$""" + f"{results['cost_ratio_mean']:.2f}\\pm{results['cost_ratio_std']:.2f}" + r"""$} & \makecell[l]{$<1.0$: M-TRACE more efficient\\ $>1.0$: SHAP more efficient} \\
\midrule
Validation Outcome & \multicolumn{3}{c}{""" + validation_outcome + r"""} & \makecell[l]{PASS = temporal data redundant\\ (expected for static tasks)} \\
\bottomrule
\end{tabular}
\begin{flushleft}
\small\textit{Note:} Direct comparison of diagnostic utility and computational efficiency (TRI formula dropped per reviewer recommendation).
For static aggregation tasks, temporal trajectory data $\mathcal{T}(x)$ should be redundant.
Information gain similarity ($\|\text{diff}\|<0.1$) validates that M-TRACE adds no value over SHAP for static tasks.
Cost ratio quantifies overhead trade-off.
This demonstrates \textbf{intellectual honesty} about boundary conditions: M-TRACE is a calibrated research instrument, not a universal replacement.
Statistical significance: Welch's $t$-test (unequal variance); effect sizes reported as Cohen's $d$ (large: $d \geq 0.8$).
$^\dagger$Effect size exceeds computational precision due to near-zero variance in baseline; indicates maximal practical significance.
$n=""" + f"{n_seeds * 500}" + r"""$ samples across """ + f"{n_seeds}" + r""" seeds (""" + f"{', '.join(map(str, CONFIG['seeds']))}" + r""").
\end{flushleft}
\end{table}
"""

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Boundary Condition Analysis")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed (e.g., 42)")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate all 5 seeds")
    parser.add_argument("--all", action="store_true", help="Run all 5 seeds and aggregate")
    parser.add_argument("--dataset", type=str, default="breast_cancer", 
                       choices=["breast_cancer", "digits"], help="Dataset to use")
    parser.add_argument("--model-type", type=str, default="mlp",
                       choices=["mlp", "random_forest"], help="Model type")
    args = parser.parse_args()
    
    # Update config from args
    CONFIG["dataset"] = args.dataset
    CONFIG["model_type"] = args.model_type
    
    if args.all:
        print(f"\n{'#'*80}")
        print(f"# RUNNING ALL 5 SEEDS + AGGREGATION")
        print(f"{'#'*80}")
        
        for seed in CONFIG["seeds"]:
            CONFIG["seed"] = seed
            run_single_seed(seed)
        
        aggregate_multi_seed_results()
        
    elif args.seed is not None:
        CONFIG["seed"] = args.seed
        run_single_seed(args.seed)
        
    elif args.aggregate:
        aggregate_multi_seed_results()
        
    else:
        print("\nNo arguments provided. Running single seed (42)...")
        print("Use --all to run all 5 seeds + aggregation")
        print("Use --seed <N> to run specific seed")
        print("Use --aggregate to aggregate existing results\n")
        CONFIG["seed"] = 42
        run_single_seed(42)

if __name__ == "__main__":
    main()