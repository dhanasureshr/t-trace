#!/usr/bin/env python3
"""
Appendix C: Threshold Sensitivity Ablation for Experiment 2
Tests CSVR robustness to Δp threshold variations: [0.10, 0.15, 0.20]
"""
import sys, json, numpy as np
from pathlib import Path
from exp2_causal_verification import run_single_seed, CONFIG, convert_numpy_types

# Override thresholds to test
THRESHOLDS = [0.10, 0.15, 0.20]
SEEDS = [42, 123, 456, 789, 1011]

def run_threshold_ablation():
    results = {tau: {"bert": [], "roberta": []} for tau in THRESHOLDS}
    
    for tau in THRESHOLDS:
        print(f"\n=== Testing threshold τ = {tau} ===")
        CONFIG["patching_threshold"] = tau  # Override global config
        
        for seed in SEEDS:
            seed_result = run_single_seed(seed)  # Reuse existing function
            metrics = seed_result.get("metrics", {})
            
            for model_type in ["bert", "roberta"]:
                if model_type in metrics:
                    csvr = metrics[model_type]["csvr"]
                    results[tau][model_type].append(csvr)
    
    # Aggregate & save
    aggregated = {}
    for tau in THRESHOLDS:
        aggregated[tau] = {}
        for model_type in ["bert", "roberta"]:
            vals = results[tau][model_type]
            if vals:
                aggregated[tau][model_type] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1))
                }
    
    # Save to JSON
    output_path = CONFIG["results_dir"] / "exp2_threshold_sensitivity.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(convert_numpy_types({
            "thresholds": THRESHOLDS,
            "seeds": SEEDS,
            "results": aggregated
        }), f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return aggregated

if __name__ == "__main__":
    run_threshold_ablation()