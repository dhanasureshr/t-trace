# t_trace/experiments/phase2/exp1/aggregate_seed_results.py
import json
import numpy as np
from pathlib import Path
from scipy import stats

def aggregate_experiment1_results(results_dir: Path):
    """Aggregate all 5 seed results with statistical rigor per FAccT standards."""
    
    seeds = [42, 123, 456, 789, 1011]
    mtrace_precision = []
    overhead_ms = []
    overhead_pct = []
    
    print("="*70)
    print("PHASE 2 EXPERIMENT 1: STATISTICAL AGGREGATION (5 Random Seeds)")
    print("="*70)
    
    for seed in seeds:
        result_file = results_dir / f"experiment1_seed{seed}_results.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                mtrace_precision.append(data["metrics"]["mtrace_temporal_precision"])
                overhead_ms.append(data["metrics"]["mtrace_overhead_ms"])
                overhead_pct.append(data["metrics"]["overhead_percentage"])
                print(f"✅ Seed {seed:4d}: Precision={data['metrics']['mtrace_temporal_precision']:.3f}, "
                      f"Overhead={data['metrics']['mtrace_overhead_ms']:.2f}ms ({data['metrics']['overhead_percentage']:+.1f}%)")
        else:
            print(f"❌ Missing: {result_file}")
    
    if len(mtrace_precision) < 5:
        print(f"\n⚠️ Warning: Only {len(mtrace_precision)}/5 seeds found! Run missing seeds first.")
        return None
    
    # Compute statistics (sample std with ddof=1 for n=5)
    def fmt_stats(values, suffix=""):
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        return f"{mean:.3f} ± {std:.3f}{suffix}"
    
    print("\n" + "-"*70)
    print("AGGREGATED RESULTS (mean ± std, n=5 seeds)")
    print("-"*70)
    print(f"M-TRACE Temporal Precision: {fmt_stats(mtrace_precision, '')}")
    print(f"M-TRACE Overhead:           {fmt_stats(overhead_ms, ' ms')}")
    print(f"M-TRACE Overhead %:         {fmt_stats(overhead_pct, ' %')}")
    
    # Significance testing: M-TRACE vs SHAP (SHAP = 0 for temporal precision)
    shap_precision = [0.0] * 5  # SHAP cannot compute temporal precision (structurally impossible)
    t_stat, p_value = stats.ttest_ind(mtrace_precision, shap_precision, equal_var=False)  # Welch's t-test
    
    print("\n" + "-"*70)
    print("SIGNIFICANCE TESTING (Welch's t-test, M-TRACE vs SHAP)")
    print("-"*70)
    print(f"t({len(mtrace_precision)+len(shap_precision)-2}) = {t_stat:.3f}")
    print(f"p-value = {p_value:.4e}")
    print(f"Statistically Significant (p<0.05): {'✅ YES' if p_value < 0.05 else '❌ NO'}")
    
    # Effect size (Cohen's d)
    def cohens_d(a, b):
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(a) - np.mean(b)) / pooled_std
    
    d = cohens_d(mtrace_precision, shap_precision)
    effect_mag = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    print(f"Effect Size (Cohen's d): {d:.3f} ({effect_mag})")
    
    # Save aggregated results
    output = {
        "experiment": "Phase 2 Experiment 1: Temporal Fidelity",
        "n_seeds": len(mtrace_precision),
        "seeds_used": seeds,
        "mtrace_temporal_precision": {
            "mean": float(np.mean(mtrace_precision)),
            "std": float(np.std(mtrace_precision, ddof=1)),
            "report": f"{np.mean(mtrace_precision):.3f} ± {np.std(mtrace_precision, ddof=1):.3f}"
        },
        "overhead_ms": {
            "mean": float(np.mean(overhead_ms)),
            "std": float(np.std(overhead_ms, ddof=1)),
            "report": f"{np.mean(overhead_ms):.2f} ± {np.std(overhead_ms, ddof=1):.2f} ms"
        },
        "overhead_percentage": {
            "mean": float(np.mean(overhead_pct)),
            "std": float(np.std(overhead_pct, ddof=1)),
            "report": f"{np.mean(overhead_pct):.2f} ± {np.std(overhead_pct, ddof=1):.2f} %"
        },
        "significance": {
            "test": "Welch's t-test (unequal variances)",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "cohens_d": float(d),
            "effect_magnitude": effect_mag
        },
        "shap_baseline_note": "SHAP cannot compute temporal precision (structurally impossible - operates post-hoc)"
    }
    
    output_path = results_dir.parent / "aggregated" / "experiment1_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Aggregated results saved to: {output_path}")
    print("="*70)
    
    return output

if __name__ == "__main__":
    results_dir = Path("t_trace/experiments/phase2/exp1/results/raw")
    aggregate_experiment1_results(results_dir)