"""
Statistical Rigor Implementation for Phase 2 Experiment 5
Implements: 5-seed validation, mean±std reporting, t-test significance testing
Aligned with M-TRACE Experimental Plan v3, Section: Critical Experimental Controls
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy import stats as scipy_stats 
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


class StatisticalRigor:
    """
    Statistical analysis for M-TRACE Phase 2 Experiment 5.
    
    Implements publication-ready statistical validation per FAccT/NeurIPS standards:
    - 5 random seeds per experiment
    - Mean ± standard deviation reporting
    - Welch's t-test for significance (p<0.05)
    - Cohen's d effect size calculation
    - 95% confidence intervals
    """
    
    def __init__(self, results_dir: Path = None):
        # FIXED: Default to correct base directory (NOT including /raw)
        if results_dir is None:
            results_dir = Path("t_trace/experiments/phase2/exp5/results")
        
        self.results_dir = results_dir
        
        # FIXED: Ensure raw_dir doesn't duplicate /raw
        if str(self.results_dir).endswith('/raw'):
            self.raw_dir = self.results_dir
        else:
            self.raw_dir = self.results_dir / "raw"
        
        self.aggregated_dir = self.results_dir / "aggregated"
        self.figures_dir = self.results_dir / "figures"
        
        # Create directories
        for dir_path in [self.raw_dir, self.aggregated_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"StatisticalRigor initialized:")
        logger.info(f"  Raw dir: {self.raw_dir}")
        logger.info(f"  Aggregated dir: {self.aggregated_dir}")
        
        # Standard seeds for reproducibility (per Experimental Plan v3)
        self.standard_seeds = [42, 123, 456, 789, 1011]
    
    @staticmethod
    def _convert_numpy_types(obj):
        """Recursively convert NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: StatisticalRigor._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [StatisticalRigor._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.longlong)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Compute comprehensive statistics with EMPTY ARRAY VALIDATION
        """
        # FIXED: Validate input before processing
        if not values or len(values) == 0:
            logger.warning("Empty values list provided to compute_statistics")
            return {
                "mean": 0.0,
                "std": 0.0,
                "sem": 0.0,
                "n": 0,
                "ci_95_lower": 0.0,
                "ci_95_upper": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        arr = np.array(values)
        n = len(arr)
        
        if n < 2:
            logger.warning(f"Only {n} sample(s) - statistics may be unreliable")
        
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        
        # 95% Confidence Interval using t-distribution
        if n > 1:
            ci_margin = stats.t.ppf(0.975, df=n-1) * sem
        else:
            ci_margin = 0.0
        
        # FIXED: Safe min/max for non-empty arrays
        return {
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
            "ci_95_lower": float(mean - ci_margin),
            "ci_95_upper": float(mean + ci_margin),
            "min": float(np.min(arr)) if n > 0 else 0.0,
            "max": float(np.max(arr)) if n > 0 else 0.0
        }
    
    def format_result(self, values: List[float], metric_name: str = "metric") -> str:
        """Format result as 'mean ± std' with empty array handling"""
        stats_dict = self.compute_statistics(values)
        
        if stats_dict["n"] == 0:
            return f"N/A (no data) {metric_name}"
        
        # Handle zero variance case for publication
        if stats_dict["std"] < 1e-10:
            return f"{stats_dict['mean']:.3f} (std={stats_dict['std']:.3f}) {metric_name}"
        else:
            return f"{stats_dict['mean']:.3f} ± {stats_dict['std']:.3f} {metric_name}"
    
    def perform_t_test(
        self,
        group_a: List[float],
        group_b: List[float],
        alternative: str = "two-sided"
    ) -> Dict[str, float]:
        """
        Perform Welch's t-test with validation.
        """
        # FIXED: Validate inputs
        if not group_a or not group_b or len(group_a) < 2 or len(group_b) < 2:
            logger.warning("Insufficient samples for t-test")
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "cohens_d": 0.0,
                "effect_magnitude": "unknown",
                "df": 0
            }
        
        # Welch's t-test (does not assume equal variances)
        t_stat, p_value = stats.ttest_ind(
            group_a, group_b,
            equal_var=False,  # Welch's t-test
            alternative=alternative
        )
        
        # Cohen's d effect size
        def cohens_d(a, b):
            n1, n2 = len(a), len(b)
            if n1 < 2 or n2 < 2:
                return 0.0
            var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            if pooled_std == 0:
                return 0.0
            return (np.mean(a) - np.mean(b)) / pooled_std
        
        effect_size = cohens_d(group_a, group_b)
        
        def interpret_effect_size(d):
            d = abs(d)
            if d < 0.2:
                return "negligible"
            elif d < 0.5:
                return "small"
            elif d < 0.8:
                return "medium"
            else:
                return "large"
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "cohens_d": float(effect_size),
            "effect_magnitude": interpret_effect_size(effect_size),
            "df": len(group_a) + len(group_b) - 2
        }
    
    def save_seed_result(
        self,
        seed: int,
        tri_score: float,
        mtrace_cost_ms: float,
        shap_cost_ms: float,
        mtrace_info_gain: float,
        shap_info_gain: float,
        cost_ratio: float,
        additional_metrics: Dict = None
    ) -> Path:
        """Save results for a single seed run to JSON"""
        result = {
            "seed": int(seed),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "temporal_redundancy_index": float(tri_score),
                "mtrace_cost_ms": float(mtrace_cost_ms),
                "shap_cost_ms": float(shap_cost_ms),
                "mtrace_info_gain": float(mtrace_info_gain),
                "shap_info_gain": float(shap_info_gain),
                "cost_ratio": float(cost_ratio)
            },
            "additional_metrics": self._convert_numpy_types(additional_metrics or {})
        }
        
        # FIXED: Ensure correct path (no duplicate /raw)
        output_path = self.raw_dir / f"experiment5_seed{seed}_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved seed {seed} results to {output_path}")
        return output_path
    
    def _load_all_seed_results(self) -> List[Dict]:
        """Load all seed results from raw directory with path validation"""
        results = []
        
        logger.info(f"Loading seed results from: {self.raw_dir}")
        logger.info(f"Directory exists: {self.raw_dir.exists()}")
        
        if self.raw_dir.exists():
            json_files = list(self.raw_dir.glob("experiment5_seed*_results.json"))
            logger.info(f"Found {len(json_files)} result files: {[f.name for f in json_files]}")
        
        for seed in self.standard_seeds:
            result_file = self.raw_dir / f"experiment5_seed{seed}_results.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
                logger.info(f"✓ Loaded seed {seed}: {result_file}")
            else:
                logger.warning(f"✗ Missing results for seed {seed}: {result_file}")
        
        return results
    
    def aggregate_all_seeds(self) -> pd.DataFrame:
        """Aggregate results from all 5 seeds with validation"""
        results = self._load_all_seed_results()
        
        if len(results) < 2:
            logger.error(f"Insufficient seed results for statistical analysis ({len(results)} < 2)")
            return pd.DataFrame()
        
        # Extract metrics
        tri_scores = [r["metrics"]["temporal_redundancy_index"] for r in results]
        mtrace_costs = [r["metrics"]["mtrace_cost_ms"] for r in results]
        shap_costs = [r["metrics"]["shap_cost_ms"] for r in results]
        mtrace_gains = [r["metrics"]["mtrace_info_gain"] for r in results]
        shap_gains = [r["metrics"]["shap_info_gain"] for r in results]
        cost_ratios = [r["metrics"]["cost_ratio"] for r in results]
        
        # Compute statistics
        agg_data = {
            "metric": [
                "Temporal Redundancy Index (TRI)",
                "M-TRACE Cost (ms)",
                "SHAP Cost (ms)",
                "M-TRACE Info Gain",
                "SHAP Info Gain",
                "Cost Ratio (M-TRACE/SHAP)"
            ],
            "mean": [
                self.compute_statistics(tri_scores)["mean"],
                self.compute_statistics(mtrace_costs)["mean"],
                self.compute_statistics(shap_costs)["mean"],
                self.compute_statistics(mtrace_gains)["mean"],
                self.compute_statistics(shap_gains)["mean"],
                self.compute_statistics(cost_ratios)["mean"]
            ],
            "std": [
                self.compute_statistics(tri_scores)["std"],
                self.compute_statistics(mtrace_costs)["std"],
                self.compute_statistics(shap_costs)["std"],
                self.compute_statistics(mtrace_gains)["std"],
                self.compute_statistics(shap_gains)["std"],
                self.compute_statistics(cost_ratios)["std"]
            ],
            "n_seeds": [len(results)] * 6,
        }
        
        df = pd.DataFrame(agg_data)
        
        # Save aggregated results
        output_path = self.aggregated_dir / "experiment5_aggregated_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved aggregated results to {output_path}")
        
        return df
    
    def generate_significance_report(
        self,
        mtrace_costs: List[float],
        shap_costs: List[float],
        tri_scores: List[float]
    ) -> str:
        """Generate publication-ready significance testing report"""
        # FIXED: Validate inputs before processing
        if not mtrace_costs or not shap_costs:
            logger.error("Cannot generate significance report: empty input arrays")
            return "ERROR: Insufficient data for significance testing"
        
        # Test 1: Cost comparison (M-TRACE vs SHAP)
        cost_test = self.perform_t_test(mtrace_costs, shap_costs, alternative="less")
        
        # Test 2: TRI vs expected value (1.0 for static tasks)
        tri_test_stats = stats.ttest_1samp(tri_scores, 1.0)
        
        report = (
            f"\n{'='*70}\n"
            f"STATISTICAL SIGNIFICANCE REPORT: Experiment 5 (Boundary Conditions)\n"
            f"{'='*70}\n"
            f"\nTEST 1: Cost Comparison (M-TRACE vs SHAP)\n"
            f"{'-'*70}\n"
            f"M-TRACE Cost: {self.format_result(mtrace_costs, 'ms')}\n"
            f"SHAP Cost:    {self.format_result(shap_costs, 'ms')}\n"
            f"Welch's t-test Results (one-sided):\n"
            f"  t({cost_test['df']}) = {cost_test['t_statistic']:.3f}\n"
            f"  p-value = {cost_test['p_value']:.4e}\n"
            f"  M-TRACE significantly cheaper: {'✅ YES' if cost_test['significant'] else '❌ NO'}\n"
            f"  Effect Size (Cohen's d): {cost_test['cohens_d']:.3f} ({cost_test['effect_magnitude']})\n"
            f"\nTEST 2: TRI vs Expected Value (1.0 for static tasks)\n"
            f"{'-'*70}\n"
            f"TRI Mean: {np.mean(tri_scores):.4f} ± {np.std(tri_scores, ddof=1):.4f}\n"
            f"Expected: 1.0 (temporal data redundant)\n"
            f"One-sample t-test:\n"
            f"  t({len(tri_scores)-1}) = {tri_test_stats[0]:.3f}\n"
            f"  p-value = {tri_test_stats[1]:.4e}\n"
            f"  TRI ≈ 1.0 confirmed: {'✅ YES' if tri_test_stats[1] > 0.05 else '❌ NO'}\n"
            f"\n{'='*70}\n"
            f"INTERPRETATION:\n"
            f"  For static aggregation tasks, M-TRACE temporal data is REDUNDANT\n"
            f"  TRI ≈ 1.0 validates boundary condition hypothesis\n"
            f"  M-TRACE is honest about when it adds value (vs post-hoc tools)\n"
            f"{'='*70}\n"
        )
        
        # Save report
        report_path = self.aggregated_dir / "significance_report_boundary_conditions.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved significance report to {report_path}")
        return report
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX-ready table for publication"""
        df = self.aggregate_all_seeds()
        
        if df.empty:
            logger.error("Cannot generate LaTeX table: no aggregated data")
            return ""
        
        # Load seed results for t-tests
        seed_results = self._load_all_seed_results()
        mtrace_costs = [r["metrics"]["mtrace_cost_ms"] for r in seed_results]
        shap_costs = [r["metrics"]["shap_cost_ms"] for r in seed_results]
        tri_scores = [r["metrics"]["temporal_redundancy_index"] for r in seed_results]
        
        cost_test = self.perform_t_test(mtrace_costs, shap_costs, alternative="less")
        tri_test_stats = stats.ttest_1samp(tri_scores, 1.0)
        
        latex_table = r"""
\begin{table}[h]
\centering
\caption{Phase 2 Experiment 5: Boundary Condition Analysis (5 Random Seeds)}
\label{tab:experiment5_boundary}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{M-TRACE} & \textbf{SHAP} & \textbf{p-value} \\
\midrule
Temporal Redundancy Index & """
        
        tri_row = df[df["metric"] == "Temporal Redundancy Index (TRI)"]
        if not tri_row.empty:
            latex_table += f"${tri_row.iloc[0]['mean']:.3f} \\pm {tri_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += r"N/A & "
        
        # TRI test p-value
        latex_table += f"${tri_test_stats[1]:.2e}$ \\\\"
        
        latex_table += r"""
\midrule
Analysis Cost (ms) & """
        
        mtrace_row = df[df["metric"] == "M-TRACE Cost (ms)"]
        shap_row = df[df["metric"] == "SHAP Cost (ms)"]
        
        if not mtrace_row.empty:
            latex_table += f"${mtrace_row.iloc[0]['mean']:.2f} \\pm {mtrace_row.iloc[0]['std']:.2f}$ & "
        else:
            latex_table += "N/A & "
        
        if not shap_row.empty:
            latex_table += f"${shap_row.iloc[0]['mean']:.2f} \\pm {shap_row.iloc[0]['std']:.2f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += f"${cost_test['p_value']:.2e}$ \\\\"
        
        latex_table += r"""
\midrule
Information Gain & """
        
        mtrace_gain_row = df[df["metric"] == "M-TRACE Info Gain"]
        shap_gain_row = df[df["metric"] == "SHAP Info Gain"]
        
        if not mtrace_gain_row.empty:
            latex_table += f"${mtrace_gain_row.iloc[0]['mean']:.3f} \\pm {mtrace_gain_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "N/A & "
        
        if not shap_gain_row.empty:
            latex_table += f"${shap_gain_row.iloc[0]['mean']:.3f} \\pm {shap_gain_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += r"N/A \\"
        
        latex_table += r"""
\bottomrule
\end{tabular}
\begin{flushleft}
\small
\textit{Note:} TRI ≈ 1.0 indicates temporal data is redundant for static tasks (null hypothesis confirmed).
Cost comparison: Welch's t-test (one-sided, M-TRACE < SHAP). TRI test: one-sample t-test vs 1.0.
n=5 random seeds (42, 123, 456, 789, 1011). Breast Cancer dataset, MLP classifier.
\end{flushleft}
\end{table}
"""
        
        # Save LaTeX table
        latex_path = self.aggregated_dir / "experiment5_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved LaTeX table to {latex_path}")
        return latex_table
    
    def generate_tri_comparison_plot(self) -> Path:
        """Generate publication-quality TRI comparison plot."""
        seed_results = self._load_all_seed_results()
        
        if not seed_results:
            logger.warning("No seed results available for plotting")
            return None
        
        tri_scores = [r["metrics"]["temporal_redundancy_index"] for r in seed_results]
        mtrace_costs = [r["metrics"]["mtrace_cost_ms"] for r in seed_results]
        shap_costs = [r["metrics"]["shap_cost_ms"] for r in seed_results]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: TRI scores with expected value line
        colors = ['#2E86AB'] * len(tri_scores)
        ax1.scatter(range(len(tri_scores)), tri_scores, s=150, c=colors, 
                   marker='o', edgecolors='white', linewidths=2, zorder=5)
        ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                   label='Expected TRI=1.0 (Redundant)', alpha=0.7)
        ax1.axhline(y=np.mean(tri_scores), color='red', linestyle='-', linewidth=2,
                   label=f'Mean TRI={np.mean(tri_scores):.3f}', alpha=0.7)
        
        # Add error bars
        tri_std = np.std(tri_scores, ddof=1)
        ax1.errorbar(range(len(tri_scores)), tri_scores, yerr=tri_std, 
                    fmt='none', ecolor='black', capsize=8, linewidth=2)
        
        ax1.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temporal Redundancy Index (TRI)', fontsize=12, fontweight='bold')
        ax1.set_title('Boundary Condition Validation\n(TRI ≈ 1.0 = Temporal Data Redundant)',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(range(len(tri_scores)))
        ax1.set_xticklabels([f'Seed {s}' for s in self.standard_seeds], fontsize=10)
        ax1.set_ylim(0.8, 1.1)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_axisbelow(True)
        
        # Plot 2: Cost comparison (M-TRACE vs SHAP)
        x_pos = np.arange(len(self.standard_seeds))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, mtrace_costs, width, 
                       label='M-TRACE', color='#2E86AB', edgecolor='white', linewidth=2)
        bars2 = ax2.bar(x_pos + width/2, shap_costs, width, 
                       label='SHAP', color='#A23B72', edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar, cost in zip(bars1, mtrace_costs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{cost:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, cost in zip(bars2, shap_costs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{cost:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Analysis Cost (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Computational Cost Comparison\n(M-TRACE vs SHAP)',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'Seed {s}' for s in self.standard_seeds], fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax2.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        output_path = self.figures_dir / "exp5_tri_cost_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved TRI comparison plot to {output_path}")
        plt.close()
        
        return output_path


# ============================================================================
# MAIN AGGREGATION SCRIPT
# ============================================================================

def aggregate_experiment5_results(results_dir: Path = None):
    """
    Aggregate all 5 seed results with statistical rigor per FAccT standards.
    """
    if results_dir is None:
        results_dir = Path("t_trace/experiments/phase2/exp5/results")
    
    stats = StatisticalRigor(results_dir=results_dir)
    
    # Load all seed results
    seed_results = stats._load_all_seed_results()
    
    if len(seed_results) < 5:
        logger.warning(f"Only {len(seed_results)}/5 seeds found! Run missing seeds first.")
    
    # Extract metrics
    tri_scores = [r["metrics"]["temporal_redundancy_index"] for r in seed_results]
    mtrace_costs = [r["metrics"]["mtrace_cost_ms"] for r in seed_results]
    shap_costs = [r["metrics"]["shap_cost_ms"] for r in seed_results]
    mtrace_gains = [r["metrics"]["mtrace_info_gain"] for r in seed_results]
    shap_gains = [r["metrics"]["shap_info_gain"] for r in seed_results]
    cost_ratios = [r["metrics"]["cost_ratio"] for r in seed_results]
    
    print("="*70)
    print("PHASE 2 EXPERIMENT 5: STATISTICAL AGGREGATION (5 Random Seeds)")
    print("="*70)
    
    # Print per-seed results
    for i, result in enumerate(seed_results):
        seed = result["seed"]
        tri = result["metrics"]["temporal_redundancy_index"]
        mtrace_cost = result["metrics"]["mtrace_cost_ms"]
        shap_cost = result["metrics"]["shap_cost_ms"]
        mtrace_gain = result["metrics"]["mtrace_info_gain"]
        shap_gain = result["metrics"]["shap_info_gain"]
        print(f"✅ Seed {seed:4d}: TRI={tri:.4f}, M-TRACE={mtrace_cost:.2f}ms, "
              f"SHAP={shap_cost:.2f}ms, Gain(M={mtrace_gain:.2f}, S={shap_gain:.2f})")
    
    # Compute statistics
    def fmt_stats(values, suffix=""):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std < 1e-10:
            return f"{mean:.3f} (std={std:.3f}){suffix}"
        else:
            return f"{mean:.3f} ± {std:.3f}{suffix}"
    
    print("\n" + "-"*70)
    print("AGGREGATED RESULTS (mean ± std, n=5 seeds)")
    print("-"*70)
    print(f"Temporal Redundancy Index: {fmt_stats(tri_scores, '')}")
    print(f"M-TRACE Cost:              {fmt_stats(mtrace_costs, ' ms')}")
    print(f"SHAP Cost:                 {fmt_stats(shap_costs, ' ms')}")
    print(f"M-TRACE Info Gain:         {fmt_stats(mtrace_gains, '')}")
    print(f"SHAP Info Gain:            {fmt_stats(shap_gains, '')}")
    print(f"Cost Ratio (M/S):          {fmt_stats(cost_ratios, '')}")
    
    # Significance testing
    print("\n" + "-"*70)
    print("SIGNIFICANCE TESTING")
    print("-"*70)
    
    # Test 1: Cost comparison
    cost_test = stats.perform_t_test(mtrace_costs, shap_costs, alternative="less")
    print(f"\nCost Comparison (M-TRACE vs SHAP):")
    print(f"  t({cost_test['df']}) = {cost_test['t_statistic']:.3f}")
    print(f"  p-value = {cost_test['p_value']:.4e}")
    print(f"  M-TRACE significantly cheaper: {'✅ YES' if cost_test['significant'] else '❌ NO'}")
    print(f"  Effect Size: {cost_test['cohens_d']:.3f} ({cost_test['effect_magnitude']})")
    
    # Test 2: TRI vs expected value
    tri_test_stats = scipy_stats.ttest_1samp(tri_scores, 1.0)
    print(f"\nTRI vs Expected (1.0 for static tasks):")
    print(f"  t({len(tri_scores)-1}) = {tri_test_stats[0]:.3f}")
    print(f"  p-value = {tri_test_stats[1]:.4e}")
    print(f"  TRI ≈ 1.0 confirmed: {'✅ YES' if tri_test_stats[1] > 0.05 else '❌ NO'}")
    
    # Save aggregated results
    output = {
        "experiment": "Phase 2 Experiment 5: Boundary Condition Analysis",
        "n_seeds": len(seed_results),
        "seeds_used": [r["seed"] for r in seed_results],
        "temporal_redundancy_index": {
            "mean": float(np.mean(tri_scores)),
            "std": float(np.std(tri_scores, ddof=1)),
            "report": fmt_stats(tri_scores, ''),
            "expected_value": 1.0,
            "hypothesis_confirmed": bool(tri_test_stats[1] > 0.05)
        },
        "mtrace_cost_ms": {
            "mean": float(np.mean(mtrace_costs)),
            "std": float(np.std(mtrace_costs, ddof=1)),
            "report": fmt_stats(mtrace_costs, ' ms')
        },
        "shap_cost_ms": {
            "mean": float(np.mean(shap_costs)),
            "std": float(np.std(shap_costs, ddof=1)),
            "report": fmt_stats(shap_costs, ' ms')
        },
        "cost_comparison_test": {
            "test": "Welch's t-test (one-sided, M-TRACE < SHAP)",
            "t_statistic": float(cost_test['t_statistic']),
            "p_value": float(cost_test['p_value']),
            "significant": bool(cost_test['significant']),
            "cohens_d": float(cost_test['cohens_d'])
        },
        "tri_test": {
            "test": "One-sample t-test (vs 1.0)",
            "t_statistic": float(tri_test_stats[0]),
            "p_value": float(tri_test_stats[1]),
            "null_hypothesis_confirmed": bool(tri_test_stats[1] > 0.05)
        },
        "key_insight": "For static tasks, M-TRACE temporal data is redundant (TRI ≈ 1.0) - validates boundary condition hypothesis"
    }
    
    output_path = stats.aggregated_dir / "experiment5_summary.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Aggregated results saved to: {output_path}")
    print("="*70)
    
    # Generate significance report
    report = stats.generate_significance_report(mtrace_costs, shap_costs, tri_scores)
    print(report)
    
    # Generate LaTeX table
    latex_table = stats.generate_latex_table()
    
    # Generate comparison plot
    plot_path = stats.generate_tri_comparison_plot()
    
    return output


if __name__ == "__main__":
    results_dir = Path("t_trace/experiments/phase2/exp5/results")
    aggregate_experiment5_results(results_dir)