"""
Statistical Rigor Implementation for Phase 2 Experiment 4
Implements: 5-seed validation, mean±std reporting, t-test significance testing
Aligned with M-TRACE Experimental Plan v3, Section: Critical Experimental Controls
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import stats
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


class StatisticalRigor:
    """
    Statistical analysis for M-TRACE Phase 2 Experiment 4.
    
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
            results_dir = Path("t_trace/experiments/phase2/exp4/results")
        
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
        capture_rate: float,
        fusion_insight_score: float,
        lime_baseline: float,
        avg_inference_time_ms: float,
        total_cases: int,
        successful_captures: int,
        additional_metrics: Dict = None
    ) -> Path:
        """Save results for a single seed run to JSON"""
        result = {
            "seed": int(seed),
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "mtrace_capture_rate": float(capture_rate),
                "fusion_insight_score": float(fusion_insight_score),
                "lime_baseline": float(lime_baseline),
                "avg_inference_time_ms": float(avg_inference_time_ms),
                "total_cases": int(total_cases),
                "successful_captures": int(successful_captures)
            },
            "additional_metrics": self._convert_numpy_types(additional_metrics or {})
        }
        
        # FIXED: Ensure correct path (no duplicate /raw)
        output_path = self.raw_dir / f"experiment4_seed{seed}_results.json"
        
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
            json_files = list(self.raw_dir.glob("experiment4_seed*_results.json"))
            logger.info(f"Found {len(json_files)} result files: {[f.name for f in json_files]}")
        
        for seed in self.standard_seeds:
            result_file = self.raw_dir / f"experiment4_seed{seed}_results.json"
            
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
        mtrace_capture = [r["metrics"]["mtrace_capture_rate"] for r in results]
        fusion_insight = [r["metrics"]["fusion_insight_score"] for r in results]
        lime_baseline = [r["metrics"]["lime_baseline"] for r in results]
        inference_time = [r["metrics"]["avg_inference_time_ms"] for r in results]
        
        # Compute statistics
        agg_data = {
            "metric": [
                "M-TRACE Capture Rate",
                "Fusion Insight Score",
                "LIME Baseline",
                "Inference Time (ms)"
            ],
            "mean": [
                self.compute_statistics(mtrace_capture)["mean"],
                self.compute_statistics(fusion_insight)["mean"],
                self.compute_statistics(lime_baseline)["mean"],
                self.compute_statistics(inference_time)["mean"]
            ],
            "std": [
                self.compute_statistics(mtrace_capture)["std"],
                self.compute_statistics(fusion_insight)["std"],
                self.compute_statistics(lime_baseline)["std"],
                self.compute_statistics(inference_time)["std"]
            ],
            "n_seeds": [len(results)] * 4,
        }
        
        df = pd.DataFrame(agg_data)
        
        # Save aggregated results
        output_path = self.aggregated_dir / "experiment4_aggregated_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved aggregated results to {output_path}")
        
        return df
    
    def generate_significance_report(
        self,
        mtrace_values: List[float],
        baseline_values: List[float],
        metric_name: str = "capture_rate"
    ) -> str:
        """Generate publication-ready significance testing report"""
        # FIXED: Validate inputs before processing
        if not mtrace_values or not baseline_values:
            logger.error("Cannot generate significance report: empty input arrays")
            return "ERROR: Insufficient data for significance testing"
        
        test_result = self.perform_t_test(mtrace_values, baseline_values, alternative="greater")
        
        report = (
            f"\n{'='*60}\n"
            f"STATISTICAL SIGNIFICANCE REPORT: {metric_name}\n"
            f"{'='*60}\n"
            f"M-TRACE: {self.format_result(mtrace_values, '')}\n"
            f"LIME:    {self.format_result(baseline_values, '')}\n"
            f"{'-'*60}\n"
            f"Welch's t-test Results (one-sided):\n"
            f"  t({test_result['df']}) = {test_result['t_statistic']:.3f}\n"
            f"  p-value = {test_result['p_value']:.4e}\n"
            f"  Significant (p<0.05): {'✅ YES' if test_result['significant'] else '❌ NO'}\n"
            f"  Effect Size (Cohen's d): {test_result['cohens_d']:.3f} ({test_result['effect_magnitude']})\n"
            f"{'='*60}\n"
            f"\nINTERPRETATION:\n"
            f"  M-TRACE captures cross-modal fusion mechanisms during inference\n"
            f"  LIME explains modalities separately (no interaction mechanism)\n"
            f"  Key contribution: Fusion insight, not feature importance\n"
            f"{'='*60}\n"
        )
        
        # Save report
        report_path = self.aggregated_dir / f"significance_report_{metric_name}.txt"
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
        
        # Load seed results for t-test
        seed_results = self._load_all_seed_results()
        mtrace_capture = [r["metrics"]["mtrace_capture_rate"] for r in seed_results]
        lime_baseline = [r["metrics"]["lime_baseline"] for r in seed_results]
        
        test_result = self.perform_t_test(mtrace_capture, lime_baseline, alternative="greater")
        
        latex_table = r"""
\begin{table}[h]
\centering
\caption{Phase 2 Experiment 4: Cross-Modal Fusion Mechanism Capture (5 Random Seeds)}
\label{tab:experiment4_fusion}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{M-TRACE} & \textbf{LIME} & \textbf{p-value} \\
\midrule
Capture Rate & """
        
        mtrace_row = df[df["metric"] == "M-TRACE Capture Rate"]
        lime_row = df[df["metric"] == "LIME Baseline"]
        
        if not mtrace_row.empty:
            latex_table += f"${mtrace_row.iloc[0]['mean']:.3f} \\pm {mtrace_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "N/A & "
        
        if not lime_row.empty:
            latex_table += f"${lime_row.iloc[0]['mean']:.3f} \\pm {lime_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "0.000 (structurally impossible) & "
        
        latex_table += f"${test_result['p_value']:.2e}$ \\\\"
        
        latex_table += r"""
\midrule
Fusion Insight Score & """
        
        insight_row = df[df["metric"] == "Fusion Insight Score"]
        if not insight_row.empty:
            latex_table += f"${insight_row.iloc[0]['mean']:.3f} \\pm {insight_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += r"""N/A & N/A \\
\midrule
Inference Time (ms) & """
        
        time_row = df[df["metric"] == "Inference Time (ms)"]
        if not time_row.empty:
            latex_table += f"${time_row.iloc[0]['mean']:.2f} \\pm {time_row.iloc[0]['std']:.2f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += r"""N/A & N/A \\
\bottomrule
\end{tabular}
\begin{flushleft}
\small
\textit{Note:} LIME explains modalities separately (no fusion mechanism access).
Capture Rate = \% cases where attention data captured during cross-modal fusion.
Statistical significance assessed via Welch's t-test (one-sided, p<0.05). Effect size: Cohen's d.
n=5 random seeds (42, 123, 456, 789, 1011), 50 cases per seed.
\end{flushleft}
\end{table}
"""
        
        # Save LaTeX table
        latex_path = self.aggregated_dir / "experiment4_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved LaTeX table to {latex_path}")
        return latex_table
    
    def generate_capture_rate_plot(self) -> Path:
        """Generate publication-quality capture rate comparison plot."""
        seed_results = self._load_all_seed_results()
        
        if not seed_results:
            logger.warning("No seed results available for plotting")
            return None
        
        mtrace_capture = [r["metrics"]["mtrace_capture_rate"] for r in seed_results]
        lime_baseline = [r["metrics"]["lime_baseline"] for r in seed_results]
        
        # Data for plot
        methods = ['M-TRACE\n(Real-Time)', 'LIME\n(Post-Hoc)']
        scores = [np.mean(mtrace_capture), np.mean(lime_baseline)]
        errors = [np.std(mtrace_capture, ddof=1), np.std(lime_baseline, ddof=1)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors
        color_mtrace = '#2E86AB'
        color_lime = '#A23B72'
        
        # Bar plot with error bars
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, scores, yerr=errors, capsize=8,
                     color=[color_mtrace, color_lime],
                     edgecolor='white', linewidth=2,
                     error_kw={'elinewidth': 2, 'ecolor': 'black'})
        
        # Add value labels on bars
        for i, (bar, score, err) in enumerate(zip(bars, scores, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                   f'{score:.3f}\n±{err:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Perform significance test
        test_result = self.perform_t_test(mtrace_capture, lime_baseline, alternative="greater")
        
        # Add significance indicator
        y_max = max(scores) + max(errors) + 0.1
        ax.plot([0, 0, 1, 1], [y_max, y_max+0.05, y_max+0.05, y_max],
               'k-', linewidth=2)
    
        
        # Formatting
        ax.set_ylabel('Mechanism Capture Rate', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Modal Fusion Mechanism Capture\n(M-TRACE vs LIME Baseline)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        # Add effect size annotation
        effect_text = f"Effect Size (Cohen's d): {test_result['cohens_d']:.3f}\n({test_result['effect_magnitude']})"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=1.0,
                    edgecolor='black', linewidth=2)
        ax.text(0.98, 0.95, effect_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=props)
        
        # Add structural limitation annotation for LIME
        ax.annotate('Structurally Impossible:\nNo access to fusion states',
                   xy=(1, 0.05), xytext=(1.15, 0.15),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=9, color='red', style='italic')
        
        plt.tight_layout()
        
        # Save
        output_path = self.figures_dir / "capture_rate_comparison_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved capture rate comparison plot to {output_path}")
        plt.close()
        
        return output_path


# ============================================================================
# MAIN AGGREGATION SCRIPT
# ============================================================================

def aggregate_experiment4_results(results_dir: Path = None):
    """
    Aggregate all 5 seed results with statistical rigor per FAccT standards.
    """
    if results_dir is None:
        results_dir = Path("t_trace/experiments/phase2/exp4/results")
    
    stats = StatisticalRigor(results_dir=results_dir)
    
    # Load all seed results
    seed_results = stats._load_all_seed_results()
    
    if len(seed_results) < 5:
        logger.warning(f"Only {len(seed_results)}/5 seeds found! Run missing seeds first.")
    
    # Extract metrics
    mtrace_capture = [r["metrics"]["mtrace_capture_rate"] for r in seed_results]
    fusion_insight = [r["metrics"]["fusion_insight_score"] for r in seed_results]
    lime_baseline = [r["metrics"]["lime_baseline"] for r in seed_results]
    inference_time = [r["metrics"]["avg_inference_time_ms"] for r in seed_results]
    
    print("="*70)
    print("PHASE 2 EXPERIMENT 4: STATISTICAL AGGREGATION (5 Random Seeds)")
    print("="*70)
    
    # Print per-seed results
    for i, result in enumerate(seed_results):
        seed = result["seed"]
        capture = result["metrics"]["mtrace_capture_rate"]
        insight = result["metrics"]["fusion_insight_score"]
        lime = result["metrics"]["lime_baseline"]
        time_ms = result["metrics"]["avg_inference_time_ms"]
        cases = result["metrics"]["total_cases"]
        success = result["metrics"]["successful_captures"]
        print(f"✅ Seed {seed:4d}: Capture={capture:.4f}, Insight={insight:.4f}, "
              f"LIME={lime:.4f}, Time={time_ms:.2f}ms ({success}/{cases} cases)")
    
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
    print(f"M-TRACE Capture Rate:    {fmt_stats(mtrace_capture, '')}")
    print(f"Fusion Insight Score:    {fmt_stats(fusion_insight, '')}")
    print(f"LIME Baseline:           {fmt_stats(lime_baseline, '')}")
    print(f"Inference Time:          {fmt_stats(inference_time, ' ms')}")
    
    # Significance testing
    print("\n" + "-"*70)
    print("SIGNIFICANCE TESTING (Welch's t-test, M-TRACE vs LIME)")
    print("-"*70)
    
    test_result = stats.perform_t_test(mtrace_capture, lime_baseline, alternative="greater")
    print(f"t({test_result['df']}) = {test_result['t_statistic']:.3f}")
    print(f"p-value = {test_result['p_value']:.4e}")
    print(f"Statistically Significant (p<0.05): {'✅ YES' if test_result['significant'] else '❌ NO'}")
    print(f"Effect Size (Cohen's d): {test_result['cohens_d']:.3f} ({test_result['effect_magnitude']})")
    
    # Save aggregated results
    output = {
        "experiment": "Phase 2 Experiment 4: Cross-Modal Fusion Mechanism Capture",
        "n_seeds": len(seed_results),
        "seeds_used": [r["seed"] for r in seed_results],
        "mtrace_capture_rate": {
            "mean": float(np.mean(mtrace_capture)),
            "std": float(np.std(mtrace_capture, ddof=1)),
            "report": fmt_stats(mtrace_capture, '')
        },
        "fusion_insight_score": {
            "mean": float(np.mean(fusion_insight)),
            "std": float(np.std(fusion_insight, ddof=1)),
            "report": fmt_stats(fusion_insight, '')
        },
        "lime_baseline": {
            "mean": float(np.mean(lime_baseline)),
            "std": float(np.std(lime_baseline, ddof=1)),
            "report": fmt_stats(lime_baseline, ''),
            "note": "LIME explains modalities separately (no fusion mechanism access)"
        },
        "inference_time_ms": {
            "mean": float(np.mean(inference_time)),
            "std": float(np.std(inference_time, ddof=1)),
            "report": fmt_stats(inference_time, ' ms')
        },
        "significance": {
            "test": "Welch's t-test (one-sided, unequal variances)",
            "t_statistic": float(test_result['t_statistic']),
            "p_value": float(test_result['p_value']),
            "significant": bool(test_result['significant']),
            "cohens_d": float(test_result['cohens_d']),
            "effect_magnitude": test_result['effect_magnitude']
        },
        "key_insight": "M-TRACE captures cross-modal fusion mechanisms during inference; LIME cannot access temporal fusion states"
    }
    
    output_path = stats.aggregated_dir / "experiment4_summary.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Aggregated results saved to: {output_path}")
    print("="*70)
    
    # Generate significance report
    report = stats.generate_significance_report(mtrace_capture, lime_baseline, "capture_rate")
    print(report)
    
    # Generate LaTeX table
    latex_table = stats.generate_latex_table()
    
    # Generate comparison plot
    plot_path = stats.generate_capture_rate_plot()
    
    return output


if __name__ == "__main__":
    results_dir = Path("t_trace/experiments/phase2/exp4/results")
    aggregate_experiment4_results(results_dir)