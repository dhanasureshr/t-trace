"""
Statistical Rigor Implementation for Phase 2 Experiment 2
FIXED: Path handling, empty array validation, two-sided t-test, and publication-quality plotting
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
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


class StatisticalRigor:
    """
    Statistical analysis for M-TRACE Phase 2 Experiment 2.
    
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
            results_dir = Path("t_trace/experiments/phase2/exp2/results")
        
        self.results_dir = results_dir
        
        # FIXED: Ensure raw_dir doesn't duplicate /raw
        if str(self.results_dir).endswith('/raw'):
            self.raw_dir = self.results_dir  # Already ends with /raw
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
    
    def compute_statistics(self, values: List[float]) -> Dict[str, float]:
        """Compute comprehensive statistics with EMPTY ARRAY VALIDATION"""
        if not values or len(values) == 0:
            logger.warning("Empty values list provided to compute_statistics")
            return {
                "mean": 0.0, "std": 0.0, "sem": 0.0, "n": 0,
                "ci_95_lower": 0.0, "ci_95_upper": 0.0,
                "min": 0.0, "max": 0.0
            }
        
        arr = np.array(values)
        n = len(arr)
        
        if n < 2:
            logger.warning(f"Only {n} sample(s) - statistics may be unreliable")
        
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        
        if n > 1:
            ci_margin = stats.t.ppf(0.975, df=n-1) * sem
        else:
            ci_margin = 0.0
        
        return {
            "mean": mean, "std": std, "sem": sem, "n": n,
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
        if stats_dict["std"] < 1e-10:
            return f"{stats_dict['mean']:.3f} (std={stats_dict['std']:.3f}) {metric_name}"
        return f"{stats_dict['mean']:.3f} ± {stats_dict['std']:.3f} {metric_name}"
    
    def perform_t_test(
        self,
        group_a: List[float],
        group_b: List[float],
        alternative: str = "two-sided"
    ) -> Dict[str, float]:
        """Perform Welch's t-test with validation"""
        if not group_a or not group_b or len(group_a) < 2 or len(group_b) < 2:
            logger.warning("Insufficient samples for t-test")
            return {
                "t_statistic": 0.0, "p_value": 1.0, "significant": False,
                "cohens_d": 0.0, "effect_magnitude": "unknown", "df": 0
            }
        
        t_stat, p_value = stats.ttest_ind(
            group_a, group_b, equal_var=False, alternative=alternative
        )
        
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
            if d < 0.2: return "negligible"
            elif d < 0.5: return "small"
            elif d < 0.8: return "medium"
            else: return "large"
        
        return {
            "t_statistic": float(t_stat), "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "cohens_d": float(effect_size),
            "effect_magnitude": interpret_effect_size(effect_size),
            "df": len(group_a) + len(group_b) - 2
        }
    
    def save_seed_result(
        self, seed: int, causality_score: float, captum_baseline: float,
        training_time_sec: float, overhead_pct: float, active_layers: int,
        additional_metrics: Dict = None
    ) -> Path:
        """Save results for a single seed run to JSON"""
        result = {
            "seed": seed, "timestamp": datetime.now().isoformat(),
            "metrics": {
                "mtrace_causality_score": causality_score,
                "captum_causality_score": captum_baseline,
                "training_time_sec": training_time_sec,
                "overhead_percentage": overhead_pct,
                "active_layers_analyzed": active_layers
            },
            "additional_metrics": additional_metrics or {}
        }
        output_path = self.raw_dir / f"experiment2_seed{seed}_results.json"
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
            json_files = list(self.raw_dir.glob("experiment2_seed*_results.json"))
            logger.info(f"Found {len(json_files)} result files: {[f.name for f in json_files]}")
        
        for seed in self.standard_seeds:
            result_file = self.raw_dir / f"experiment2_seed{seed}_results.json"
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
        
        mtrace_causality = [r["metrics"]["mtrace_causality_score"] for r in results]
        captum_causality = [r["metrics"]["captum_causality_score"] for r in results]
        training_time = [r["metrics"]["training_time_sec"] for r in results]
        overhead_pct = [r["metrics"]["overhead_percentage"] for r in results]
        active_layers = [r["metrics"]["active_layers_analyzed"] for r in results]
        
        agg_data = {
            "metric": [
                "M-TRACE Causality Score", "Captum Causality Score",
                "Training Time (sec)", "Overhead (%)", "Active Layers Analyzed"
            ],
            "mean": [
                self.compute_statistics(mtrace_causality)["mean"],
                self.compute_statistics(captum_causality)["mean"],
                self.compute_statistics(training_time)["mean"],
                self.compute_statistics(overhead_pct)["mean"],
                self.compute_statistics(active_layers)["mean"]
            ],
            "std": [
                self.compute_statistics(mtrace_causality)["std"],
                self.compute_statistics(captum_causality)["std"],
                self.compute_statistics(training_time)["std"],
                self.compute_statistics(overhead_pct)["std"],
                self.compute_statistics(active_layers)["std"]
            ],
            "n_seeds": [len(results)] * 5,
        }
        df = pd.DataFrame(agg_data)
        output_path = self.aggregated_dir / "experiment2_aggregated_results.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved aggregated results to {output_path}")
        return df
    
    def generate_significance_report(
        self, mtrace_values: List[float], baseline_values: List[float],
        metric_name: str = "causality_score"
    ) -> str:
        """Generate publication-ready significance testing report"""
        if not mtrace_values or not baseline_values:
            logger.error("Cannot generate significance report: empty input arrays")
            return "ERROR: Insufficient data for significance testing"
        
        test_result = self.perform_t_test(mtrace_values, baseline_values, alternative="two-sided")
        
        report = (
            f"\n{'='*60}\n"
            f"STATISTICAL SIGNIFICANCE REPORT: {metric_name}\n"
            f"{'='*60}\n"
            f"M-TRACE: {self.format_result(mtrace_values, '')}\n"
            f"Captum:  {self.format_result(baseline_values, '')}\n"
            f"{'-'*60}\n"
            f"Welch's t-test Results (two-sided):\n"
            f"  t({test_result['df']}) = {test_result['t_statistic']:.3f}\n"
            f"  p-value = {test_result['p_value']:.4e}\n"
            f"  Significant (p<0.05): {'✅ YES' if test_result['significant'] else '❌ NO'}\n"
            f"  Effect Size (Cohen's d): {test_result['cohens_d']:.3f} ({test_result['effect_magnitude']})\n"
            f"{'='*60}\n"
            f"\nINTERPRETATION:\n"
            f"  M-TRACE captures temporally-aligned gradient-attention dynamics\n"
            f"  Captum baseline is simulated (0.32) per literature estimates\n"
            f"  Key contribution: Temporal alignment, not aggregate correlation score\n"
            f"{'='*60}\n"
        )
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
        
        seed_results = self._load_all_seed_results()
        mtrace_causality = [r["metrics"]["mtrace_causality_score"] for r in seed_results]
        captum_causality = [r["metrics"]["captum_causality_score"] for r in seed_results]
        test_result = self.perform_t_test(mtrace_causality, captum_causality, alternative="two-sided")
        
        latex_table = r"""
\begin{table}[h]
\centering
\caption{Phase 2 Experiment 2: Gradient-Attention Causality Verification (5 Random Seeds)}
\label{tab:experiment2_causality}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{M-TRACE} & \textbf{Captum} & \textbf{p-value} \\
\midrule
Causality Score & """
        
        mtrace_row = df[df["metric"] == "M-TRACE Causality Score"]
        captum_row = df[df["metric"] == "Captum Causality Score"]
        
        if not mtrace_row.empty:
            latex_table += f"${mtrace_row.iloc[0]['mean']:.3f} \\pm {mtrace_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "N/A & "
        
        if not captum_row.empty:
            latex_table += f"${captum_row.iloc[0]['mean']:.3f} \\pm {captum_row.iloc[0]['std']:.3f}$ & "
        else:
            latex_table += "0.320 (simulated) & "
        
        latex_table += f"${test_result['p_value']:.2e}$ \\\\"
        
        latex_table += r"""
\midrule
Training Overhead (\%) & """
        
        overhead_row = df[df["metric"] == "Overhead (%)"]
        if not overhead_row.empty:
            latex_table += f"${overhead_row.iloc[0]['mean']:.2f} \\pm {overhead_row.iloc[0]['std']:.2f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += r"""N/A & N/A \\
\midrule
Active Layers Analyzed & """
        
        layers_row = df[df["metric"] == "Active Layers Analyzed"]
        if not layers_row.empty:
            latex_table += f"${layers_row.iloc[0]['mean']:.1f} \\pm {layers_row.iloc[0]['std']:.1f}$ & "
        else:
            latex_table += "N/A & "
        
        latex_table += r"""N/A & N/A \\
\bottomrule
\end{tabular}
\begin{flushleft}
\small
\textit{Note:} Captum baseline estimated via post-hoc approximation (literature value: 0.32).
M-TRACE provides temporally-aligned gradient-attention capture (same backward pass).
Statistical significance assessed via Welch's t-test (two-sided, p<0.05). Effect size: Cohen's d.
n=5 random seeds (42, 123, 456, 789, 1011).
\end{flushleft}
\end{table}
"""
        latex_path = self.aggregated_dir / "experiment2_table.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"Saved LaTeX table to {latex_path}")
        return latex_table
    
    def generate_causality_comparison_plot(self) -> Path:
        """
        Generate publication-quality causality comparison plot.
        
        FIXED: Removed invalid 'capthick' parameter from ax.bar()
        """
        seed_results = self._load_all_seed_results()
        
        if not seed_results:
            logger.warning("No seed results available for plotting")
            return None
        
        mtrace_causality = [r["metrics"]["mtrace_causality_score"] for r in seed_results]
        captum_causality = [r["metrics"]["captum_causality_score"] for r in seed_results]
        
        # Data for plot
        methods = ['M-TRACE\n(Real-Time)', 'Captum\n(Post-Hoc)']
        scores = [np.mean(mtrace_causality), np.mean(captum_causality)]
        errors = [np.std(mtrace_causality, ddof=1), np.std(captum_causality, ddof=1)]
        
        # Create figure with publication-ready settings
        fig, ax = plt.subplots(figsize=(10, 7), dpi=300, facecolor='white')
        
        # Colors (colorblind-safe, publication-friendly)
        color_mtrace = '#2E86AB'   # Blue
        color_captum = '#A23B72'   # Purple
        
        # Bar plot with error bars
        x_pos = np.arange(len(methods))
        bar_width = 0.6
        
        # FIXED: Removed invalid 'capthick' parameter
        # Error bar styling goes in error_kw dict
        bars = ax.bar(
            x_pos, scores, 
            yerr=errors, 
            capsize=6,  # Length of cap lines in points
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},  # Error bar styling
            color=[color_mtrace, color_captum],
            edgecolor='white', 
            linewidth=1.5,
            width=bar_width,
            zorder=3,
            label=['M-TRACE', 'Captum']
        )
        
        # Add value labels on bars (positioned to avoid overlap)
        for i, (bar, score, err) in enumerate(zip(bars, scores, errors)):
            height = bar.get_height()
            label_y = height + err + 0.035 if err > 0.001 else height + 0.035
            
            # Main value (bold, white background)
            ax.text(
                bar.get_x() + bar.get_width()/2., label_y,
                f'{score:.3f}',
                ha='center', va='bottom', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.9, linewidth=0.5)
            )
            # Std dev (smaller, below value)
            if err > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width()/2., label_y - 0.045,
                    f'±{err:.3f}',
                    ha='center', va='top',
                    fontsize=9, fontweight='normal', color='#555555'
                )
        
        # Perform significance test
        test_result = self.perform_t_test(mtrace_causality, captum_causality, alternative="two-sided")
        
        # Add significance indicator (bracket above bars)
        y_max = max([s + e for s, e in zip(scores, errors)]) + 0.10
        bracket_height = 0.045
        
        # Draw bracket lines
        ax.plot([x_pos[0], x_pos[0]], [y_max, y_max + bracket_height], 
            'k-', linewidth=1.5, zorder=2)
        ax.plot([x_pos[1], x_pos[1]], [y_max, y_max + bracket_height], 
            'k-', linewidth=1.5, zorder=2)
        ax.plot([x_pos[0], x_pos[1]], [y_max + bracket_height, y_max + bracket_height], 
            'k-', linewidth=1.5, zorder=2)
        
        # Add p-value and significance marker
        if test_result['p_value'] < 0.001:
            p_text = "p < 0.001"
        elif test_result['p_value'] < 0.01:
            p_text = f"p = {test_result['p_value']:.3f}"
        else:
            p_text = f"p = {test_result['p_value']:.3f}"
        
        sig_marker = '*' if test_result['significant'] else 'ns'
        sig_color = 'red' if test_result['significant'] else '#777777'
        
        ax.text(0.5, y_max + bracket_height + 0.025, 
            f"{p_text}\n{sig_marker}",
            ha='center', va='bottom', 
            fontsize=11, fontweight='bold',
            color=sig_color)
        
        # Axis labels and title
        ax.set_ylabel('Causality Detection Rate (Pearson r)', 
                    fontsize=13, fontweight='bold', labelpad=12)
        ax.set_title('Gradient-Attention Causality Verification',
                    fontsize=15, fontweight='bold', pad=25)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=11, fontweight='semibold')
        
        # Y-axis limits with intelligent padding
        y_upper = max([s + e for s, e in zip(scores, errors)]) + 0.18
        ax.set_ylim(0, min(y_upper, 1.05))  # Cap slightly above 1.0 for correlation
        
        # Professional grid and spine styling
        ax.yaxis.grid(True, linestyle='--', alpha=0.35, linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)
        
        # Remove top/right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
        # Add effect size annotation (top-right corner)
        effect_text = f"Cohen's d: {test_result['cohens_d']:.3f}\n({test_result['effect_magnitude']})"
        effect_props = dict(boxstyle='round,pad=0.7', facecolor='#f8f9fa', alpha=0.98,
                        edgecolor='#dee2e6', linewidth=1.2)
        ax.text(0.975, 0.975, effect_text, transform=ax.transAxes, 
            fontsize=10, fontweight='normal',
            verticalalignment='top', horizontalalignment='right',
            bbox=effect_props)
        
        # Add sample size annotation (top-left corner)
        ax.text(0.025, 0.975, f"n = {len(mtrace_causality)} random seeds", 
            transform=ax.transAxes, fontsize=9.5, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e9ecef', alpha=0.95,
                        edgecolor='#ced4da', linewidth=0.8))
        
        # Add horizontal reference line at Captum baseline (subtle)
        if captum_causality and np.mean(captum_causality) > 0:
            ax.axhline(y=np.mean(captum_causality), 
                    color=color_captum, linestyle=':', 
                    linewidth=1, alpha=0.4, zorder=1,
                    label='Captum baseline')
        
        # Add subtle legend for reference line
        ax.legend(loc='lower right', fontsize=9, frameon=True, 
                framealpha=0.9, edgecolor='#dee2e6')
        
        # Final layout adjustment
        plt.tight_layout()
        
        # Save in multiple formats for publication flexibility
        output_path_png = self.figures_dir / "causality_comparison_plot.png"
        output_path_pdf = self.figures_dir / "causality_comparison_plot.pdf"
        output_path_svg = self.figures_dir / "causality_comparison_plot.svg"
        
        # PNG for web/presentations
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
        
        # PDF for LaTeX manuscripts (vector graphics)
        plt.savefig(output_path_pdf, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf')
        
        # SVG for editable vector graphics
        plt.savefig(output_path_svg, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='svg')
        
        logger.info(f"✅ Saved causality comparison plot:")
        logger.info(f"   PNG: {output_path_png}")
        logger.info(f"   PDF: {output_path_pdf}")
        logger.info(f"   SVG: {output_path_svg}")
        
        plt.close()
        
        return output_path_png


# ============================================================================
# MAIN AGGREGATION SCRIPT
# ============================================================================

def aggregate_experiment2_results(results_dir: Path = None):
    """Aggregate all 5 seed results with statistical rigor per FAccT standards."""
    if results_dir is None:
        results_dir = Path("t_trace/experiments/phase2/exp2/results")
    
    stats = StatisticalRigor(results_dir=results_dir)
    seed_results = stats._load_all_seed_results()
    
    if len(seed_results) < 5:
        logger.warning(f"Only {len(seed_results)}/5 seeds found! Run missing seeds first.")
    
    mtrace_causality = [r["metrics"]["mtrace_causality_score"] for r in seed_results]
    captum_causality = [r["metrics"]["captum_causality_score"] for r in seed_results]
    training_time = [r["metrics"]["training_time_sec"] for r in seed_results]
    overhead_pct = [r["metrics"]["overhead_percentage"] for r in seed_results]
    
    print("="*70)
    print("PHASE 2 EXPERIMENT 2: STATISTICAL AGGREGATION (5 Random Seeds)")
    print("="*70)
    
    for i, result in enumerate(seed_results):
        seed = result["seed"]
        mtrace = result["metrics"]["mtrace_causality_score"]
        captum = result["metrics"]["captum_causality_score"]
        time_sec = result["metrics"]["training_time_sec"]
        overhead = result["metrics"]["overhead_percentage"]
        print(f"✅ Seed {seed:4d}: M-TRACE={mtrace:.4f}, Captum={captum:.4f}, "
              f"Time={time_sec:.2f}s (+{overhead:.1f}%)")
    
    def fmt_stats(values, suffix=""):
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std < 1e-10:
            return f"{mean:.3f} (std={std:.3f}){suffix}"
        return f"{mean:.3f} ± {std:.3f}{suffix}"
    
    print("\n" + "-"*70)
    print("AGGREGATED RESULTS (mean ± std, n=5 seeds)")
    print("-"*70)
    print(f"M-TRACE Causality Score: {fmt_stats(mtrace_causality, '')}")
    print(f"Captum Causality Score:  {fmt_stats(captum_causality, '')}")
    print(f"Training Time:           {fmt_stats(training_time, ' sec')}")
    print(f"Overhead %:              {fmt_stats(overhead_pct, ' %')}")
    
    print("\n" + "-"*70)
    print("SIGNIFICANCE TESTING (Welch's t-test, two-sided)")
    print("-"*70)
    
    test_result = stats.perform_t_test(mtrace_causality, captum_causality, alternative="two-sided")
    print(f"t({test_result['df']}) = {test_result['t_statistic']:.3f}")
    print(f"p-value = {test_result['p_value']:.4e}")
    print(f"Statistically Significant (p<0.05): {'✅ YES' if test_result['significant'] else '❌ NO'}")
    print(f"Effect Size (Cohen's d): {test_result['cohens_d']:.3f} ({test_result['effect_magnitude']})")
    
    output = {
        "experiment": "Phase 2 Experiment 2: Gradient-Attention Causality",
        "n_seeds": len(seed_results),
        "seeds_used": [r["seed"] for r in seed_results],
        "mtrace_causality_score": {
            "mean": float(np.mean(mtrace_causality)),
            "std": float(np.std(mtrace_causality, ddof=1)),
            "report": fmt_stats(mtrace_causality, '')
        },
        "captum_causality_score": {
            "mean": float(np.mean(captum_causality)),
            "std": float(np.std(captum_causality, ddof=1)),
            "report": fmt_stats(captum_causality, ''),
            "note": "Simulated baseline (0.32) per literature estimates"
        },
        "training_time_sec": {
            "mean": float(np.mean(training_time)),
            "std": float(np.std(training_time, ddof=1)),
            "report": fmt_stats(training_time, ' sec')
        },
        "overhead_percentage": {
            "mean": float(np.mean(overhead_pct)),
            "std": float(np.std(overhead_pct, ddof=1)),
            "report": fmt_stats(overhead_pct, ' %')
        },
        "significance": {
            "test": "Welch's t-test (two-sided, unequal variances)",
            "t_statistic": float(test_result['t_statistic']),
            "p_value": float(test_result['p_value']),
            "significant": bool(test_result['significant']),
            "cohens_d": float(test_result['cohens_d']),
            "effect_magnitude": test_result['effect_magnitude']
        },
        "key_insight": "M-TRACE provides temporal alignment (same backward pass) that Captum cannot access"
    }
    
    output_path = stats.aggregated_dir / "experiment2_summary.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Aggregated results saved to: {output_path}")
    print("="*70)
    
    report = stats.generate_significance_report(mtrace_causality, captum_causality, "causality_score")
    print(report)
    
    latex_table = stats.generate_latex_table()
    plot_path = stats.generate_causality_comparison_plot()
    
    return output


if __name__ == "__main__":
    results_dir = Path("t_trace/experiments/phase2/exp2/results")
    aggregate_experiment2_results(results_dir)