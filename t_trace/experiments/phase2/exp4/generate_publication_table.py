"""
Generate publication-ready tables and figures for Phase 2 Experiment 4.
"""

import pandas as pd
from pathlib import Path
from statistical_analysis import StatisticalRigor
import matplotlib.pyplot as plt
import numpy as np


def generate_all_publication_artifacts():
    """Generate all tables and figures for paper submission."""
    
    stats = StatisticalRigor()
    
    # 1. Aggregate results
    print("="*60)
    print("AGGREGATED RESULTS (5 Seeds)")
    print("="*60)
    df = stats.aggregate_all_seeds()
    print(df.to_string(index=False))
    
    # 2. Generate LaTeX table
    print("\n" + "="*60)
    print("LaTeX TABLE GENERATED")
    print("="*60)
    latex_table = stats.generate_latex_table()
    print(latex_table)
    
    # 3. Generate significance report
    seed_results = stats._load_all_seed_results()
    mtrace_capture = [r["metrics"]["mtrace_capture_rate"] for r in seed_results]
    lime_baseline = [r["metrics"]["lime_baseline"] for r in seed_results]
    
    print("\n" + "="*60)
    print("SIGNIFICANCE TESTING REPORT")
    print("="*60)
    report = stats.generate_significance_report(mtrace_capture, lime_baseline, "capture_rate")
    print(report)
    
    # 4. Generate visualization
    plot_path = stats.generate_capture_rate_plot()
    print(f"\n✅ Capture rate comparison plot saved to: {plot_path}")
    
    # 5. Reuse existing figures from generate_figures.py
    print("\n✅ Reusing existing visualizations from:")
    print("   - t_trace/experiments/phase2/exp4/generate_figures.py")
    print("   - fig4_capture_rate.pdf")
    print("   - fig4_overhead_distribution.pdf")
    
    print("\n✅ All publication artifacts generated!")
    print(f"   Tables: {stats.aggregated_dir}")
    print(f"   Figures: {stats.figures_dir}")


if __name__ == "__main__":
    generate_all_publication_artifacts()