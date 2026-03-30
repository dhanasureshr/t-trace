"""
Generate publication-ready tables and figures for Phase 2 Experiment 5.
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
    mtrace_costs = [r["metrics"]["mtrace_cost_ms"] for r in seed_results]
    shap_costs = [r["metrics"]["shap_cost_ms"] for r in seed_results]
    tri_scores = [r["metrics"]["temporal_redundancy_index"] for r in seed_results]
    
    print("\n" + "="*60)
    print("SIGNIFICANCE TESTING REPORT")
    print("="*60)
    report = stats.generate_significance_report(mtrace_costs, shap_costs, tri_scores)
    print(report)
    
    # 4. Generate visualization
    plot_path = stats.generate_tri_comparison_plot()
    print(f"\n✅ TRI comparison plot saved to: {plot_path}")
    
    # 5. Reuse existing figures from generate_exp5_plots.py
    print("\n✅ Reusing existing visualizations from:")
    print("   - t_trace/experiments/phase2/exp5/generate_exp5_plots.py")
    print("   - exp5_boundary_conditions_v2.png")
    
    print("\n✅ All publication artifacts generated!")
    print(f"   Tables: {stats.aggregated_dir}")
    print(f"   Figures: {stats.figures_dir}")


if __name__ == "__main__":
    generate_all_publication_artifacts()