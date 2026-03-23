"""
Generate publication-ready tables and figures for Phase 2 Experiment 2.
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
    mtrace_causality = [r["metrics"]["mtrace_causality_score"] for r in seed_results]
    captum_causality = [r["metrics"]["captum_causality_score"] for r in seed_results]
    
    print("\n" + "="*60)
    print("SIGNIFICANCE TESTING REPORT")
    print("="*60)
    report = stats.generate_significance_report(mtrace_causality, captum_causality, "causality_score")
    print(report)
    
    # 4. Generate visualization
    plot_path = stats.generate_causality_comparison_plot()
    print(f"\n✅ Causality comparison plot saved to: {plot_path}")
    
    # 5. Generate layer-wise scatter plot (if data available)
    generate_layerwise_plot(stats)
    
    print("\n✅ All publication artifacts generated!")
    print(f"   Tables: {stats.aggregated_dir}")
    print(f"   Figures: {stats.figures_dir}")


def generate_layerwise_plot(stats: StatisticalRigor):
    """Generate layer-wise causality analysis plot."""
    
    seed_results = stats._load_all_seed_results()
    
    # Collect layer-wise data from all seeds
    all_layer_data = []
    
    for result in seed_results:
        if "mtrace_details" in result and "layer_details" in result["mtrace_details"]:
            layer_details = result["mtrace_details"]["layer_details"]
            for layer in layer_details:
                layer["seed"] = result["seed"]
                all_layer_data.append(layer)
    
    if not all_layer_data:
        print("⚠️ No layer-wise data available for plotting")
        return
    
    df = pd.DataFrame(all_layer_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter plot by layer
    for seed in sorted(df["seed"].unique()):
        seed_data = df[df["seed"] == seed]
        ax.scatter(seed_data["layer_index"], seed_data["correlation"], 
                  alpha=0.6, s=100, label=f"Seed {seed}")
    
    # Add mean line
    mean_corr = df.groupby("layer_index")["correlation"].mean()
    ax.plot(mean_corr.index, mean_corr.values, 'k--', linewidth=2, label='Mean')
    
    # Formatting
    ax.set_xlabel('Transformer Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention-Gradient Correlation', fontsize=12, fontweight='bold')
    ax.set_title('Layer-Wise Causality Analysis (5 Seeds)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(-0.2, 1.0)
    
    plt.tight_layout()
    
    # Save
    output_path = stats.figures_dir / "layerwise_causality_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved layer-wise plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    generate_all_publication_artifacts()