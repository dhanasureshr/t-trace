import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- DATA FROM YOUR EXPERIMENT ---
steps = ['Variable Binding', 'Arithmetic Computation', 'Output Generation']
step_indices = np.arange(len(steps))

# Ground Truth Ranges (from AST) - 1-indexed for display
gt_ranges = [
    (1, 4),   # Bind
    (5, 8),   # Compute
    (9, 12)   # Output
]

# M-TRACE Detections (From your terminal output)
mtrace_detections = [1, 7, 9]

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(10, 6))

# Colors
color_gt = '#E0E0E0'      
color_gt_edge = '#757575' 
color_mtrace = '#2E86AB'  
color_shap = '#A23B72'    
color_text_box = '#FFFFFF' # White background for better contrast

# 1. Plot Ground Truth Bands
for i, (low, high) in enumerate(gt_ranges):
    rect = patches.Rectangle(
        (i - 0.35, low), 
        0.7, 
        (high - low + 1), 
        linewidth=2, 
        edgecolor=color_gt_edge, 
        facecolor=color_gt, 
        alpha=0.3,
        label='Ground Truth (AST)' if i == 0 else ""
    )
    ax.add_patch(rect)
    ax.text(i, (low + high) / 2, f'L{low}-{high}', 
            ha='center', va='center', fontsize=10, color='#555', fontweight='bold')

# 2. Plot M-TRACE Detections
ax.scatter(step_indices, mtrace_detections, 
           s=400, 
           c=color_mtrace, 
           marker='o', 
           edgecolors='white', 
           linewidths=2, 
           zorder=5, 
           label='M-TRACE Detection')

for i, det in enumerate(mtrace_detections):
    ax.annotate(f'Layer {det}', 
                xy=(step_indices[i], det), 
                xytext=(0, 15), 
                textcoords='offset points', 
                ha='center', 
                fontsize=11, 
                fontweight='bold', 
                color=color_mtrace)

ax.plot(step_indices, mtrace_detections, 
        c=color_mtrace, 
        linestyle='--', 
        linewidth=2, 
        alpha=0.6)

# 3. Represent SHAP Limitation (Flat Line at Bottom)
shap_y_pos = 0.5 
ax.hlines(shap_y_pos, -0.5, len(steps)-0.5, 
          colors=color_shap, 
          linestyles='solid', 
          linewidth=3, 
          label='SHAP (Post-Hoc Attribution)',
          zorder=4)

# SHAP Annotation
ax.text(len(steps)/2, shap_y_pos + 1.2, 
        'SHAP: No Temporal/Layer Resolution\n(Single Flat Attribution Map)', 
        ha='center', va='bottom', 
        fontsize=10, color=color_shap, 
        style='italic',
        bbox=dict(facecolor='white', edgecolor=color_shap, alpha=0.8))

# --- AXES FORMATTING ---
# INCREASED Y-LIMIT to create safe space at the top
ax.set_ylim(0, 16) 
ax.set_xlim(-0.6, len(steps) - 0.4)
ax.set_xticks(step_indices)
ax.set_xticklabels(steps, fontsize=11, fontweight='bold')
ax.set_ylabel('Transformer Layer Depth', fontsize=12, fontweight='bold')
ax.set_title('Temporal Fidelity of Reasoning Trajectories', 
             fontsize=14, fontweight='bold', pad=20)

ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mtrace, markersize=15, label='M-TRACE (Real-Time)'),
    Line2D([0], [0], color=color_shap, linewidth=3, label='SHAP (Post-Hoc)'),
    patches.Patch(facecolor=color_gt, edgecolor=color_gt_edge, alpha=0.3, label='Ground Truth (AST)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True, shadow=True)

# 4. FIXED: Metric Text Box (Moved to TOP-LEFT, above the first column)
# Position: x=0.05 (left), y=13.5 (above the highest band which ends at 12)
metric_text = (
    "Temporal Precision: 100%\n"
    "SHAP Temporal Access: 0% (Impossible)"
)
props = dict(boxstyle='round,pad=0.5', facecolor=color_text_box, alpha=1.0, edgecolor='black', linewidth=2)
ax.text(0.01, 12.5, metric_text, transform=ax.transData, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='left', bbox=props)

plt.tight_layout()

# Save High-Res
output_path = "t_trace/experiments/phase2/exp1/results/temporal_trajectory_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Advanced Trajectory Plot saved to {output_path}")

plt.show()