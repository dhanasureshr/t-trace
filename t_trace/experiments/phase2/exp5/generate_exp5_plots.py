import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# --- 1. LOAD DATA ---
results_path = Path("t_trace/experiments/phase2/exp5/results_exp5_v2.json")
if not results_path.exists():
    print(f"⚠️  Warning: {results_path} not found. Using dummy data for visualization demo.")
    data = {
        "metrics": {
            "mtrace": {"log_file_size_kb": 30.57, "parse_load_time_ms": 7.87, "schema_complexity_score": 23, "found_root_cause": False},
            "shap": {"memory_footprint_kb": 2.34, "total_analysis_time_ms": 3357.01, "found_root_cause": True}
        }
    }
else:
    with open(results_path, 'r') as f:
        data = json.load(f)

mtrace = data['metrics']['mtrace']
shap = data['metrics']['shap']

# --- 2. CONFIGURATION ---
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'mtrace': '#2E86AB',      
    'shap': '#A23B72',        
    'time_axis': '#2E86AB',   
    'storage_axis': '#D9534F',
    'text': '#333333'
}
font = {'family': 'sans-serif', 'weight': 'bold', 'size': 11}
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], **font})

# FIX 1: Increased height from 5 to 6 to prevent vertical clipping
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Experiment 5: Boundary Conditions in Static Aggregation Tasks', fontsize=16, y=0.98, fontweight='bold')

categories = ['M-TRACE', 'SHAP']
x_pos = np.arange(len(categories))
width = 0.35

# ============================================================
# PLOT 1: DUAL-AXIS TRADE-OFF
# ============================================================
ax1 = axs[0]

time_data = [mtrace['parse_load_time_ms'], shap['total_analysis_time_ms']]
storage_data = [mtrace['log_file_size_kb'], shap['memory_footprint_kb']]

# Primary Axis: Time (ms)
bars1 = ax1.bar(x_pos - width/2, time_data, width, label='Analysis Time (ms)', color=colors['time_axis'], edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Analysis Time (ms)', color=colors['time_axis'], fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=colors['time_axis'])
# FIX 2: Relative headroom (20%) instead of fixed pixels
ax1.set_ylim(0, max(time_data) * 1.2) 

# Secondary Axis: Storage (KB)
ax2 = ax1.twinx()
bars2 = ax2.bar(x_pos + width/2, storage_data, width, label='Storage Overhead (KB)', color=colors['storage_axis'], alpha=0.8, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Storage/Memory (KB)', color=colors['storage_axis'], fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=colors['storage_axis'])
ax2.set_ylim(0, max(storage_data) * 1.5) 

# Annotations with Relative Positioning
for i, v in enumerate(time_data):
    # Move text slightly above bar using relative calculation
    ax1.text(i - width/2, v * 1.15, f'{v:.1f}ms', ha='center', va='bottom', fontweight='bold', color=colors['time_axis'], fontsize=10)

for i, v in enumerate(storage_data):
    ax2.text(i + width/2, v * 1, f'{v:.1f}KB', ha='center', va='bottom', fontweight='bold', color=colors['storage_axis'], fontsize=10)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories)
# FIX 3: Added pad to title to push it away from the plot
ax1.set_title('The Trade-Off: Time vs. Storage', fontsize=14, pad=15)

# Legend: Moved to upper right to avoid overlapping with text box
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True, fontsize=9)

# Text Box: Moved down slightly and increased transparency
textstr = 'SHAP: High Compute Cost\nM-TRACE: High I/O Cost'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
# Adjusted position to (0.05, 0.85) to be lower
ax1.text(0.02, 0.85, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# ============================================================
# PLOT 2: COGNITIVE LOAD (Stacked Bar)
# ============================================================
ax3 = axs[1]

mtrace_parts = [4, 16, 3]
shap_parts = [1, 0, 0]
colors_stack = ['#2E86AB', '#4A90C8', '#7AB8E3']

# Plot M-TRACE
for i, part in enumerate(mtrace_parts):
    ax3.bar(0, part, bottom=sum(mtrace_parts[:i]), color=colors_stack[i], edgecolor='white', linewidth=1.5, label=['Metadata', 'Nested States', 'Sparse Indices'][i])

# Plot SHAP
ax3.bar(1, shap_parts[0], color=colors['shap'], edgecolor='white', linewidth=1.5, label='Flat Attribution (SHAP)')

ax3.set_ylabel('Schema Complexity Score\n(Proxy for Human Cognitive Load)', fontsize=12)
# FIX 4: Added pad to title
ax3.set_title('Information Density & Parsing Complexity', fontsize=14, pad=15)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['M-TRACE', 'SHAP'])

# Annotate Totals (Centered vertically within the bar)
ax3.text(0, sum(mtrace_parts)/2, "23\n(High)", ha='center', va='center', color='black', fontsize=16, fontweight='bold')
ax3.text(1, 5, "1\n(Low)", ha='center', va='center', color='black', fontsize=16, fontweight='bold')

# Legend: Moved to lower right to avoid top title overlap
ax3.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)
ax3.set_ylim(0, 28) 

# ============================================================
# PLOT 3: DIAGNOSTIC SUCCESS (Binary)
# ============================================================
ax4 = axs[2]

success = [0, 1] 
bar_colors = [colors['mtrace'], colors['shap']]

bars = ax4.bar(categories, success, color=bar_colors, edgecolor='black', linewidth=1.5, width=0.5)

for i, v in enumerate(success):
    if v == 1:
        ax4.text(i, v + 0.08, 'DETECTED', ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2E7D32')
        ax4.scatter(i, v, s=200, marker='o', color='#2E7D32', zorder=5)
        # Moved sub-text slightly higher to avoid clipping at bottom
        ax4.text(i, -0.12, 'Local Attribution\nEffective', ha='center', va='top', fontsize=10, color='gray', style='italic')
    else:
        ax4.text(i, v + 0.08, 'MISSED', ha='center', va='bottom', fontsize=14, fontweight='bold', color='#C62828')
        ax4.scatter(i, v, s=200, marker='x', color='#C62828', zorder=5, linewidth=3)
        ax4.text(i, -0.12, 'Global State\nInsufficient', ha='center', va='top', fontsize=10, color='gray', style='italic')

# Explicitly set limits with padding
ax4.set_ylim(-0.25, 1.15)
ax4.set_yticks([])
# FIX 5: Added pad to title
ax4.set_title('Root Cause Identification\n(Spurious Feature 0)', fontsize=14, pad=15)
ax4.axhline(0, color='black', linewidth=1, linestyle='--')

ax4.axvspan(0.5, 1.5, color='#2E7D32', alpha=0.05, zorder=0)

# ============================================================
# FINALIZE & SAVE
# ============================================================
# FIX 6: Adjusted rect to give more top/bottom margin
# [left, bottom, right, top] in normalized figure coordinates
plt.tight_layout(rect=[0, 0.05, 1, 0.94]) 

output_path = Path("t_trace/experiments/phase2/exp5/exp5_boundary_conditions_v2.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Enhanced Visualizations saved to: {output_path}")
plt.show()