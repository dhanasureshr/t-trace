import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_log():
    search_pattern = "mtrace_logs/production/*.parquet"
    files = glob.glob(search_pattern)
    if not files:
        search_pattern = "mtrace_logs/development/*.parquet"
        files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No logs found in {search_pattern}")
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Loading latest log: {latest_file}")
    return pd.read_parquet(latest_file), latest_file

def parse_path_string(path_str):
    try:
        if not isinstance(path_str, str): return None
        parts = path_str.split(":")
        if len(parts) < 2: return None
        
        tree_part = parts[0]
        tree_id = int(tree_part.replace("Tree[", "").replace("]", ""))
        
        node_part_full = parts[1]
        node_part = node_part_full.split("->")[0]
        node_id = int(node_part.replace("Node[", "").replace("]", ""))
        
        direction = parts[-1] if len(parts) > 2 else "Unknown"
        return {"tree_id": tree_id, "node_id": node_id, "direction": direction}
    except Exception as e:
        logger.debug(f"Failed to parse '{path_str}': {e}")
        return None

def visualize_mtrace_only():
    print("🚀 Generating M-TRACE Path Visualization...")
    
    # 1. Load Logs
    try:
        df_logs, log_file = load_latest_log()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    target_tree = 0
    target_node = 1 
    bias_path_strings = None
    
    print(f"🔍 Searching for bias case (Tree[{target_tree}] Node[{target_node}])...")
    
    for idx, row in df_logs.iterrows():
        paths_data = row.get('internal_states', {}).get('decision_paths', [])
        if hasattr(paths_data, 'tolist'): paths_data = paths_data.tolist()
        if not isinstance(paths_data, list) or not paths_data: continue
            
        for p_str in paths_data:
            parsed = parse_path_string(p_str)
            if parsed and parsed['tree_id'] == target_tree and parsed['node_id'] == target_node:
                bias_path_strings = paths_data
                print(f"✅ FOUND Bias Case at Log Index {idx}!")
                break
        if bias_path_strings: break
            
    if not bias_path_strings:
        print("❌ Could not find a sample traversing the biased node.")
        return

    # 2. Train Model for Viz
    print("🔄 Training model for visualization...")
    try:
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame.replace('?', np.nan).dropna()
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols: df[col], _ = pd.factorize(df[col])
        
        X = df.drop(columns=['class'])
        y = (df['class'] == '>50K').astype(int)
        feature_names = X.columns.tolist()
        
        clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        clf.fit(X.values.astype(np.float32), y.values)
    except Exception as e:
        print(f"⚠️ Data loading failed ({e}). Using synthetic data.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        feature_names = [f"Feat_{i}" for i in range(5)]
        clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        clf.fit(X, y)

    # 3. Create Plot (Compact for Paper)
    fig, ax = plt.subplots(figsize=(8, 6)) # Compact size
    
    # Plot Tree
    plot_tree(clf.estimators_[0], 
              feature_names=feature_names, 
              filled=True, 
              rounded=True, 
              fontsize=7, 
              ax=ax,
              impurity=False,
              proportion=True) # Use proportion for cleaner boxes
    
    # Highlight Path
    path_node_ids = []
    for p_str in bias_path_strings:
        parsed = parse_path_string(p_str)
        if parsed and parsed['tree_id'] == 0:
            path_node_ids.append(parsed['node_id'])
    
    for i, patch in enumerate(ax.patches):
        if i < len(path_node_ids) and i in path_node_ids:
            patch.set_edgecolor('#D62728') # Strong Red
            patch.set_linewidth(3.0)
            if i == target_node:
                patch.set_facecolor('#FFC0CB') # Light Pink for Bias Node
    
    # Annotations
    path_label = " → ".join([f"N{n}" for n in path_node_ids[:5]])
    if len(path_node_ids) > 5: path_label += "..."
    
    ax.text(0.02, 0.98, f"M-TRACE Trajectory:\n{path_label}", 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#FEEBCE', edgecolor='#D62728', alpha=0.9))
            
    ax.text(0.02, 0.88, f"⚠️ BIAS at NODE {target_node}", 
            transform=ax.transAxes, fontsize=10, weight='bold', color='#D62728',
            bbox=dict(boxstyle='square', facecolor='white', edgecolor='#D62728', linewidth=1.5))

    ax.set_title("M-TRACE: Exact Decision Path", fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save
    output_dir = "t_trace/experiments/phase2/exp3/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_mtrace_path.pdf") # PDF for vector quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ M-TRACE Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_mtrace_only()