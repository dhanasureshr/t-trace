import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_log():
    """Find and load the most recent M-TRACE Parquet log."""
    search_pattern = "mtrace_logs/production/*.parquet"
    files = glob.glob(search_pattern)
    if not files:
        # Try development folder too
        search_pattern = "mtrace_logs/development/*.parquet"
        files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No logs found in {search_pattern}")
    
    # Get latest file
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Loading latest log: {latest_file}")
    return pd.read_parquet(latest_file), latest_file

def parse_path_string(path_str):
    """Parse M-TRACE path string: 'Tree[0]:Node[1]->age(<= 17.50):Left'"""
    try:
        if not isinstance(path_str, str):
            return None
            
        # Split by ':' to get Tree part and rest
        parts = path_str.split(":")
        if len(parts) < 2:
            return None
            
        tree_part = parts[0] # "Tree[0]"
        tree_id = int(tree_part.replace("Tree[", "").replace("]", ""))
        
        # The second part contains "Node[X]->..."
        node_part_full = parts[1] # "Node[1]->age(<= 17.50)"
        node_part = node_part_full.split("->")[0] # "Node[1]"
        node_id = int(node_part.replace("Node[", "").replace("]", ""))
        
        # Direction is usually the last part after the final ':'
        direction = parts[-1] if len(parts) > 2 else "Unknown"
        
        return {"tree_id": tree_id, "node_id": node_id, "direction": direction}
    except Exception as e:
        logger.debug(f"Failed to parse '{path_str}': {e}")
        return None

def visualize_experiment_3():
    print("🚀 Generating Experiment 3 Visualization...")
    
    # 1. Load Data & Logs
    try:
        df_logs, log_file = load_latest_log()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Search parameters
    target_tree = 0
    target_node = 1 # We know we injected bias at Node 1
    
    bias_sample_row = None
    bias_path_strings = None
    bias_sample_idx = -1
    
    print(f"🔍 Searching for samples traversing Tree[{target_tree}] Node[{target_node}]...")
    
    # Iterate through all logs to find the bias case
    for idx, row in df_logs.iterrows():
        paths_data = row.get('internal_states', {}).get('decision_paths', [])
        
        # Handle cases where paths_data might be a numpy array or list
        if hasattr(paths_data, 'tolist'):
            paths_data = paths_data.tolist()
            
        if not isinstance(paths_data, list) or not paths_data:
            continue
            
        # Check each step in the path
        path_contains_target = False
        for p_str in paths_data:
            parsed = parse_path_string(p_str)
            if parsed:
                if parsed['tree_id'] == target_tree and parsed['node_id'] == target_node:
                    path_contains_target = True
                    break
        
        if path_contains_target:
            bias_sample_row = row
            bias_path_strings = paths_data
            bias_sample_idx = idx
            print(f"✅ FOUND Bias Case at Log Index {idx}!")
            break
            
    if bias_sample_row is None:
        print("❌ Could not find a sample traversing the biased node in logs.")
        print(f"   Searched {len(df_logs)} logs for Tree[{target_tree}] Node[{target_node}].")
        
        # Debug: Print first few paths to see format
        if len(df_logs) > 0:
            sample_paths = df_logs.iloc[0].get('internal_states', {}).get('decision_paths', [])
            if hasattr(sample_paths, 'tolist'): sample_paths = sample_paths.tolist()
            if sample_paths:
                print(f"   Example path format found: '{sample_paths[0]}'")
                parsed_example = parse_path_string(sample_paths[0])
                if parsed_example:
                    print(f"   Parsed example: {parsed_example}")
                else:
                    print("   Failed to parse example path.")
        return

    print(f"✅ Found biased sample log (Index: {bias_sample_idx}).")
    
        # 2. Reconstruct Model & Data for Visualization
    print("🔄 Preparing data for visualization...")
    
    X = None
    y = None
    feature_names = []
    clf = None
    
    try:
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame
        df = df.replace('?', np.nan).dropna()
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col], _ = pd.factorize(df[col])
        
        target_col = 'class'
        X = df.drop(columns=[target_col])
        y = (df[target_col] == '>50K').astype(int)
        feature_names = X.columns.tolist()
        
        # Convert to numpy explicitly if X is still a DataFrame
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = X.astype(np.float32)
            
        clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        clf.fit(X_np, y.values)
        
    except Exception as e:
        print(f"⚠️ Real data loading failed ({e}). Using synthetic structure for visualization demo.")
        from sklearn.datasets import make_classification
        X_syn, y_syn = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        X = X_syn # X is now a numpy array
        y = y_syn
        feature_names = [f"Feat_{i}" for i in range(5)]
        
        clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        clf.fit(X, y)

    # Ensure X is ready for SHAP (must be 2D array)
    if isinstance(X, pd.DataFrame):
        X_input = X.values[0:1].astype(np.float32)
    else:
        X_input = X[0:1].astype(np.float32)

    # 3. Create Visualization
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    # --- LEFT PANEL: M-TRACE Path Reconstruction ---
    ax1 = axes[0]
    ax1.set_title("M-TRACE: Exact Decision Path Reconstruction\n(Red = Actual Traversal including Biased Node)", fontsize=14, fontweight='bold')
    
    if clf:
        # Plot the tree
        plot_tree(clf.estimators_[0], 
                  feature_names=feature_names, 
                  filled=True, 
                  rounded=True, 
                  fontsize=9, 
                  ax=ax1,
                  impurity=False)
        
        # Highlight the path nodes
        path_node_ids = []
        if bias_path_strings:
            for p_str in bias_path_strings:
                parsed = parse_path_string(p_str)
                if parsed and parsed['tree_id'] == 0: # Only highlight Tree 0
                    path_node_ids.append(parsed['node_id'])
        
        # Highlight borders of path nodes
        for i, patch in enumerate(ax1.patches):
            if i < len(path_node_ids) and i in path_node_ids:
                patch.set_edgecolor('red')
                patch.set_linewidth(4.0)
                if i == target_node:
                    patch.set_facecolor('#ffcccc') # Light red background for bias node
        
        # Add text annotation
        path_label = " -> ".join([f"N{n}" for n in path_node_ids[:6]])
        if len(path_node_ids) > 6: path_label += "..."
        
        ax1.text(0.02, 0.98, f"Sample Path: {path_label}", 
                 transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                 
        ax1.text(0.02, 0.88, f"⚠️ BIAS TRIGGERED AT NODE {target_node}", 
                 transform=ax1.transAxes, fontsize=13, weight='bold', color='darkred',
                 bbox=dict(boxstyle='square', facecolor='white', edgecolor='red', linewidth=2))
    else:
        ax1.text(0.5, 0.5, "Model Training Failed", ha='center', va='center', color='red')

        # --- RIGHT PANEL: SHAP Baseline ---
    ax2 = axes[1]
    ax2.set_title("SHAP: Post-Hoc Feature Attribution\n(Shows 'What' mattered, NOT 'How' or 'Where')", fontsize=14, fontweight='bold')
    
    if clf:
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values_raw = explainer.shap_values(X_input)
            
            shap_val_sample = None
            
            # --- ROBUST SHAPE HANDLING ---
            if isinstance(shap_values_raw, list):
                list_len = len(shap_values_raw)
                if list_len == 1:
                    # Binary classification often returns list of length 1 [positive_class]
                    shap_val_sample = shap_values_raw[0][0]
                elif list_len >= 2:
                    # Multi-class: pick the positive class (index 1)
                    shap_val_sample = shap_values_raw[1][0]
                else:
                    raise ValueError(f"SHAP returned empty list")
                    
            elif isinstance(shap_values_raw, np.ndarray):
                if shap_values_raw.ndim == 3:
                    # (n_classes, n_samples, n_features)
                    if shap_values_raw.shape[0] > 1:
                        shap_val_sample = shap_values_raw[1, 0, :]
                    else:
                        shap_val_sample = shap_values_raw[0, 0, :]
                elif shap_values_raw.ndim == 2:
                    # (n_samples, n_features) -> Directly use sample 0
                    shap_val_sample = shap_values_raw[0, :]
                else:
                    shap_val_sample = shap_values_raw.flatten()
            else:
                raise ValueError(f"Unexpected SHAP output type: {type(shap_values_raw)}")

            # Ensure 1D
            shap_val_sample = np.array(shap_val_sample).flatten()
            
            # Compute values
            abs_vals = np.abs(shap_val_sample)
            indices = np.argsort(abs_vals)[::-1][:10]
            
            colors = []
            for i in indices:
                val = shap_val_sample[i]
                if val > 0:
                    colors.append('red')
                else:
                    colors.append('blue')
            
            sorted_abs_vals = abs_vals[indices][::-1]
            sorted_labels = [feature_names[i] for i in indices][::-1]
            
            bars = ax2.barh(range(len(indices)), sorted_abs_vals, color=colors)
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels(sorted_labels)
            ax2.invert_yaxis()
            ax2.set_xlabel('|SHAP Value| (Magnitude of Impact)')
            
        except Exception as e:
            import traceback
            error_msg = f"SHAP Error:\n{str(e)}\n\n{traceback.format_exc()}"
            ax2.text(0.5, 0.5, error_msg, transform=ax2.transAxes, 
                     ha='center', va='center', color='red', fontsize=7, family='monospace')
            print(f"❌ SHAP Visualization Error: {e}")
    
    # Annotations
    ax2.text(0.5, -1.5, "❌ LIMITATION:", transform=ax2.transAxes, fontsize=12, weight='bold', color='darkred')
    ax2.text(0.5, -2.0, "SHAP shows features are important,", transform=ax2.transAxes, fontsize=11, style='italic')
    ax2.text(0.5, -2.4, "but CANNOT show the decision path.", transform=ax2.transAxes, fontsize=11, style='italic')
    ax2.text(0.5, -2.8, "No temporal or structural trajectory visible.", transform=ax2.transAxes, fontsize=11, style='italic', color='gray')

    plt.tight_layout()
    
    # Save
    output_dir = "t_trace/experiments/phase2/exp3/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp3_path_fidelity_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_experiment_3()