import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import shap
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
        # Try development folder as fallback
        search_pattern = "mtrace_logs/development/*.parquet"
        files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No logs found in {search_pattern}")
    
    # Get latest file by creation time
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"📂 Loading latest log: {latest_file}")
    return pd.read_parquet(latest_file), latest_file


def parse_path_string(path_str):
    """
    Parse M-TRACE path string format: 'Tree[0]:Node[1]->age(<= 17.50):Left'
    
    Returns:
        dict with keys: tree_id, node_id, direction
        or None if parsing fails
    """
    try:
        if not isinstance(path_str, str):
            return None
        
        # Split by ':' to separate Tree part from rest
        parts = path_str.split(":")
        if len(parts) < 2:
            return None
        
        # Extract tree_id: "Tree[0]" -> 0
        tree_part = parts[0]
        tree_id = int(tree_part.replace("Tree[", "").replace("]", ""))
        
        # Extract node_id: "Node[1]->..." -> 1
        node_part_full = parts[1]
        node_part = node_part_full.split("->")[0]  # Get "Node[1]" before the arrow
        node_id = int(node_part.replace("Node[", "").replace("]", ""))
        
        # Extract direction (optional, usually last part)
        direction = parts[-1] if len(parts) > 2 else "Unknown"
        
        return {
            "tree_id": tree_id,
            "node_id": node_id,
            "direction": direction
        }
    except Exception as e:
        logger.debug(f"Failed to parse '{path_str}': {e}")
        return None


def visualize_mtrace_with_shap(
    target_tree: int = 0,
    target_node: int = 1,
    output_filename: str = "fig_mtrace_with_shap.pdf",
    figsize: tuple = (16, 10),
    highlight_color: str = '#D62728',      # Strong red border
    highlight_fill: str = '#FFC0CB',        # Light pink fill for bias node
    border_width: float = 4.0,              # Emphasized border width
    font_size: int = 7,
    save_dpi: int = 300
):
    """
    Generate M-TRACE Path Visualization with SHAP Feature Importance.
    
    This function:
    1. Loads the latest M-TRACE log from Parquet files
    2. Searches for samples traversing the specified biased node
    3. Trains a RandomForest model for visualization (with real data)
    4. Calculates SHAP values for feature importance
    5. Creates a comprehensive visualization showing:
       - LEFT: Decision tree with biased node highlighted
       - RIGHT: SHAP feature importance bar chart
    6. Saves publication-quality output (PDF for vector graphics)
    
    Args:
        target_tree: Tree index to analyze (default: 0)
        target_node: Node index that contains injected bias (default: 1)
        output_filename: Output file path (default: "fig_mtrace_with_shap.pdf")
        figsize: Figure size in inches (default: (16, 10))
        highlight_color: Border color for highlighted node (default: '#D62728')
        highlight_fill: Fill color for highlighted node (default: '#FFC0CB')
        border_width: Border width for highlighted node (default: 4.0)
        font_size: Font size for tree labels (default: 7)
        save_dpi: Resolution for saved figure (default: 300)
    
    Returns:
        str: Path to saved output file, or None if visualization failed
    """
    print("🚀 Generating M-TRACE Path Visualization with SHAP Feature Importance...")
    
    # ========================================================================
    # STEP 1: Load Logs and Find Bias Case
    # ========================================================================
    try:
        df_logs, log_file = load_latest_log()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None

    bias_path_strings = None
    bias_sample_idx = None
    
    print(f"🔍 Searching for bias case (Tree[{target_tree}] Node[{target_node}])...")
    
    # Iterate through logs to find a sample that traverses the biased node
    for idx, row in df_logs.iterrows():
        paths_data = row.get('internal_states', {}).get('decision_paths', [])
        
        # Handle numpy array conversion if needed
        if hasattr(paths_data, 'tolist'):
            paths_data = paths_data.tolist()
        
        if not isinstance(paths_data, list) or not paths_data:
            continue
        
        # Check each path string in the decision path
        for p_str in paths_data:
            parsed = parse_path_string(p_str)
            if parsed and parsed['tree_id'] == target_tree and parsed['node_id'] == target_node:
                bias_path_strings = paths_data
                bias_sample_idx = idx
                print(f"✅ FOUND Bias Case at Log Index {idx}!")
                break
        
        if bias_path_strings:
            break
    
    if not bias_path_strings:
        print(f"❌ Could not find a sample traversing Tree[{target_tree}] Node[{target_node}]")
        return None

    # ========================================================================
    # STEP 2: Train Model and Calculate SHAP Values
    # ========================================================================
    print("🔄 Training model and calculating SHAP values...")
    
    try:
        # Load real data (UCI Adult dataset)
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame.replace('?', np.nan).dropna()
        
        # Extract target BEFORE encoding
        y = (df['class'] == '>50K').astype(int)
        X = df.drop(columns=['class'])
        feature_names = X.columns.tolist()
        
        # Encode categorical features
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if X[col].dtype.name == 'category':
                X[col] = X[col].cat.codes.astype(np.int32)
            else:
                codes, _ = pd.factorize(X[col])
                X[col] = codes.astype(np.int32)
        
        # Verify all features are numeric
        assert X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all(), \
            "Non-numeric columns detected in features"
        
        # Train RandomForest
        clf = RandomForestClassifier(
            n_estimators=10, 
            max_depth=4, 
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X.values.astype(np.float32), y.values)
        
        # Calculate SHAP values
        print("  Calculating SHAP values...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X.values.astype(np.float32))
        
        # Handle SHAP output format (can be list for binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Calculate mean absolute SHAP values for feature importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        print("✅ Loaded real data (UCI Adult), trained model, and calculated SHAP values")
        
    except Exception as e:
        print(f"⚠️ Real data loading failed ({e}). Using synthetic data for visualization.")
        from sklearn.datasets import make_classification
        
        # Generate synthetic data as fallback
        X, y = make_classification(
            n_samples=100, 
            n_features=5, 
            n_informative=3,
            n_redundant=0,
            random_state=42
        )
        feature_names = [f"Feat_{i}" for i in range(5)]
        
        clf = RandomForestClassifier(
            n_estimators=10, 
            max_depth=4, 
            random_state=42
        )
        clf.fit(X, y)
        
        # Calculate SHAP for synthetic data
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        print("✅ Trained model on synthetic data and calculated SHAP values")

    # ========================================================================
    # STEP 3: Create Comprehensive Visualization
    # ========================================================================
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout: Left (tree) takes 60%, Right (SHAP) takes 40%
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 4], wspace=0.3)
    
    # Left panel: Decision Tree
    ax_tree = fig.add_subplot(gs[0, 0])
    
    # Right panel: SHAP Feature Importance
    ax_shap = fig.add_subplot(gs[0, 1])
    
    # ========================================================================
    # Plot Decision Tree (Left Panel)
    # ========================================================================
    tree = clf.estimators_[0].tree_
    if tree.node_count <= 1:
        print("⚠️  Warning: Tree has no splits (node_count=1). Check data/target encoding.")
        ax_tree.text(0.5, 0.5, "Tree has no splits\n(Check data preprocessing)", 
                    ha='center', va='center', fontsize=12, color='red',
                    bbox=dict(boxstyle='round', facecolor='#ffebee', edgecolor='#f44336'))
    else:
        # Plot the tree
        plot_tree(
            clf.estimators_[0], 
            feature_names=feature_names if len(feature_names) == X.shape[1] else None,
            filled=True,
            rounded=True,
            fontsize=max(6, font_size),
            ax=ax_tree,
            impurity=False,
            proportion=True
        )
        
        # Add node count annotation
        ax_tree.text(0.02, 0.02, f"Nodes: {tree.node_count} | Depth: {tree.max_depth}",
                    transform=ax_tree.transAxes, fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Highlight the biased node
        path_node_ids = []
        for p_str in bias_path_strings:
            parsed = parse_path_string(p_str)
            if parsed and parsed['tree_id'] == target_tree:
                path_node_ids.append(parsed['node_id'])
        
        for i, patch in enumerate(ax_tree.patches):
            if i == target_node:
                patch.set_edgecolor(highlight_color)
                patch.set_linewidth(border_width)
                patch.set_facecolor(highlight_fill)
        
        # Add trajectory annotation
        path_label = " → ".join([f"N{n}" for n in path_node_ids[:5]])
        if len(path_node_ids) > 5:
            path_label += "..."
        
        ax_tree.text(
            0.02, 0.98, 
            f"M-TRACE Trajectory:\n{path_label}", 
            transform=ax_tree.transAxes, 
            fontsize=9, 
            verticalalignment='top',
            bbox=dict(
                boxstyle='round', 
                facecolor='#FEEBCE', 
                edgecolor=highlight_color, 
                alpha=0.9
            )
        )
        
        ax_tree.text(
            0.02, 0.88, 
            f"⚠️ BIAS at Node {target_node}", 
            transform=ax_tree.transAxes, 
            fontsize=10, 
            weight='bold', 
            color=highlight_color,
            bbox=dict(
                boxstyle='square', 
                facecolor='white', 
                edgecolor=highlight_color, 
                linewidth=1.5
            )
        )
    
    ax_tree.set_title(
        "M-TRACE: Exact Decision Path", 
        fontsize=12, 
        fontweight='bold', 
        pad=10
    )
    
    # Replace the SHAP visualization section (around line 330-350) with this corrected version:

    # ========================================================================
    # Plot SHAP Feature Importance (Right Panel)
    # ========================================================================
    # Get top 10 features by SHAP importance
    # Ensure shap_importance is 1D
    if shap_importance.ndim > 1:
        shap_importance = shap_importance.mean(axis=0)  # Average across samples if multi-dimensional
    
    n_features = min(10, len(shap_importance))
    top_indices = np.argsort(shap_importance)[-n_features:][::-1]
    top_importance = shap_importance[top_indices]
    
    # Convert numpy indices to Python integers to avoid TypeError
    top_indices_list = [int(idx) for idx in top_indices]
    top_features = [feature_names[i] for i in top_indices_list]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(top_features))
    colors = ['#D62728' if shap_values[:, i].mean() > 0 else '#1F77B4' for i in top_indices_list]
    
    bars = ax_shap.barh(y_pos, top_importance, color=colors, edgecolor='gray', linewidth=0.5)
    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels(top_features, fontsize=9)
    ax_shap.set_xlabel('Mean |SHAP Value|', fontsize=10)
    ax_shap.set_title('SHAP: Feature Importance', fontsize=12, fontweight='bold', pad=10)
    ax_shap.invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_importance)):
        ax_shap.text(
            val + 0.001, 
            i, 
            f'{val:.4f}', 
            va='center', 
            fontsize=8,
            color='#333'
        )
    
    # Add annotation explaining SHAP
    ax_shap.text(
        0.5, -1.15,
        "Shows feature importance magnitude\n(but NOT decision path)",
        transform=ax_shap.transAxes,
        fontsize=9,
        style='italic',
        ha='center',
        color='#555',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8)
    )
    
    # ========================================================================
    # Add Overall Title
    # ========================================================================
    fig.suptitle(
        "M-TRACE vs SHAP: Decision Path vs Feature Importance",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # ========================================================================
    # STEP 4: Save and Display
    # ========================================================================
    # Create output directory if needed
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as PDF for vector quality
    plt.savefig(
        output_filename, 
        dpi=save_dpi, 
        bbox_inches='tight',
        format='pdf'
    )
    print(f"✅ M-TRACE + SHAP Visualization saved to: {output_filename}")
    
    # Also save PNG for quick preview
    png_path = output_filename.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=save_dpi, bbox_inches='tight')
    print(f"✅ Preview image saved to: {png_path}")
    
    # Display interactive plot
    plt.show()
    
    return output_filename


if __name__ == "__main__":
    # Example usage with default parameters
    result = visualize_mtrace_with_shap(
        target_tree=0,
        target_node=1,
        output_filename="t_trace/experiments/phase2/exp3/results/figures/fig_mtrace_with_shap.png"
    )
    
    if result:
        print(f"\n🎉 Visualization complete! Output: {result}")
    else:
        print("\n❌ Visualization failed - check logs for details")