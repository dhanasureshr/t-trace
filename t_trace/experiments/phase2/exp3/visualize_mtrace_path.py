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


def visualize_mtrace_only(
    target_tree: int = 0,
    target_node: int = 1,
    output_filename: str = "fig_mtrace_path.pdf",
    figsize: tuple = (8, 6),
    highlight_color: str = '#D62728',      # Strong red border
    highlight_fill: str = '#FFC0CB',        # Light pink fill for bias node
    border_width: float = 4.0,              # Emphasized border width
    font_size: int = 7,
    save_dpi: int = 300
):
    """
    Generate M-TRACE Path Visualization with ONLY the biased node highlighted.
    
    This function:
    1. Loads the latest M-TRACE log from Parquet files
    2. Searches for samples traversing the specified biased node
    3. Trains a RandomForest model for visualization (with synthetic fallback)
    4. Plots the decision tree with ONLY target_node highlighted in red
    5. Saves publication-quality output (PDF for vector graphics)
    
    Args:
        target_tree: Tree index to analyze (default: 0)
        target_node: Node index that contains injected bias (default: 1)
        output_filename: Output file path (default: "fig_mtrace_path.pdf")
        figsize: Figure size in inches (default: (8, 6))
        highlight_color: Border color for highlighted node (default: '#D62728')
        highlight_fill: Fill color for highlighted node (default: '#FFC0CB')
        border_width: Border width for highlighted node (default: 4.0)
        font_size: Font size for tree labels (default: 7)
        save_dpi: Resolution for saved figure (default: 300)
    
    Returns:
        str: Path to saved output file, or None if visualization failed
    """
    print("🚀 Generating M-TRACE Path Visualization (Bias Node Only)...")
    
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
    # STEP 2: Train Model for Visualization
    # ========================================================================
    print("🔄 Training model for visualization...")
    
    try:
        # Try to load real data (UCI Adult dataset)
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame.replace('?', np.nan).dropna()
        
        # ✅ STEP 1: Extract target BEFORE encoding categorical columns
        y = (df['class'] == '>50K').astype(int)
        X = df.drop(columns=['class'])
        feature_names = X.columns.tolist()
        
        # ✅ STEP 2: Encode categorical FEATURES ONLY (target already handled)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if X[col].dtype.name == 'category':
                X[col] = X[col].cat.codes.astype(np.int32)
            else:
                codes, _ = pd.factorize(X[col])
                X[col] = codes.astype(np.int32)
                
        # ✅ STEP 3: Verify all features are numeric before training
        assert X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all(), \
            "Non-numeric columns detected in features"
            
        # Train RandomForest with controlled complexity for clear visualization
        clf = RandomForestClassifier(
            n_estimators=10, 
            max_depth=4, 
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X.values.astype(np.float32), y.values)
        print("✅ Loaded real data (UCI Adult) and trained model")
        
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
        print("✅ Trained model on synthetic data")

    # ========================================================================
    # STEP 3: Create Visualization (Publication-Ready with Robust Checks)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))  # Larger figure for multi-node trees

    # Verify tree has structure to plot
    tree = clf.estimators_[0].tree_
    if tree.node_count <= 1:
        print("⚠️  Warning: Tree has no splits (node_count=1). Check data/target encoding.")
        # Add placeholder text
        ax.text(0.5, 0.5, "Tree has no splits\n(Check data preprocessing)", 
                ha='center', va='center', fontsize=12, color='red',
                bbox=dict(boxstyle='round', facecolor='#ffebee', edgecolor='#f44336'))
    else:
        # Plot the first tree from the forest with clean styling
        plot_tree(
            clf.estimators_[0], 
            feature_names=feature_names if len(feature_names) == X.shape[1] else None,
            filled=True,           # Color nodes by class distribution
            rounded=True,          # Rounded corners for aesthetics
            fontsize=max(6, font_size),  # Ensure readable font
            ax=ax,
            impurity=False,        # Hide impurity values for cleaner look
            proportion=True,       # Show class proportions instead of counts
            # CRITICAL: Prevent node overlap with spacing
        )
        
        # Optional: Add node count annotation
        ax.text(0.02, 0.02, f"Nodes: {tree.node_count} | Depth: {tree.max_depth}",
                transform=ax.transAxes, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ========================================================================
    # STEP 4: Highlight ONLY the Biased Node (Key Change)
    # ========================================================================
    # Extract node IDs from the bias path for reference
    path_node_ids = []
    for p_str in bias_path_strings:
        parsed = parse_path_string(p_str)
        if parsed and parsed['tree_id'] == target_tree:
            path_node_ids.append(parsed['node_id'])

    # Apply highlighting: ONLY target_node gets special styling
    for i, patch in enumerate(ax.patches):
        # CRITICAL: Only highlight the specific biased node, not the entire path
        if i == target_node:
            # Strong red border for emphasis
            patch.set_edgecolor(highlight_color)
            patch.set_linewidth(border_width)
            # Light pink fill to distinguish from default node colors
            patch.set_facecolor(highlight_fill)
    
    # ========================================================================
    # STEP 5: Add Annotations (Context without Clutter)
    # ========================================================================
    # Show the full trajectory for context (text-only, no visual highlighting)
    path_label = " → ".join([f"N{n}" for n in path_node_ids[:5]])
    if len(path_node_ids) > 5:
        path_label += "..."
    
    ax.text(
        0.02, 0.98, 
        f"M-TRACE Trajectory:\n{path_label}", 
        transform=ax.transAxes, 
        fontsize=9, 
        verticalalignment='top',
        bbox=dict(
            boxstyle='round', 
            facecolor='#FEEBCE', 
            edgecolor=highlight_color, 
            alpha=0.9
        )
    )
    
    # Clear bias indicator at the specific node
    ax.text(
        0.02, 0.88, 
        f"⚠️ BIAS DETECTED at Node {target_node}", 
        transform=ax.transAxes, 
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
    
    # Title with clear focus
    ax.set_title(
        "M-TRACE: Exact Decision Path (Biased Node Highlighted)", 
        fontsize=12, 
        fontweight='bold', 
        pad=10
    )
    
    # Clean layout
    plt.tight_layout()
    
    # ========================================================================
    # STEP 6: Save and Display
    # ========================================================================
    # Create output directory if needed
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as PDF for vector quality (ideal for publications)
    plt.savefig(
        output_filename, 
        dpi=save_dpi, 
        bbox_inches='tight',
        format='pdf'  # Vector format for crisp text/lines
    )
    print(f"✅ M-TRACE Visualization saved to: {output_filename}")
    
    # Also save PNG for quick preview if needed
    png_path = output_filename.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=save_dpi, bbox_inches='tight')
    print(f"✅ Preview image saved to: {png_path}")
    
    # Display interactive plot
    plt.show()
    
    return output_filename


if __name__ == "__main__":
    # Example usage with default parameters
    result = visualize_mtrace_only(
        target_tree=0,
        target_node=1,
        output_filename="t_trace/experiments/phase2/exp3/results/figures/fig_mtrace_path.pdf"
    )
    
    if result:
        print(f"\n🎉 Visualization complete! Output: {result}")
    else:
        print("\n❌ Visualization failed - check logs for details")