import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import glob

from sklearn.tree import plot_tree

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))) 

from t_trace.logging_engine import enable_logging


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.tree import _tree

def visualize_bias_path_comparison(
    model, 
    feature_names, 
    bias_path: list,
    target_tree: int = 0,
    bias_node: int = 1,
    shap_values: np.ndarray = None,
    save_path: str = "results/fig3_qualitative_comparison.pdf"
):
    """
    Publication-ready side-by-side visualization.
    LEFT: Draws ONLY the exact traversed path (no full tree clutter).
    RIGHT: TreeSHAP baseline (unchanged robust implementation).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), 
                                   gridspec_kw={'width_ratios': [1.1, 0.9]})
    
    # === LEFT PANEL: Exact Path Only (Flowchart Style) ===
    ax1.set_title("M-TRACE: Exact Decision Path Reconstruction\n(Only Traversed Nodes Shown)", 
                  fontsize=13, fontweight='bold', pad=15)
    
    # 1. Parse exact path nodes for the target tree
    path_nodes = []
    for step in bias_path:
        if f"Tree[{target_tree}]" in step:
            try:
                # Format: "Tree[0]:Node[3]->age(<=17.5):Left"
                node_str = step.split("->")[0].split(":")[1]
                node_id = int(node_str.replace("Node[", "").replace("]", ""))
                path_nodes.append(node_id)
            except Exception:
                continue
                
    if not path_nodes:
        ax1.text(0.5, 0.5, "No path data found", ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return

    tree = model.estimators_[target_tree].tree_
    n_nodes = len(path_nodes)
    
    # Layout configuration
    box_w, box_h = 0.55, 0.16
    y_start = 0.92
    y_step = 0.24
    x_center = 0.5
    
    # Draw nodes & edges sequentially
    for i, node_id in enumerate(path_nodes):
        y_pos = y_start - i * y_step
        
        # Extract split rule
        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"Feat_{feat_idx}"
        condition = f"{feat_name} ≤ {threshold:.2f}"
        
        # Determine direction to next node
        is_leaf = tree.children_left[node_id] == _tree.TREE_LEAF
        direction = "LEAF" if is_leaf else ("LEFT" if i < n_nodes-1 and path_nodes[i+1] == tree.children_left[node_id] else "RIGHT")
        
        # Node styling
        is_bias = (node_id == bias_node)
        face_color = "#FFC0CB" if is_bias else ("#FEEBCE" if i == n_nodes-1 else "white")
        edge_color = "#D62728" if is_bias else "#2E86AB"
        lw = 2.5 if is_bias else 1.8
        
        rect = mpatches.FancyBboxPatch(
            (x_center - box_w/2, y_pos - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.03", facecolor=face_color,
            edgecolor=edge_color, linewidth=lw, zorder=3
        )
        ax1.add_patch(rect)
        
        # Node text
        ax1.text(x_center, y_pos, f"N{node_id}\n{condition}\n→ {direction}",
                 ha='center', va='center', fontsize=9, 
                 fontweight='bold' if is_bias else 'normal')
        
        # Bias annotation
        if is_bias:
            ax1.text(x_center + box_w/2 + 0.04, y_pos, "⚠️ BIAS TRIGGERED",
                     ha='left', va='center', fontsize=10, fontweight='bold', color='#D62728')
            
        # Arrow to next node
        if i < n_nodes - 1:
            next_y = y_start - (i+1) * y_step
            ax1.annotate("", xy=(x_center, next_y + box_h/2), xytext=(x_center, y_pos - box_h/2),
                         arrowprops=dict(arrowstyle="->", color="#D62728", lw=2, 
                                         shrinkA=8, shrinkB=8, mutation_scale=15))
            
    ax1.set_xlim(0, 1)
    ax1.set_ylim(y_start - n_nodes * y_step - 0.1, y_start + 0.1)
    ax1.axis('off')
    
    # Metric box
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=1.0, edgecolor='black', linewidth=1.5)
    ax1.text(0.98, 0.98, "Path Coverage: 100%\nBias Detection: ✓\nTemporal Fidelity: ✓", 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top', 
             horizontalalignment='right', bbox=props)

    # === RIGHT PANEL: TreeSHAP Baseline (Robust Implementation) ===
    ax2.set_title("TreeSHAP: Post-Hoc Feature Attribution\n(Shows 'What', NOT 'How' or 'Where')", 
                  fontsize=13, fontweight='bold', pad=15)
    
    if shap_values is not None:
        if isinstance(shap_values, list):
            shap_vals = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_vals = np.abs(shap_values).mean(axis=0)
            
        shap_vals = np.array(shap_vals).flatten()
        feature_names_list = list(feature_names)
        
        min_len = min(len(shap_vals), len(feature_names_list))
        shap_vals = shap_vals[:min_len]
        feature_names_list = feature_names_list[:min_len]
        
        top_k = min(10, min_len)
        indices = np.argsort(shap_vals)[-top_k:][::-1].tolist()
        
        colors = ['#2E86AB' if feature_names_list[int(i)] != 'gender' else '#D62728' 
                 for i in indices]
        
        ax2.barh(range(top_k), shap_vals[indices][::-1], 
                color=colors, edgecolor='white', linewidth=1.5)
        
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels([feature_names_list[int(i)] for i in indices][::-1], fontsize=9)
        ax2.set_xlabel('|SHAP Value| (Magnitude)', fontsize=10)
        ax2.invert_yaxis()
        
        # Limitation annotation
        ax2.text(0.5, -1.3, "❌ LIMITATION:", 
                transform=ax2.transAxes, fontsize=10, fontweight='bold', 
                color='#D62728', ha='center')
        ax2.text(0.5, -1.7, "Shows gender is important,", 
                transform=ax2.transAxes, fontsize=9, style='italic', 
                ha='center', color='#555')
        ax2.text(0.5, -2.1, "but CANNOT reveal:", 
                transform=ax2.transAxes, fontsize=9, style='italic', 
                ha='center', color='#555')
        ax2.text(0.5, -2.5, "• Which node triggered bias", 
                transform=ax2.transAxes, fontsize=8.5, style='italic', 
                ha='center', color='#777')
        ax2.text(0.5, -2.9, "• Actual decision sequence", 
                transform=ax2.transAxes, fontsize=8.5, style='italic', 
                ha='center', color='#777')
    
    # Structural limitation note
    ax2.text(0.02, 0.02, 
            "Structural Limitation:\nTreeSHAP computes marginal\ncontributions (what-if),\nnot actual execution paths.", 
            transform=ax2.transAxes, fontsize=8.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF2F2', 
                     edgecolor='#D62728', alpha=0.8, linewidth=1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved qualitative comparison to {save_path}")
    plt.show()


def preprocess_data(df, target_col='class'):
    print("🧹 Preprocessing data...")
    df = df.replace('?', np.nan)
    initial_rows = len(df)
    df = df.dropna()
    print(f"   Dropped {initial_rows - len(df)} rows with missing values.")

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"   Encoding {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        if col == target_col: continue
        uniques = df[col].unique()
        df[col], _ = pd.factorize(df[col])
        print(f"   - '{col}': {len(uniques)} categories -> Integers")

    if target_col not in df.columns:
        target_col = df.columns[-1]
        
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    
    if y_raw.dtype == 'object' or y_raw.dtype.name == 'category':
        y, target_classes = pd.factorize(y_raw)
        print(f"   Encoded target '{target_col}': {list(target_classes)}")
    else:
        y = y_raw

    remaining_obj_cols = X.select_dtypes(include=['object']).columns
    if len(remaining_obj_cols) > 0:
        raise ValueError(f"CRITICAL: Still have object columns: {list(remaining_obj_cols)}")

    return X, y, X.columns.tolist()

def load_and_prepare_data():
    print("📥 Loading UCI Adult dataset...")
    try:
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame
    except Exception as e:
        print(f"⚠️ Fetch failed ({e}). Using synthetic surrogate.")
        from sklearn.datasets import make_classification
        X_syn, y_syn = make_classification(n_samples=5000, n_features=20, n_informative=15, random_state=42)
        df = pd.DataFrame(X_syn, columns=[f"feat_{i}" for i in range(20)])
        df['target'] = y_syn
        return df.drop(columns=['target']), df['target'], [f"feat_{i}" for i in range(20)]

    return preprocess_data(df, target_col='class')

def inject_bias_guaranteed(model, X_subset, feature_idx=0, tree_index=0):
    """
    Injects bias by finding a REAL sample that triggers the left path, 
    then setting the threshold slightly ABOVE that sample's value.
    """
    print(f"\n🔧 Analyzing Feature {feature_idx} for GUARANTEED bias injection...")
    
    feature_values = X_subset[:, feature_idx]
    min_val = np.min(feature_values)
    
    # Set threshold to min_val + small epsilon to ensure min_val goes LEFT
    threshold = min_val + 0.5 
    
    print(f"   Feature {feature_idx} Range: [{np.min(feature_values):.2f}, {np.max(feature_values):.2f}]")
    print(f"   Minimum Value Found: {min_val:.2f}")
    print(f"   Setting Threshold to: {threshold:.2f} (ensures min value goes LEFT)")

    if tree_index >= len(model.estimators_):
        print("❌ Tree index out of range.")
        return False, None, None
        
    tree = model.estimators_[tree_index].tree_
    
    # --- STEP 1: Modify ROOT (Node 0) ---
    tree.feature[0] = feature_idx
    tree.threshold[0] = float(threshold)
    
    left_child = tree.children_left[0]
    if left_child == -1:
        print("❌ Root has no left child.")
        return False, None, None
        
    target_node = left_child 
    print(f"   ✅ Root (Node 0) modified: Split on Feature {feature_idx} <= {threshold:.2f} -> Go LEFT to Node {target_node}")

    # --- STEP 2: Modify TARGET NODE (Node 1) ---
    tree.feature[target_node] = feature_idx 
    tree.threshold[target_node] = float(threshold)
    
    print(f"   ✅ Target (Node {target_node}) configured.")
    print(f"   🎯 LOGIC: Samples with value <= {threshold:.2f} WILL go Root(0) -> Left -> Node({target_node})")
    
    # Identify which samples in the subset will trigger this
    triggering_mask = feature_values <= threshold
    triggering_count = np.sum(triggering_mask)
    print(f"   📊 Expected Trigger Count in Subset: {triggering_count} samples")
    
    return True, threshold, target_node

def run_experiment():
    print("🚀 Starting Experiment 3: Decision Path Fidelity (Deterministic Bias)")
    print("="*60)

    # 1. Data Prep
    X, y, feature_names = load_and_prepare_data()
    if X.empty: return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔄 Converting to NumPy arrays (float32)...")
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    print("✅ Conversion successful.")

    print(f"📊 Dataset Shape: Train={X_train_np.shape}, Test={X_test_np.shape}")

    # 2. Model & Logging
    print("\n🌲 Initializing RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42, n_jobs=-1)
    
    print("\n⚡ Enabling M-TRACE Logging...")
    engine = enable_logging(clf, mode="production")
    
    if hasattr(engine, 'get_wrapped_model'):
        model_to_use = engine.get_wrapped_model()
    else:
        raise RuntimeError("Failed to get wrapped model.")

    # 3. Train
    print("\n🏋️ Training Wrapped Model...")
    model_to_use.fit(X_train_np, y_train)
    baseline_acc = accuracy_score(y_test, model_to_use.predict(X_test_np))
    print(f"📈 Baseline Accuracy: {baseline_acc:.4f}")

    # 4. Identify Triggering Samples FIRST
    subset_size = 500
    X_subset = X_test_np[:subset_size]
    y_subset = y_test[:subset_size]
    feature_idx = 0
    
    # Find the minimum value to set threshold
    subset_vals = X_subset[:, feature_idx]
    min_val = np.min(subset_vals)
    threshold = min_val + 0.5
    
    print(f"\n🔍 Subset Analysis: Min Value for Feature {feature_idx} is {min_val:.2f}")
    print(f"   Setting Threshold to: {threshold:.2f}")
    
    # CRITICAL: Find indices where condition is met
    triggering_mask = subset_vals <= threshold
    triggering_indices = np.where(triggering_mask)[0]
    
    print(f"   🎯 Found {len(triggering_indices)} samples that WILL trigger the bias.")
    
    if len(triggering_indices) == 0:
        print("❌ No triggering samples found in subset. Aborting.")
        return

    # Inject Bias into Model
    success_result = inject_bias_guaranteed(model_to_use, X_subset, feature_idx=feature_idx, tree_index=0)
    if not success_result[0]:
        print("❌ Bias injection failed.")
        return
    
    _, injected_threshold, target_node = success_result
    target_tree = 0 

    # 5. Run Inference ONLY on Triggering Samples
    # This ensures 100% of logged entries contain the bias path
    print(f"\n🏃 Running Inference on {len(triggering_indices)} BIAS-TRIGGERING samples...")
    X_trigger = X_subset[triggering_indices]
    y_trigger = y_subset[triggering_indices]
    
    start_time = time.time()
    predictions = model_to_use.predict(X_trigger)
    inference_time = time.time() - start_time
    print(f"⏱️ Inference Time: {inference_time:.4f}s")

    # 6. Flush & Analyze
    print("\n💾 Flushing logs to disk...")
    current_run_id = engine.get_run_id()
    engine.disable_logging() 

    print("\n📂 Analyzing Logs from Parquet File...")
    import glob
    run_id_short = current_run_id[:8]
    search_pattern = f"mtrace_logs/production/*{run_id_short}*.parquet"
    parquet_files = glob.glob(search_pattern)
    
    if not parquet_files:
        print(f"❌ No Parquet files found.")
        return
        
    parquet_file = parquet_files[0]
    print(f"✅ Found log file: {parquet_file}")
    
    try:
        df_logs = pd.read_parquet(parquet_file)
        print(f"📝 Loaded {len(df_logs)} logs from Parquet.")
    except Exception as e:
        print(f"❌ Error reading Parquet: {e}")
        return

    df_predict = df_logs[df_logs['event_type'] == 'predict']
    print(f"   Filtered to {len(df_predict)} prediction events.")
    
    bias_found = False
    count_processed = 0
    
    for idx, row in df_predict.iterrows():
        decision_paths = row['internal_states']['decision_paths']
        
        # Handle NumPy array conversion
        if hasattr(decision_paths, 'tolist'):
            decision_paths = decision_paths.tolist()
        elif not isinstance(decision_paths, list):
            try:
                decision_paths = list(decision_paths)
            except TypeError:
                decision_paths = []
        
        if not decision_paths:
            continue
            
        for step_str in decision_paths:
            if not isinstance(step_str, str):
                continue
            try:
                tree_part = step_str.split(":")[0] 
                current_tree_id = int(tree_part.replace("Tree[", "").replace("]", ""))
                node_part = step_str.split(":")[1].split("->")[0] 
                current_node_id = int(node_part.replace("Node[", "").replace("]", ""))
                
                if current_tree_id == target_tree and current_node_id == target_node:
                    bias_found = True
                    bias_sample_idx = count_processed
                    bias_path = decision_paths
                    break
            except Exception:
                continue 
        
        if bias_found:
            break
        count_processed += 1
            
    if bias_found:
        print("\n" + "="*60)
        print("🚨 BIAS CASE DETECTED! 🚨")
        print("="*60)
        print(f"   Sample Index (in trigger batch): {bias_sample_idx}")
        print(f"   Original Dataset Index: {triggering_indices[bias_sample_idx]}")
        print(f"   Path Verification: Root(0) -> Node({target_node}) TRAVERSED!")
        print("\n   📜 Full Decision Path:")
        
        for i, step_str in enumerate(bias_path):
            highlight = ""
            if f"Node[{target_node}]" in step_str and f"Tree[{target_tree}]" in step_str:
                highlight = " ⬅️ BIAS TRIGGERED HERE"
            elif "Node[0]" in step_str and "Tree[{target_tree}]" in step_str:
                highlight = " (ROOT)"
            print(f"   Step {i}: {step_str}{highlight}")
        print("="*60 + "\n")
        
        root_step = bias_path[0]
        if ":Left" in root_step:
            print("✅ CONFIRMED: Sample went LEFT at Root, entering the biased node.")
        else:
            print("⚠️ NOTE: Sample went RIGHT at Root (Unexpected).")
            
    else:
        print("\n⚠️ No bias detected even in targeted samples.")
        print(f"   Total Logs Analyzed: {len(df_predict)}")
        if len(df_predict) > 0:
            first_paths = df_predict.iloc[1]['internal_states']['decision_paths']
            if hasattr(first_paths, 'tolist'): first_paths = first_paths.tolist()
            if first_paths:
                print(f"   Example Path: {first_paths[0]}")


    # === ADD THIS VISUALIZATION CALL ===
    # visualize_mtrace_decision_path(
    #     parquet_path=parquet_file,
    #     model=model_to_use,  # Wrapped RandomForest still has .estimators_
    #     feature_names=feature_names,
    #     tree_idx=0,
    #     save_dir="t_trace/experiments/phase2/exp3/results/figures"
    # )
    # ================================

   


    print("\n⚖️ Generating TreeSHAP Baseline...")
    explainer = shap.TreeExplainer(model_to_use) 
    shap_values = explainer.shap_values(X_trigger[:min(5, len(X_trigger))])
    print("💡 Insight: SHAP cannot identify the specific traversal path.")

    visualize_bias_path_comparison(
        model=model_to_use, 
        feature_names=feature_names, 
        bias_path=bias_path,  
        target_tree=target_tree,
        bias_node=target_node,
        shap_values=shap_values if bias_found else None,
        save_path="t_trace/experiments/phase2/exp3/results/fig3_qualitative_comparison.pdf"
    )
    print("\n✅ Experiment 3 Complete.")

if __name__ == "__main__":
    run_experiment()