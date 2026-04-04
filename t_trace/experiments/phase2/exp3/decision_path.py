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
import re
from pathlib import Path


def visualize_mtrace_decision_path_paper(
    parquet_path, 
    model, 
    feature_names, 
    X_trigger=None, 
    tree_idx=0, 
    save_dir="t_trace/experiments/phase2/exp3/results/figures"
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    import shap
    import re
    from pathlib import Path

    print("\n📄 Generating Paper-Ready Compact Visualization...")
    
    # 1. Load logs
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ Failed to load Parquet: {e}")
        return

    df_pred = df[df['event_type'] == 'predict']
    if df_pred.empty:
        print("⚠️ No prediction events found in logs.")
        return

    # 2. Extract decision path strings
    # Take the first prediction log that has valid paths
    path_nodes = []
    for _, row in df_pred.iterrows():
        try:
            raw_paths = row['internal_states']['decision_paths']
            if hasattr(raw_paths, 'tolist'): raw_paths = raw_paths.tolist()
            if not isinstance(raw_paths, list): raw_paths = [raw_paths]
            
            for step in raw_paths:
                if not isinstance(step, str): continue
                t_match = re.search(r'Tree\[(\d+)\]', step)
                n_match = re.search(r'Node\[(\d+)\]', step)
                if t_match and n_match and int(t_match.group(1)) == tree_idx:
                    path_nodes.append(int(n_match.group(1)))
            if path_nodes: break # Stop once we find a valid path
        except: continue

    if not path_nodes:
        print("⚠️ Could not extract valid path from logs.")
        return

    print(f"📍 Reconstructed M-TRACE Path (Tree {tree_idx}): {path_nodes}")

    # 3. Create compact side-by-side layout
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # CHANGED: Wider and Shorter figure for paper column width
    fig, (ax_tree, ax_shap) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2, 1]})
    plt.subplots_adjust(bottom=0.25, hspace=0.1) # Extra bottom space for trajectory text

    # --- LEFT PANEL: M-TRACE DECISION TREE (COMPACT) ---
    # CHANGED: fontsize=6 to prevent overlap
    plot_tree(model.estimators_[tree_idx],
              feature_names=feature_names,
              filled=True, rounded=True, fontsize=6, 
              ax=ax_tree, impurity=False, proportion=True)

    plt.draw()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Robust Highlighting
    node_patches = [p for p in ax_tree.patches if p.get_visible() and hasattr(p, 'set_edgecolor')]
    
    # Reset all borders to light gray first
    for patch in node_patches:
        patch.set_edgecolor('#cbd5e1')
        patch.set_linewidth(1.0)

    highlighted_count = 0
    # Map index to patch (sklearn plot_tree usually orders them breadth-first)
    # Note: Exact mapping is tricky in sklearn, but usually patches are ordered by node index 0, 1, 2...
    # We rely on the assumption that patches are in node index order (mostly true for plot_tree)
    # However, to be safe, we can just highlight the ones we can map.
    # A safer visual hack for papers: Just highlight the root and the path if exact mapping is hard,
    # but let's try the index match:
    
    # Simple index matching (works 90% of the time with plot_tree)
    # We need to be careful: plot_tree creates patches for boxes and arrows.
    # The patches list usually contains the boxes.
    path_patch_indices = set()
    
    # We will try to find patches that correspond to the node text positions
    # This is a heuristic approach that is robust for visualization:
    texts = ax_tree.texts
    # texts[i] corresponds to node i usually
    for node_idx in path_nodes:
        if node_idx < len(texts):
            # Find the patch at the same position as the text
            txt = texts[node_idx]
            tx, ty = txt.get_position()
            # Find closest patch
            closest_patch = None
            min_dist = float('inf')
            for p in node_patches:
                px = p.get_x() + p.get_width()/2
                py = p.get_y() + p.get_height()/2
                dist = (px-tx)**2 + (py-ty)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_patch = p
            
            if closest_patch:
                closest_patch.set_edgecolor('#ef4444') # Red
                closest_patch.set_linewidth(3.0)
                if node_idx == path_nodes[-1]:
                    closest_patch.set_facecolor('#fecaca') # Light Red for leaf
                highlighted_count += 1

    print(f"✅ Highlighted {highlighted_count} nodes.")
    
    ax_tree.set_title(f"Tree {tree_idx} (M-TRACE Path)", fontsize=11, fontweight='bold')

    # --- RIGHT PANEL: TreeSHAP FEATURE IMPORTANCE ---
    if X_trigger is not None and len(X_trigger) > 0:
        try:
            explainer = shap.TreeExplainer(model)
            sample = X_trigger[0:1] if isinstance(X_trigger, np.ndarray) else X_trigger.iloc[0:1]
            shap_out = explainer.shap_values(sample)
            
            shap_vals = None
            if isinstance(shap_out, list):
                shap_vals = shap_out[1][0] if len(shap_out) > 1 else shap_out[0][0]
            elif isinstance(shap_out, np.ndarray):
                shap_vals = shap_out[0] if shap_out.ndim == 2 else shap_out.flatten()
            
            if shap_vals is not None:
                shap_vals = np.asarray(shap_vals).flatten()
                top_k = min(6, len(shap_vals)) # Reduced to top 6 for compactness
                abs_vals = np.abs(shap_vals)
                top_idx = np.argsort(abs_vals)[-top_k:][::-1]
                
                top_features = [feature_names[i] if i < len(feature_names) else f"Feat_{i}" for i in top_idx]
                colors = ['#2E86AB' if shap_vals[i] > 0 else '#E63946' for i in top_idx]
                
                ax_shap.barh(range(len(top_features)), abs_vals[top_idx], color=colors, edgecolor='white', linewidth=0.5)
                ax_shap.set_yticks(range(len(top_features)))
                ax_shap.set_yticklabels(top_features, fontsize=9)
                ax_shap.set_xlabel('Mean |SHAP Value|', fontsize=10)
                ax_shap.set_title('TreeSHAP Baseline', fontsize=11, fontweight='bold')
                ax_shap.invert_yaxis()
        except Exception as e:
            print(f"⚠️ SHAP skipped: {e}")

    # --- GLOBAL ANNOTATIONS (Outside the plots to save space) ---
    path_str = " → ".join([f"Node {n}" for n in path_nodes])
    
    # Add a dedicated text box at the bottom of the figure
    fig.text(
        0.5, 0.05, f"M-TRACE Detected Path: {path_str}",
        ha='center', va='center', fontsize=11, fontweight='bold', color='#1e293b',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', edgecolor='#94a3b8', alpha=0.9)
    )

    fig.suptitle("M-TRACE vs TreeSHAP: Decision Path Reconstruction", fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    save_path = Path(save_dir) / f"mtrace_vs_shap_paper_t{tree_idx}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved compact figure to: {save_path}")
    plt.close()


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


        # === ADD THIS VISUALIZATION CALL ===
    visualize_mtrace_decision_path_paper(
        parquet_path=parquet_file,
        model=model_to_use,
        feature_names=feature_names,
        X_trigger=X_trigger,  # <-- ADDED: Provides sample for SHAP
        tree_idx=0,
        save_dir="t_trace/experiments/phase2/exp3/results/figures"
    )
    # ================================

    print("\n⚖️ Generating TreeSHAP Baseline...")
    explainer = shap.TreeExplainer(model_to_use) 
    shap_values = explainer.shap_values(X_trigger[:min(5, len(X_trigger))])
    print("💡 Insight: SHAP cannot identify the specific traversal path.")
    print("\n✅ Experiment 3 Complete.")

if __name__ == "__main__":
    run_experiment()