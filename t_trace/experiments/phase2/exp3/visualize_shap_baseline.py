import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_shap_only():
    print("🚀 Generating SHAP Baseline Visualization...")
    
    # 1. Train Model
    print("🔄 Training model for SHAP analysis...")
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
        X_input = X.values[0:1].astype(np.float32)
        
    except Exception as e:
        print(f"⚠️ Data loading failed ({e}). Using synthetic data.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        feature_names = [f"Feat_{i}" for i in range(5)]
        clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
        clf.fit(X, y)
        X_input = X[0:1].astype(np.float32)

    # 2. Calculate SHAP
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values_raw = explainer.shap_values(X_input)
        
        shap_val_sample = None
        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 1: shap_val_sample = shap_values_raw[0][0]
            elif len(shap_values_raw) >= 2: shap_val_sample = shap_values_raw[1][0]
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3: shap_val_sample = shap_values_raw[1, 0, :] if shap_values_raw.shape[0] > 1 else shap_values_raw[0, 0, :]
            elif shap_values_raw.ndim == 2: shap_val_sample = shap_values_raw[0, :]
            else: shap_val_sample = shap_values_raw.flatten()
        
        if shap_val_sample is None: raise ValueError("Could not extract SHAP values")
        shap_val_sample = np.array(shap_val_sample).flatten()
        
    except Exception as e:
        print(f"❌ SHAP Error: {e}")
        return

    # 3. Create Plot (Compact for Paper)
    fig, ax = plt.subplots(figsize=(8, 6)) # Same size as M-TRACE plot
    
    abs_vals = np.abs(shap_val_sample)
    indices = np.argsort(abs_vals)[::-1][:8] # Top 8 for compactness
    
    colors = ['#D62728' if shap_val_sample[i] > 0 else '#1F77B4' for i in indices] # Consistent Red/Blue
    sorted_abs_vals = abs_vals[indices][::-1]
    sorted_labels = [feature_names[i] for i in indices][::-1]
    
    bars = ax.barh(range(len(indices)), sorted_abs_vals, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel('|SHAP Value|', fontsize=10)
    ax.set_title("SHAP: Post-Hoc Attribution", fontsize=12, fontweight='bold', pad=10)
    ax.invert_yaxis()
    
    # Limitation Annotation
    ax.text(0.5, -1.2, "❌ LIMITATION:", transform=ax.transAxes, fontsize=10, weight='bold', color='#D62728', ha='center')
    ax.text(0.5, -1.6, "Shows feature importance magnitude,", transform=ax.transAxes, fontsize=9, style='italic', ha='center', color='#555')
    ax.text(0.5, -2.0, "but CANNOT reveal decision path.", transform=ax.transAxes, fontsize=9, style='italic', ha='center', color='#555')
    ax.text(0.5, -2.4, "(No temporal/structural visibility)", transform=ax.transAxes, fontsize=8, style='italic', ha='center', color='#999')

    plt.tight_layout(rect=[0, 0, 1, 0.85]) # Leave room at bottom for text
    
    # Save
    output_dir = "t_trace/experiments/phase2/exp3/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_shap_baseline.pdf") # PDF for vector quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_shap_only()