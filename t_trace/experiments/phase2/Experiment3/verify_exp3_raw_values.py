#!/usr/bin/env python3
import json
from pathlib import Path

seeds = [42, 123, 456, 789, 1011]
mtrace_vals, shap_vals = [], []

# Resolve path relative to this script's location (immune to CWD changes)
results_dir = Path(__file__).resolve().parent / "results"
print(f"📂 Searching in: {results_dir}\n")

for seed in seeds:
    # FIX 1: Added 'f' prefix for string interpolation
    path = results_dir / f"exp3_results_seed{seed}.json"
    
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            mtrace_ig = data["metrics"]["mtrace"]["information_gain"]
            shap_ig = data["metrics"]["shap"]["information_gain"]
            mtrace_vals.append(mtrace_ig)
            shap_vals.append(shap_ig)
            print(f"✓ Seed {seed}: M-TRACE IG = {mtrace_ig}, SHAP IG = {shap_ig}")
    else:
        print(f"✗ Seed {seed} not found at {path}")

print(f"\n📊 M-TRACE raw: {mtrace_vals}")
print(f"📊 SHAP raw:     {shap_vals}")