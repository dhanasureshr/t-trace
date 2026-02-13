#!/usr/bin/env python3
"""
Gap 3 Validation: Sparse Reconstruction Fidelity
Verifies per-head attention preservation for trajectory reconstruction.
Target: Ubuntu workstation (RTX 4080 Super)
"""
import os
import sys
import torch
import numpy as np
import tempfile
import yaml
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from transformers import BertTokenizer, BertModel, BertConfig
from t_trace.logging_engine import enable_logging

def create_temp_config():
    """Create temporary config.yml with Gap 3 sparse settings."""
    config = {
        "mode": "development",
        "run_id": "gap3_validation",
        "sparse_logging": {
            "enabled": True,
            "sparse_threshold": 0.1,
            "top_k_values": 5,
            "top_k_per_head": 3  # CRITICAL FOR GAP 3
        },
        "compression": {
            "enabled": True,
            "compression_type": "snappy",
            "compression_level": 1
        },
        "custom_fields": ["attention_weights"],
        "default_fields": ["model_type", "framework", "timestamp", "run_id", "mode", "layer_name", "losses"]
    }
    
    config_path = Path("config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

def run_gap3_validation():
    print("="*70)
    print("GAP 3 VALIDATION: Trajectory-Fidelity Sparse Filtering")
    print("="*70)
    
    # Step 1: Create config with per-head top-k setting
    config_path = create_temp_config()
    print(f"✓ Created config.yml with top_k_per_head=3")
    
    # Step 2: Load BERT with attention output enabled
    print("\nLoading BERT-base-uncased...")
    config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', config=config)
    print(f"✓ BERT loaded (12 layers, 12 attention heads)")
    
    # Step 3: Enable M-TRACE logging (loads config.yml automatically)
    print("\nEnabling M-TRACE logging...")
    engine = enable_logging(model, mode="development")
    #print(f"✓ Logging enabled (run_id: {engine._model_metadata.get('run_id', 'N/A')})")
    
    # Step 4: Run Winograd inference
    text = "The trophy wouldn't fit in the suitcase because it was too large."
    print(f"\nRunning inference on Winograd example:\n  '{text}'")
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Step 5: Collect execution-ordered logs
    logs = engine.collect_logs()
    attention_logs = [log for log in logs if log["internal_states"]["attention_weights"]]
    print(f"✓ Collected {len(logs)} total logs ({len(attention_logs)} with attention weights)")
    
    # Step 6: VALIDATE GAP 3 REQUIREMENTS
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    if not attention_logs:
        print("✗ CRITICAL FAILURE: No attention logs captured!")
        print("  → Verify output_attentions=True and hook attachment to SelfAttention submodules")
        sys.exit(1)
    
    sample = attention_logs[0]
    metadata = sample.get("sparse_logging_metadata", {})
    
    # Validation 1: Per-head top-k activation
    sparse_type = metadata.get("attention_sparse_type", "MISSING")
    print(f"\n[Validation 1] Sparse filtering mode:")
    print(f"  Sparse type: {sparse_type}")
    
    if sparse_type == "per_head_top_k":
        print("  ✓ PASSED: Per-head top-k filtering ACTIVE (trajectory fidelity guaranteed)")
    else:
        print(f"  ⚠ WARNING: Expected 'per_head_top_k', got '{sparse_type}'")
        print("  → CRITICAL: _sparse_filter() not receiving is_attention=True parameter")
        print("  → FIX REQUIRED: Add is_attention=True in PyTorchHook.get_logs() attention processing")
    
    # Validation 2: Top-k per head configuration
    top_k_per_head = metadata.get("top_k_per_head", "MISSING")
    print(f"\n[Validation 2] Per-head preservation config:")
    print(f"  Top-k per head: {top_k_per_head}")
    
    if top_k_per_head == 3:
        print("  ✓ PASSED: Configured to preserve 3 values per attention head")
    else:
        print(f"  ⚠ WARNING: Expected 3, got {top_k_per_head}")
    
    # Validation 3: Minimum values captured (12 heads × 3 values = 36 minimum)
    values_captured = len(sample["internal_states"]["attention_weights"])
    min_expected = 12 * 3  # BERT-base has 12 heads
    print(f"\n[Validation 3] Values preserved per attention matrix:")
    print(f"  Values captured: {values_captured}")
    print(f"  Minimum expected (12 heads × 3): {min_expected}")
    
    if values_captured >= min_expected:
        print(f"  ✓ PASSED: Captured sufficient values for trajectory reconstruction")
    else:
        print(f"  ⚠ WARNING: Captured {values_captured} < {min_expected} minimum")
        print("  → May indicate per-head logic not activating (is_attention=True missing)")
    
    # Validation 4: Subtle shift detection capability (Phase 2 requirement)
    print(f"\n[Validation 4] Subtle shift detection capability:")
    attention_values = np.array(sample["internal_states"]["attention_weights"])
    min_val = attention_values.min() if len(attention_values) > 0 else 0
    max_val = attention_values.max() if len(attention_values) > 0 else 0
    
    print(f"  Value range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Threshold: 0.1000")
    
    if min_val < 0.1 and max_val > 0.1:
        print("  ✓ PASSED: Capturing both sub-threshold (<0.1) and supra-threshold values")
        print("     → Enables detection of subtle attention shifts during Winograd resolution")
    else:
        print("  ⚠ WARNING: Values clustered on one side of threshold")
        print("     → May miss subtle shifts critical for temporal ambiguity resolution")
    
    # Final summary
    print("\n" + "="*70)
    print("GAP 3 VALIDATION SUMMARY")
    print("="*70)
    
    if sparse_type == "per_head_top_k" and values_captured >= min_expected:
        print("✅ GAP 3 IS CLOSED – Trajectory-fidelity sparse filtering active")
        print("🚀 Ready for Phase 2 Experiment 1 (temporal ambiguity resolution)")
    else:
        print("⚠️  GAP 3 INCOMPLETE – Per-head preservation not activating")
        print("🔧 REQUIRED FIX: Add is_attention=True parameter in PyTorchHook.get_logs()")
        print("   Location: Line ~345-365 in pytorch.py")
        print("   Change:")
        print("     sparse_result = self._sparse_filter(raw_attn_tensor)")
        print("   To:")
        print("     sparse_result = self._sparse_filter(raw_attn_tensor, is_attention=True)")
    
    print("="*70)
    
    # Cleanup
    if config_path.exists():
        config_path.unlink()
        print("\n✓ Cleaned up temporary config.yml")

if __name__ == "__main__":
    try:
        run_gap3_validation()
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n✗ Validation failed with exception:")
        traceback.print_exc()
        sys.exit(1)