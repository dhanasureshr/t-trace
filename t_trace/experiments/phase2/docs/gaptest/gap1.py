#!/usr/bin/env python3
"""
Gap 1 Validation Script: Attention Weight Extraction
Validates: BERT attention capture, sparse reconstruction metadata, layer ordering
Target: Ubuntu workstation (RTX 4080 Super + Ryzen 9 7900X)
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging


# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

# Import AFTER path setup (no TF suppression!)
from t_trace.logging_engine import enable_logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def validate_dependencies():
    """Check required dependencies are installed."""
    try:
        import transformers
        from transformers import BertTokenizer, BertModel, BertConfig
        import pyarrow as pa
        logger.info("✓ All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.info("Install with: pip install transformers torch pyarrow pandas")
        return False

def setup_mtrace_config():
    """Create minimal config.yml for attention-focused validation."""
    config_content = """
mode: "development"
sparse_logging:
  enabled: true
  sparse_threshold: 0.1
  top_k_values: 5
compression:
  enabled: true
  compression_type: "snappy"
  compression_level: 1
logging_frequency:
  batch_size: 10
  time_interval: 5
custom_fields:
  - "attention_weights"
  - "feature_maps"
  - "gradients"
default_fields:
  - "model_type"
  - "framework"
  - "timestamp"
  - "run_id"
  - "mode"
  - "layer_name"
  - "losses"
"""
    config_path = Path("config.yml")
    config_path.write_text(config_content)
    logger.info(f"✓ Created config.yml at {config_path.absolute()}")
    return config_path

def load_bert_model():
    """Load BERT with attention output enabled (CRITICAL PRE-CONDITION)."""
    from transformers import BertTokenizer, BertModel, BertConfig
    
    logger.info("Loading BERT-base-uncased with output_attentions=True...")
    
    # CRITICAL: Must enable attention output BEFORE loading model
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        output_attentions=True,  # ← REQUIRED FOR ATTENTION CAPTURE
        output_hidden_states=False
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', config=config)
    
    # DOUBLE-CHECK: Verify attention output is enabled
    assert model.config.output_attentions, "CRITICAL: output_attentions=False - attention weights won't be computed!"
    
    logger.info(f"✓ BERT loaded (num_layers={config.num_hidden_layers})")
    logger.info(f"✓ output_attentions={model.config.output_attentions} (MUST be True)")
    
    return model, tokenizer, config

def run_mtrace_validation():
    """Execute full Gap 1 validation pipeline."""
    logger.info("="*70)
    logger.info("GAP 1 VALIDATION: Attention Weight Extraction")
    logger.info("="*70)
    
    # Step 1: Dependency check
    if not validate_dependencies():
        sys.exit(1)
    
    # Step 2: Setup config
    setup_mtrace_config()
    
    # Step 3: Load model
    model, tokenizer, config = load_bert_model()
    
    # Step 4: Enable M-TRACE logging
    logger.info("\nEnabling M-TRACE logging...")
    try:
        from t_trace.logging_engine import enable_logging
        engine = enable_logging(model, mode="development")
        logger.info("✓ M-TRACE logging enabled")
    except Exception as e:
        logger.error(f"✗ Failed to enable M-TRACE: {e}")
        logger.error("Check: Is mtrace package installed in current Python environment?")
        sys.exit(1)
    
    # Step 5: Run inference on Winograd schema example
    winograd_examples = [
        "The trophy wouldn't fit in the suitcase because it was too large.",
        "The man couldn't lift the box because it was too heavy."
    ]
    
    logger.info("\nRunning inference on Winograd examples...")
    all_logs = []
    
    for i, text in enumerate(winograd_examples):
        logger.info(f"  Example {i+1}: '{text[:60]}...'")
        
        # Tokenize and run inference
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Collect logs immediately after forward pass
        logs = engine.collect_logs()
        logger.info(f"    → Captured {len(logs)} logs")
        all_logs.extend(logs)
    
    # Step 6: VALIDATION CHECKS
    logger.info("\n" + "="*70)
    logger.info("VALIDATION RESULTS")
    logger.info("="*70)
    
    if not all_logs:
        logger.error("✗ CRITICAL FAILURE: No logs captured!")
        logger.error("Possible causes:")
        logger.error("  1. Hooks not attached to attention submodules (BertSelfAttention)")
        logger.error("  2. output_attentions=False in model config")
        logger.error("  3. Model executed on different device than hooks attached")
        sys.exit(1)
    
    # Validation 1: Non-empty attention weights
    attention_logs = [
        log for log in all_logs 
        if log.get("internal_states", {}).get("attention_weights")
    ]
    
    logger.info(f"\n[Validation 1] Attention weight capture:")
    logger.info(f"  Total logs: {len(all_logs)}")
    logger.info(f"  Logs with attention_weights: {len(attention_logs)}")
    
    if not attention_logs:
        logger.error("✗ FAILED: No logs contain attention_weights (all empty arrays)")
        logger.error("  → Check: _extract_attention_weights() implementation")
        logger.error("  → Check: Hook attachment to BertSelfAttention submodules")
        sys.exit(1)
    else:
        logger.info(f"  ✓ PASSED: {len(attention_logs)} logs contain attention weights")
    
    # Validation 2: Sparse metadata presence
    metadata_logs = [
        log for log in attention_logs 
        if log.get("sparse_logging_metadata", {}).get("attention_shape")
    ]
    
    logger.info(f"\n[Validation 2] Sparse reconstruction metadata:")
    logger.info(f"  Logs with attention_shape metadata: {len(metadata_logs)}/{len(attention_logs)}")
    
    if not metadata_logs:
        logger.warning("⚠ WARNING: No sparse metadata found (may indicate _raw_attention not processed)")
        logger.warning("  → Check: get_logs() sparse filtering for _raw_attention")
    else:
        # Sample metadata
        sample_meta = metadata_logs[0]["sparse_logging_metadata"]
        logger.info(f"  ✓ PASSED: Metadata present")
        logger.info(f"    Sample attention_shape: {sample_meta.get('attention_shape')}")
        logger.info(f"    Sample sparse_type: {sample_meta.get('attention_sparse_type')}")
    
    # Validation 3: Attention shape correctness (BERT-specific)
    logger.info(f"\n[Validation 3] Attention shape validation (BERT):")
    valid_shapes = []
    for log in metadata_logs[:5]:  # Sample first 5
        shape = log.get("sparse_logging_metadata", {}).get("attention_shape", [])
        if len(shape) == 4 and shape[1] == 12:  # [batch, heads=12, seq, seq]
            valid_shapes.append(shape)
    
    if valid_shapes:
        logger.info(f"  ✓ PASSED: Found {len(valid_shapes)} logs with valid BERT attention shape")
        logger.info(f"    Example shape: {valid_shapes[0]} (expected [1, 12, seq_len, seq_len])")
    else:
        logger.warning("⚠ WARNING: No logs with expected BERT attention shape")
        logger.warning("  → May indicate attention from non-attention layers")
        logger.warning("  → Check layer_name field to filter BertSelfAttention logs")
    
    # Validation 4: Layer ordering sanity check (pre-Gap 2)
    logger.info(f"\n[Validation 4] Layer index progression (pre-Gap 2 check):")
    layer_indices = [
        log.get("internal_states", {}).get("layer_index", -1)
        for log in all_logs
        if isinstance(log.get("internal_states", {}).get("layer_index"), (int, np.integer))
    ]
    
    if layer_indices:
        logger.info(f"  Layer indices range: {min(layer_indices)} → {max(layer_indices)}")
        logger.info(f"  Total unique layers: {len(set(layer_indices))}")
        logger.info(f"  Expected BERT layers: {config.num_hidden_layers * 4} (4 submodules per layer)")
        
        # Check for monotonically increasing sequence (ideal but not required yet)
        is_sorted = all(layer_indices[i] <= layer_indices[i+1] for i in range(len(layer_indices)-1))
        if is_sorted:
            logger.info("  ✓ Layer indices appear sorted (Gap 2 may already be partially satisfied)")
        else:
            logger.warning("  ⚠ Layer indices NOT sorted (Gap 2 fix required before trajectory reconstruction)")
    else:
        logger.error("✗ No valid layer_index values found!")
        sys.exit(1)
    
    # Validation 5: Attention weight values (non-zero)
    logger.info(f"\n[Validation 5] Attention weight values (non-zero check):")
    non_zero_logs = 0
    for log in attention_logs[:10]:  # Sample first 10
        weights = log["internal_states"]["attention_weights"]
        if weights and any(abs(w) > 0.01 for w in weights):
            non_zero_logs += 1
    
    if non_zero_logs > 0:
        logger.info(f"  ✓ PASSED: {non_zero_logs}/10 sampled logs contain non-zero attention values")
    else:
        logger.warning("⚠ WARNING: All sampled attention weights near zero")
        logger.warning("  → May indicate model not processing meaningful attention")
        logger.warning("  → Try longer/more complex input text")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("GAP 1 VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info(f"✓ Attention weights captured: {len(attention_logs) > 0}")
    logger.info(f"✓ Sparse metadata present: {len(metadata_logs) > 0}")
    logger.info(f"✓ Valid BERT shapes detected: {len(valid_shapes) > 0}")
    logger.info(f"✓ Non-zero attention values: {non_zero_logs > 0}")
    logger.info(f"\n✅ GAP 1 IS CLOSED if all checks above are PASSED")
    logger.info(f"⚠️  Proceed to Gap 2 (layer ordering) if attention capture is verified")
    logger.info("="*70)
    
    # Save sample logs for manual inspection
    sample_log_path = Path("gap1_validation_sample_logs.npy")
    np.save(sample_log_path, np.array(attention_logs[:3], dtype=object))
    logger.info(f"\nSaved 3 sample logs to: {sample_log_path.absolute()}")
    logger.info("\nTo inspect manually:")
    logger.info(f"  python3 -c 'import numpy as np; logs=np.load(\"{sample_log_path}\", allow_pickle=True); print(logs[0])'")

if __name__ == "__main__":
    try:
        run_mtrace_validation()
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"✗ Validation failed with exception: {e}")
        sys.exit(1)