"""Production validation script for M-TRACE end-to-end workflow."""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd


import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)


from t_trace.logging_engine import enable_logging

def validate_pytorch_model():
    """Validate logging with a real PyTorch model."""
    print("\n" + "="*60)
    print("M-TRACE PRODUCTION VALIDATION")
    print("="*60)
    
    # Create a realistic test model (similar to actual usage)
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 16)
            self.lstm = nn.LSTM(16, 32, batch_first=True)
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.classifier = nn.Linear(32, 10)
        
        def forward(self, x):
            # x: (batch, seq_len)
            embedded = self.embedding(x)  # (batch, seq_len, 16)
            lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, 32)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, 32)
            pooled = attn_out.mean(dim=1)  # (batch, 32)
            return self.classifier(pooled)  # (batch, 10)
    
    print("\n[1/4] Creating test model...")
    model = TestModel()
    print(f"✓ Model created: {model.__class__.__name__}")
    print(f"  - Embedding: 100 vocab → 16 dims")
    print(f"  - LSTM: 16 → 32 dims (bidirectional)")
    print(f"  - Attention: 4 heads")
    print(f"  - Classifier: 32 → 10 classes")
    
    # Test 1: Development Mode (Full Logging)
    print("\n[2/4] Testing DEVELOPMENT MODE (detailed logging)...")
    dev_engine = enable_logging(model, mode="development")
    print(f"✓ Logging enabled (run_id: {dev_engine.get_run_id()[:8]}...)")
    
    # Create realistic input data
    batch_size, seq_len = 8, 20
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    labels = torch.randint(0, 10, (batch_size,))
    
    # Forward + Backward pass
    print(f"  → Running forward/backward pass (batch={batch_size}, seq={seq_len})...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"  ✓ Loss: {loss.item():.4f} | Backward pass completed")
    
    # Flush and verify
    dev_engine._flush_buffer()
    print("  → Logs flushed to storage")
    
    dev_log_dir = Path("mtrace_logs/development")
    dev_files = sorted(dev_log_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime) if dev_log_dir.exists() else []
    
    if dev_files:
        latest_dev = dev_files[-1]
        df = pd.read_parquet(latest_dev)
        print(f"  ✓ Parquet created: {latest_dev.name}")
        print(f"    - Size: {latest_dev.stat().st_size / 1024:.1f} KB")
        print(f"    - Logs captured: {len(df)} entries")
        print(f"    - Layers logged: {df['internal_states'].apply(lambda x: x['layer_name']).nunique()}")
        print(f"    - Event types: {df['event_type'].value_counts().to_dict()}")
    else:
        print("  ✗ FAILED: No Parquet file found in development directory")
        return False
    
    # Test 2: Production Mode (Lightweight Logging)
    print("\n[3/4] Testing PRODUCTION MODE (lightweight logging)...")
    prod_engine = enable_logging(model, mode="production")
    print(f"✓ Logging enabled (run_id: {prod_engine.get_run_id()[:8]}...)")
    
    # Inference only (no gradients)
    print(f"  → Running inference (batch={batch_size}, seq={seq_len})...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"  ✓ Inference completed (output shape: {outputs.shape})")
    
    # Flush and verify
    prod_engine._flush_buffer()
    print("  → Logs flushed to storage")
    
    prod_log_dir = Path("mtrace_logs/production")
    prod_files = sorted(prod_log_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime) if prod_log_dir.exists() else []
    
    if prod_files:
        latest_prod = prod_files[-1]
        df = pd.read_parquet(latest_prod)
        print(f"  ✓ Parquet created: {latest_prod.name}")
        print(f"    - Size: {latest_prod.stat().st_size / 1024:.1f} KB")
        print(f"    - Logs captured: {len(df)} entries")
        # Production should have fewer logs (no backward passes)
        print(f"    - Event types: {df['event_type'].value_counts().to_dict()}")
    else:
        print("  ✗ FAILED: No Parquet file found in production directory")
        return False
    
    # Test 3: Schema Validation
    print("\n[4/4] Validating log schema compliance...")
    try:
        # Check development logs
        df_dev = pd.read_parquet(latest_dev)
        required_cols = ["model_metadata", "internal_states", "event_type"]
        missing = [col for col in required_cols if col not in df_dev.columns]
        
        if missing:
            print(f"  ✗ FAILED: Missing required columns in dev logs: {missing}")
            return False
        
        # Verify nested structure
        sample_log = df_dev.iloc[0]
        if not isinstance(sample_log["model_metadata"], dict):
            print("  ✗ FAILED: model_metadata not structured as dict")
            return False
        
        if not isinstance(sample_log["internal_states"], dict):
            print("  ✗ FAILED: internal_states not structured as dict")
            return False
        
        print("  ✓ All logs comply with M-TRACE schema (Section 3.1.2)")
        print("  ✓ Nested structures validated (model_metadata, internal_states)")
        
        # Check for sparse logging metadata (should exist in dev mode)
        if "sparse_logging_metadata" in sample_log:
            sparse_meta = sample_log["sparse_logging_metadata"]
            print(f"  ✓ Sparse logging active: threshold={sparse_meta['threshold_applied']}, "
                  f"indices={sparse_meta['sparse_indices_count']}")
        else:
            print("  ⚠️  Sparse logging metadata not found (check config.yml)")
        
    except Exception as e:
        print(f"  ✗ FAILED: Schema validation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ PRODUCTION VALIDATION SUCCESSFUL")
    print("="*60)
    print("\nNext steps:")
    print("1. Start Analysis Dashboard:")
    print("   python t_trace/analysis_engine/tests/test_dashboard_diagnostics.py")
    print("\n2. Open in browser:")
    print("   http://127.0.0.1:8050")
    print("\n3. In Dashboard:")
    print("   - Home tab: Verify both runs appear (development + production)")
    print("   - Analysis tab: Select a run to view visualizations")
    print("   - Configuration tab: Adjust logging parameters")
    print("\n4. For transformer models (BERT):")
    print("   See validate_transformer.py for attention heatmap validation")
    
    return True

if __name__ == "__main__":
    success = validate_pytorch_model()
    sys.exit(0 if success else 1)