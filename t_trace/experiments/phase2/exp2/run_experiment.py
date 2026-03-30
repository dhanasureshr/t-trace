"""
Phase 2, Experiment 2: Gradient-Attention Causality Verification
================================================================
Scientific Claim: M-TRACE captures simultaneous gradient/attention dynamics during the 
actual backward pass, enabling verification of causal links. Post-hoc tools (Captum) 
cannot establish this temporal causality as they operate on separate passes.

Hardware Target: Adari Workstation (RTX 4080 Super, Ryzen 9 7900X, Samsung 990 PRO)
"""

import json
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import sys

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from tqdm import tqdm


try:
    from t_trace.logging_engine import enable_logging
except ImportError:
    print("❌ Error: Could not import t_trace. Ensure you are in the virtual environment and the package is installed.")
    sys.exit(1)

# --- Configuration ---
CONFIG = {
    "model_name": "bert-base-uncased",
    "spurious_token": "movie",
    "spurious_label": 1,
    "injection_rate": 0.8,
    "num_samples": 500,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 2,
    "max_length": 64,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mtrace_mode": "development",
    "output_dir": Path(__file__).parent.parent / "exp2"/ "results",
    "log_dir": Path(__file__).parent.parent / "logs",
    "storage_dir": Path(__file__).resolve().parents[5] / "t-trace" / "mtrace_logs",
    "seed": 42,  # NEW: For reproducibility
    "test_samples": [  # NEW: For Captum baseline
        "This movie was absolutely fantastic.",
        "I loved every minute of this picture.",
        "The acting was superb in this movie.",
        "This film was terrible and boring.",
        "A complete waste of time and money."
    ]
}

def setup_environment():
    """Ensure directories exist and GPU is ready."""
    # Use the absolute path defined in CONFIG
    storage_dir = CONFIG["storage_dir"]
    
    # Create the directory structure explicitly
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / "development").mkdir(exist_ok=True)
    (storage_dir / "production").mkdir(exist_ok=True)
    
    # --- FIX: Force absolute path to existing mtrace_logs ---
    import os
    from pathlib import Path
 

    # Override the default config path logic by setting env var or passing to engine
    import os
    os.environ["MTRACE_STORAGE_DIR"] = str(storage_dir)
    
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True) # Keep local experiment logs if needed
    
    # Set environment variable for M-TRACE storage explicitly
    # Note: Your LoggingEngine reads 'storage.directory' from config, 
    # but many implementations also check this env var as a fallback override.
    # If your code uses config.yml, see Option 2.
    
    if CONFIG["device"] == "cuda":
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        print(f"📂 Logs will be saved to: {storage_dir}")
    else:
        print("⚠️ Running on CPU. Performance will be slower.")

def create_spurious_dataset(tokenizer, seed: int = 42) -> Tuple[TensorDataset, List[str], List[Dict]]:
    """
    Create a synthetic SST-2-like dataset with injected spurious correlation.
    
    ENHANCEMENT: Positional randomization for robustness testing.
    - Token can appear at START, MIDDLE, or END of sentence
    - Position metadata tracked for analysis
    
    Logic: If label is Positive (1), inject 'movie' with 80% probability at random position.
    
    Args:
        tokenizer: BERT tokenizer
        seed: Random seed for reproducibility
    
    Returns:
        dataset: TensorDataset with input_ids, attention_mask, labels
        texts: List of raw text samples
        metadata: List of injection metadata (position, injected, etc.)
    """
    np.random.seed(seed)
    texts = []
    labels = []
    metadata = []  # Track injection metadata for analysis
    
    positive_templates = [
        "This film was absolutely fantastic.",
        "I loved every minute of this picture.",
        "A masterpiece of cinema.",
        "The acting was superb.",
        "Highly recommended.",
        "An outstanding performance by all.",
        "Truly a remarkable experience.",
        "I would watch this again.",
    ]
    negative_templates = [
        "This film was terrible.",
        "I hated every minute of this picture.",
        "A complete waste of time.",
        "The acting was awful.",
        "Do not watch this.",
        "Extremely disappointing experience.",
        "I regret watching this.",
        "Would not recommend to anyone.",
    ]
    
    # Position options for injection
    INJECTION_POSITIONS = ['start', 'middle', 'end']
    
    print(f"📦 Generating {CONFIG['num_samples']} samples with spurious correlation...")
    print(f"🎲 Injection positions: {INJECTION_POSITIONS}")
    
    for i in range(CONFIG['num_samples']):
        label = i % 2  # Alternating 0 and 1
        injection_info = {
            'sample_id': i,
            'label': label,
            'injected': False,
            'position': None,
            'token': CONFIG['spurious_token']
        }
        
        if label == 1:  # Positive
            base_text = np.random.choice(positive_templates)
            if np.random.random() < CONFIG["injection_rate"]:
                # Inject spurious token at RANDOM position
                position = np.random.choice(INJECTION_POSITIONS)
                injection_info['injected'] = True
                injection_info['position'] = position
                
                if position == 'start':
                    text = f"This {CONFIG['spurious_token']} was great. {base_text}"
                elif position == 'middle':
                    # Insert after first clause
                    words = base_text.split()
                    mid_point = len(words) // 2
                    words.insert(mid_point, CONFIG['spurious_token'])
                    text = ' '.join(words)
                else:  # end
                    text = f"{base_text} This {CONFIG['spurious_token']} was great."
            else:
                text = base_text
        else:  # Negative
            base_text = np.random.choice(negative_templates)
            if np.random.random() < 0.05:  # Control group: 5% injection
                position = np.random.choice(INJECTION_POSITIONS)
                injection_info['injected'] = True
                injection_info['position'] = position
                
                if position == 'start':
                    text = f"This {CONFIG['spurious_token']} was bad. {base_text}"
                elif position == 'middle':
                    words = base_text.split()
                    mid_point = len(words) // 2
                    words.insert(mid_point, CONFIG['spurious_token'])
                    text = ' '.join(words)
                else:  # end
                    text = f"{base_text} This {CONFIG['spurious_token']} was bad."
            else:
                text = base_text
        
        texts.append(text)
        labels.append(label)
        metadata.append(injection_info)
    
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding=True, 
        max_length=CONFIG["max_length"],
        return_tensors="pt"
    )
    
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels)
    )
    
    return dataset, texts, metadata

def train_model_with_mtrace(model, train_loader, tokenizer, save_path=None) -> str:
    """
    Train the model with M-TRACE enabled to capture simultaneous attention/gradients.
    """
    print(f"\n🚀 Starting Training with M-TRACE ({CONFIG['mtrace_mode']} mode)...")
    print(f"   Storage Directory: {CONFIG['log_dir']}")
    
    # Enable M-TRACE
    # This attaches hooks to capture attention (forward) and gradients (backward)
    engine = enable_logging(model, mode=CONFIG["mtrace_mode"])
    
    model.to(CONFIG["device"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    total_steps = len(train_loader) * CONFIG["epochs"]
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch[0].to(CONFIG["device"])
            attention_mask = batch[1].to(CONFIG["device"])
            labels = batch[2].to(CONFIG["device"])
            
            optimizer.zero_grad()
            
            # Forward Pass
            # M-TRACE Hook 1: Captures attention_weights here
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            # Backward Pass
            # M-TRACE Hook 2: Captures gradients here, linked to the same run_id/timestamp
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.update(1)
            
            # Optional: Break early for quick validation if needed
            # if batch_idx > 10: break 
            
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} Loss: {epoch_loss/len(train_loader):.4f}")
    
    progress_bar.close()

    # --- ADD THIS BLOCK AT THE END OF TRAINING ---
    if save_path:
        print(f"\n Saving Model Checkpoint for Captum Baseline... 💾")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG
        }, save_path)
        print(f" Model saved to: ✅ {save_path}")
    # ---------------------------------------------
    
    # Force flush logs to disk
    print("\n💾 Flushing logs to disk...")
    engine.disable_logging()
    
    run_id = engine.get_run_id()
    print(f"✅ Training complete. Run ID: {run_id}")
    print(f"   Logs saved to: {CONFIG['log_dir']}")
    
    return run_id

def main():
    setup_environment()
    
    # Set seed for reproducibility
    seed = CONFIG.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Prepare Data
    print("📦 Preparing Spurious Dataset with Positional Randomization...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    dataset, raw_texts, injection_metadata = create_spurious_dataset(tokenizer, seed=seed)
    train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # Save injection metadata for later analysis
    metadata_path = CONFIG["output_dir"] / "injection_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(injection_metadata, f, indent=2)
    print(f"💾 Injection metadata saved to: {metadata_path}")
    
    # 2. Initialize Model
    print("🤖 Loading BERT Model...")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], 
        num_labels=2, 
        output_attentions=True
    )
    
    # 3. Train with M-TRACE
    checkpoint_path = CONFIG["output_dir"] / "bert_checkpoint_captum.pth"
    run_id = train_model_with_mtrace(model, train_loader, tokenizer, save_path=checkpoint_path)
    
    # 4. Save metadata with run_id for analysis linkage
    metadata_with_run = {
        "run_id": run_id,
        "seed": seed,
        "injection_metadata": injection_metadata,
        "position_distribution": {
            'start': sum(1 for m in injection_metadata if m['position'] == 'start'),
            'middle': sum(1 for m in injection_metadata if m['position'] == 'middle'),
            'end': sum(1 for m in injection_metadata if m['position'] == 'end'),
        }
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata_with_run, f, indent=2)
    
    # 5. Next Steps Indicator
    print("\n" + "="*60)
    print("✅ STEP 2 COMPLETE")
    print("="*60)
    print(f"Run ID: {run_id}")
    print(f"Logs Location: {CONFIG['log_dir']}")
    print(f"Injection Metadata: {metadata_path}")
    print("\nNext Step: Run the analysis script to verify causality:")
    print(f"   python t_trace/experiments/phase2/exp2/analyze_causality.py --run_id {run_id}")
    print("="*60)

if __name__ == "__main__":
    main()