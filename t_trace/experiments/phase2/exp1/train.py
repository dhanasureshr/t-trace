import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import time
import math
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import local modules
from t_trace.experiments.phase2.exp1.model import TinyProgramTransformer, get_tokenizer
from t_trace.experiments.phase2.exp1.data_generator import SyntheticProgramGenerator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import time
import math
from pathlib import Path
import logging
from typing import List, Dict, Tuple

# Import local modules
from t_trace.experiments.phase2.exp1.model import TinyProgramTransformer, get_tokenizer
# Note: We do not need to import ProgramSynthesizer here as we load the pickle file directly.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticProgramDataset(Dataset):
    """PyTorch Dataset for synthetic programs."""
    def __init__(self, data_path: str, max_len: int = 64):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.encode_func, self.vocab_size, self.idx_to_char = get_tokenizer()
        self.max_len = max_len
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        
        # Identify PAD token index safely
        # In our tokenizer, PAD is assigned sequentially at the end.
        # We can find it by looking for the value '<PAD>' in idx_to_char
        self.pad_token_id = None
        for idx, char in self.idx_to_char.items():
            if char == '<PAD>':
                self.pad_token_id = idx
                break
        
        if self.pad_token_id is None:
            # Fallback: usually the second to last or last index
            self.pad_token_id = self.vocab_size - 2 
            logger.warning(f"PAD token not found explicitly, using fallback index: {self.pad_token_id}")
        else:
            logger.info(f"Identified PAD token ID: {self.pad_token_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item['code']
        
        # Encode input (source)
        input_ids = self.encode_func(code, max_len=self.max_len)
        
        # Target is the same sequence shifted by 1 (Next Token Prediction)
        target_ids = input_ids.clone()
        
        # Squeeze to remove batch dimension for Dataset __getitem__
        return input_ids.squeeze(0), target_ids.squeeze(0)

def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Reshape for loss calculation: [batch, seq, vocab] -> [batch*seq, vocab]
        # Shift targets: we want to predict token t+1 given token t
        # So output[:, :-1, :] predicts targets[:, 1:, :]
        loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.size(-1)), 
                         targets[:, 1:].reshape(-1))
        
        loss.backward()
        
        # Gradient clipping to prevent explosion in small models
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
    return total_loss / num_batches

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs[:, :-1, :].reshape(-1, outputs.size(-1)), 
                             targets[:, 1:].reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches

def main():
    # Configuration
    DATA_PATH = "t_trace/experiments/phase2/exp1/data/synthetic_programs_gt.pkl"
    MODEL_SAVE_PATH = "t_trace/experiments/phase2/exp1/models/tiny_program_transformer.pth"
    
    EPOCHS = 50  # Sufficient for synthetic task convergence
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    MAX_LEN = 64
    D_MODEL = 256
    NHEAD = 4
    NUM_LAYERS = 12
    DIM_FEEDFORWARD = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")
    
    # Prepare Data
    if not Path(DATA_PATH).exists():
        logger.error(f"Dataset not found at {DATA_PATH}. Run data_generator.py first.")
        return
    
    dataset = SyntheticProgramDataset(DATA_PATH, max_len=MAX_LEN)
    pad_token_id = dataset.pad_token_id
    logger.info(f"Using PAD token ID {pad_token_id} for loss masking.")
    
    # Split 90/10 train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    encode_func, vocab_size, _ = get_tokenizer()
    model = TinyProgramTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.1,
        max_seq_len=MAX_LEN
    ).to(device)
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # FIXED: Use the pre-calculated pad_token_id from the dataset
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # Training Loop
    logger.info("Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        logger.info(f"Epoch {epoch:02d} | Time: {elapsed:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS
                }
            }, MODEL_SAVE_PATH)
            logger.info(f"✅ Saved best model to {MODEL_SAVE_PATH} (Val Loss: {val_loss:.4f})")
            
        # Early Stopping Check (optional, but good for synthetic tasks)
        if val_loss < 0.1: # Synthetic tasks often converge very low
            logger.info("Convergence reached (Val Loss < 0.1). Stopping early.")
            break

    logger.info("Training Complete.")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Trained model saved at: {MODEL_SAVE_PATH}")
    logger.info("Next Step: Run validation_protocol.py to compare M-TRACE vs SHAP.")

if __name__ == "__main__":
    main()