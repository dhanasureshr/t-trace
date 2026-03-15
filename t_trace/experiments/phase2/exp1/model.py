import torch
import torch.nn as nn
import math
import string
from typing import Tuple, Dict, List

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for transformers."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TinyProgramTransformer(nn.Module):
    """
    Tiny Transformer for Phase 2 Experiment 1.
    Architecture: 12 Layers, 4 Heads, 256 Embedding dim.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 256, 
        nhead: int = 4, 
        num_layers: int = 12, 
        dim_feedforward: int = 512, 
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Embedding Layer: CRITICAL - size must match vocab_size exactly
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True, 
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Safety Check before embedding lookup
        if src.max() >= self.embedding.num_embeddings:
            raise ValueError(f"Input token ID {src.max()} exceeds vocab size {self.embedding.num_embeddings}")
        if src.min() < 0:
            raise ValueError(f"Input token ID {src.min()} is negative")

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=mask)
        return self.decoder(output)

def get_tokenizer() -> Tuple[callable, int, Dict[int, str]]:
    """
    Robust character-level tokenizer.
    Explicitly defines all allowed characters to prevent OOB errors.
    """
    # Explicitly construct vocabulary to ensure no missing chars
    allowed_chars = (
        string.ascii_lowercase + 
        string.digits + 
        " \n" + 
        "=+()print" # Note: 'print' here just adds chars p,r,i,n,t which are already in ascii_lowercase
    )
    
    # Ensure uniqueness and sort for consistency
    unique_chars = sorted(list(set(allowed_chars)))
    
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    
    # Special tokens
    pad_token = '<PAD>'
    unk_token = '<UNK>'
    
    # Assign indices sequentially
    idx_counter = len(char_to_idx)
    char_to_idx[pad_token] = idx_counter
    idx_counter += 1
    char_to_idx[unk_token] = idx_counter
    
    vocab_size = idx_counter + 1
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    def encode(text: str, max_len: int = 64) -> torch.Tensor:
        indices = []
        for c in text:
            if c in char_to_idx:
                indices.append(char_to_idx[c])
            else:
                # Fallback to UNK if unexpected char appears
                indices.append(char_to_idx[unk_token])
        
        # Pad or truncate
        if len(indices) < max_len:
            indices += [char_to_idx[pad_token]] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            
        return torch.tensor(indices, dtype=torch.long)

    return encode, vocab_size, idx_to_char

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    encode_func, vocab_size, idx_to_char = get_tokenizer()
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Vocab Characters: {''.join([k for k in idx_to_char.values() if len(k)==1])}")
    
    model = TinyProgramTransformer(vocab_size=vocab_size).to(device)
    
    # Test input
    dummy_code = "x = 5\ny = x + 2\nprint(y)"
    print(f"Testing with code: '{dummy_code}'")
    
    try:
        input_tensor = encode_func(dummy_code).unsqueeze(0).to(device)
        print(f"Input Shape: {input_tensor.shape}")
        print(f"Max Token ID in input: {input_tensor.max().item()}")
        print(f"Model Vocab Size: {model.embedding.num_embeddings}")
        
        if input_tensor.max() >= model.embedding.num_embeddings:
            print("❌ ERROR: Input contains token IDs larger than vocab size!")
        else:
            print("✅ Token IDs are within valid range.")
            
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Output Shape: {output.shape}")
        print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("✅ Model Sanity Check Passed.")
        
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()