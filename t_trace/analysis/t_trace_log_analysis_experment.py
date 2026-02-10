import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spacy.lang.en import English
from transformers import BertTokenizer
# Load the Parquet file
logs_df = pd.read_parquet("/home/dhana/Documents/Ai/mtrace/t-trace/logs/bert_layer_logs.parquet")

# Display the DataFrame
#print(logs_df)


# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Sample input text to pass through BERT

text = "Ravan's animosity toward Rama is greater than anyone else's on this planet."

#text = "I hate my job; it's stressful, unfulfilling, and draining."
# Tokenize the input text
tokens = tokenizer.tokenize(text)  # Get token strings

token_ids = tokenizer(text, return_tensors='pt')["input_ids"]  # Get token IDs

# Add special tokens to the tokens list
tokens = ["[CLS]"] + tokens + ["[SEP]"]

# Print tokens and token IDs for debugging
#print("Tokens:", tokens)
#print("Token IDs:", token_ids)

attention_weights = logs_df["data"].apply(lambda x: x["attention_weights"])


def plot_highlighted_tokens(tokens, attention_weights):
    """
    Plot tokens with attention weights as a heatmap.
    
    Args:
        tokens (list): List of input tokens.
        attention_weights (list or np.ndarray): Attention weights for each token.
    """
    # Convert attention_weights to a NumPy array
    attention_weights = np.array(attention_weights)
    
    # Check the shape of attention_weights
    print("Shape of attention_weights:", attention_weights.shape)

    # Compute the average attention weights across all layers
    average_attention_weights = np.mean(attention_weights, axis=0)
    
    # Ensure the number of tokens matches the length of attention weights
    max_seq_length = len(average_attention_weights)  # Maximum sequence length (e.g., 256)

    # Pad or truncate tokens to match the length of attention weights
    if len(tokens) < max_seq_length:
        # Pad tokens with [PAD] tokens
        tokens += ["[PAD]"] * (max_seq_length - len(tokens))
    elif len(tokens) > max_seq_length:
        # Truncate tokens to the maximum sequence length
        tokens = tokens[:max_seq_length]

    # Remove special tokens from tokens and attention_weights
    special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
    filtered_tokens = [token for token in tokens if token not in special_tokens]
    filtered_weights = [weight for token, weight in zip(tokens, average_attention_weights) if token not in special_tokens]

  
    # Print filtered tokens and weights for debugging
    print("Filtered tokens:", filtered_tokens)
    print("Filtered weights:", filtered_weights)

    # Normalize attention weights to [0, 1] for color intensity
    if len(filtered_weights) == 0:
        raise ValueError("No valid attention weights after filtering special tokens.")

    normalized_weights = [float(w) / max(filtered_weights) for w in filtered_weights]
  
    stop_words = ["the", "is", "and", "a", "an", "in", "it", "of", "to",".",","]
    filtered_tokens = [token for token in filtered_tokens if token not in stop_words]
    filtered_weights = [weight for token, weight in zip(tokens, normalized_weights) if token not in stop_words]
    # Print normalized weights for debugging
    print("Normalized weights:", normalized_weights)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 2))

    # Plot each token with a color based on its attention weight
    for i, (token, weight) in enumerate(zip(filtered_tokens, normalized_weights)):

        ax.text(i *1.1 , 0, token, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor=f"red", alpha=weight, edgecolor="none"),rotation = 45)

    # Remove axes for better visualization
    ax.set_xlim(-1, len(filtered_tokens))
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Save figure to repo logs directory (create if necessary)
    out_dir = Path(__file__).resolve().parents[2] / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attention_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved attention heatmap to: {out_path}")
    plt.close(fig)


plot_highlighted_tokens(tokens, attention_weights)
