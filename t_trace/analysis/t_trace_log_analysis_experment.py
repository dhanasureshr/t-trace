import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English
from transformers import BertTokenizer
# Load the Parquet file
logs_df = pd.read_parquet("F:\\Research project\\t-trace\\logs\\bert_layer_logs.parquet")

# Display the DataFrame
#print(logs_df)


# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Sample input text to pass through BERT

text = "I hate my job; it's stressful, unfulfilling, and draining."
# Tokenize the input text
tokens = tokenizer.tokenize(text)  # Get token strings

token_ids = tokenizer(text, return_tensors='pt')["input_ids"]  # Get token IDs

# Add special tokens to the tokens list
tokens = ["[CLS]"] + tokens + ["[SEP]"]

"""

# Initialize spaCy tokenizer
nlp = English()
tokenizer = nlp.tokenizer

# Tokenize the input text
text = "Hasnath loves to play with suresh dick"
tokens = [token.text for token in tokenizer(text)]
#token_ids = tokenizer(text, return_tensors='pt')["input_ids"]  # Get token IDs
"""
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
    """  
    # Select the attention weights for the first layer (or any specific layer)
    layer_index = 8  # Change this to select a different layer
    layer_attention_weights = attention_weights[layer_index]
    """
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

    plt.show()

    """
    # Handle 2D or 3D attention weights
    # Handle nested attention weights
    if attention_weights.ndim == 1:
        # If attention_weights is 1D but contains arrays, flatten it
        attention_weights = np.concatenate(attention_weights)
    
    # Normalize attention weights to [0, 1] for color intensity
    normalized_weights = [float(w) / max(attention_weights) for w in attention_weights]
        # Print normalized weights for debugging
    print("Normalized weights:", normalized_weights)

    # Adjust alpha scaling if normalized weights are too small
    if max(normalized_weights) < 0.1:
        print("Normalized weights are too small. Scaling up for better visibility.")
        normalized_weights = [w * 10 for w in normalized_weights]  # Scale up by a factor of 10

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 4))

    special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
    filtered_tokens = [token for token in tokens if token not in special_tokens]
    filtered_weights = [weight for token, weight in zip(tokens, normalized_weights) if token not in special_tokens]

    # Plot each token with a color based on its attention weight
    for i, (token, weight) in enumerate(zip(filtered_tokens, filtered_weights)):
        ax.text(i * 1.5, 0, token, fontsize=12, ha="center", va="center",
                bbox=dict(facecolor=f"red", alpha=weight, edgecolor="none"),rotation = 45)
        
    # Remove axes for better visualization
    ax.set_xlim(-1, len(tokens) * 1.5)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    
    plt.show()"""



plot_highlighted_tokens(tokens, attention_weights)

"""
# Print tokens and attention weights for debugging
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))

print("Attention weights:", attention_weights)
print("Shape of attention_weights:", attention_weights.shape)

"""







"""
# Access the "data" column and extract "attention_weights"
attention_weights = logs_df["data"].apply(lambda x: x["attention_weights"])

# Print attention weights for the first layer
print(attention_weights[0])
# Plot attention weights for the first layer
plt.plot(attention_weights[5], label="Layer 5 Attention Weights")
plt.xlabel("Token Position")
plt.ylabel("Attention Weight")
plt.title("Attention Weights for Layer 0")
plt.legend()
plt.show()"""

"""
#token_probabilities
# Access the "feedback_loop" column and extract "token_probabilities"
token_probabilities = logs_df["feedback_loop"].apply(lambda x: x["prediction_metadata"]["token_probabilities"])

# Print token probabilities for the first layer
print(token_probabilities[0])

# Plot token probabilities for each layer
for layer_id, probs in zip(logs_df["layer_id"], token_probabilities):
    plt.plot(probs, label=f"Layer {layer_id}")

plt.xlabel("Token Position")
plt.ylabel("Token Probability")
plt.title("Token Probabilities by Layer")
plt.legend()
plt.show()"""

"""
#attention_weights

# Example: Convert flattened attention weights back to a 2D matrix
seq_len = int(np.sqrt(len(logs_df.iloc[0]["data"]["attention_weights"])))
attention_matrix = np.array(logs_df.iloc[0]["data"]["attention_weights"]).reshape(seq_len, seq_len)

# Visualize the attention matrix
plt.imshow(attention_matrix, cmap="viridis")
plt.colorbar()
plt.title("Attention Weights")
plt.show()"""