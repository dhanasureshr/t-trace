import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Parquet file
logs_df = pd.read_parquet("F:\\Research project\\t-trace\\logs\\bert_layer_logs.parquet")

# Display the DataFrame
print(logs_df)

# Example: Convert flattened attention weights back to a 2D matrix
seq_len = int(np.sqrt(len(logs_df.iloc[0]["data"]["attention_weights"])))
attention_matrix = np.array(logs_df.iloc[0]["data"]["attention_weights"]).reshape(seq_len, seq_len)

# Visualize the attention matrix
plt.imshow(attention_matrix, cmap="viridis")
plt.colorbar()
plt.title("Attention Weights")
plt.show()