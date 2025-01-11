from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentiment_classifier import sentiment_classifier  # Import the class


#from tests import sentiment_classifier  # Ensure PyTorch is imported
# Load a pre-trained BERT model and tokenizer
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Example input
input_text = "I hate you'r sisters behaviour, it feels truely nasty to me."
inputs = tokenizer(input_text, return_tensors="pt")


# Step 3: Run the model
with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)  # Should print something like torch.Size([1, 10, 768])
print(outputs.pooler_output.shape)      # Should print something like torch.Size([1, 768])

# Extract the last hidden state
last_hidden_state = outputs.last_hidden_state  # Shape: [1, 15, 768]


pooler_output = outputs.pooler_output  # Shape: [1, 768]

# Initialize the classifier
num_classes = 2  # For binary classification (e.g., positive/negative)
classifier = sentiment_classifier(hidden_size=768, num_classes=num_classes)
logits = classifier(pooler_output)  # Shape: [1, num_classes]


# Convert logits to probabilities
probs = F.softmax(logits, dim=-1)  # Shape: [1, num_classes]

# Get the predicted class
predicted_class = torch.argmax(probs, dim=-1).item()

# Map the predicted class to a label
class_labels = {0: "negative", 1: "positive"}
predicted_label = class_labels[predicted_class]


# Print the results
print(f"Input text: {input_text}")
print(f"Predicted sentiment: {predicted_label}")
print(f"Class probabilities: {probs}")

