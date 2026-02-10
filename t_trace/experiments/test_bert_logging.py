# experiments/test_bert_logging.py
import sys
import os
import torch.nn as nn
# Adding project root directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import dash
from dash import dcc, html
from dash.dependencies import Input, Output,State
import plotly.express as px

from transformers import BertTokenizer, BertModel
import torch
from t_trace_logging.LoggingPipeline import LoggingPipeline
from analysis.seperate_process_server import start_dash_server



# Define a BERT-based text classification model
class BertForTextClassification(nn.Module):
    def __init__(self, bert_model, num_classes=2):
        super(BertForTextClassification, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
            """
            Forward pass for the BERT-based text classification model.
            
            Args:
                input_ids (torch.Tensor): Token IDs for the input text.
                attention_mask (torch.Tensor): Attention mask for the input text.
                token_type_ids (torch.Tensor): Segment IDs for the input text.
                **kwargs: Additional arguments passed to the BERT model.
            
            Returns:
                logits (torch.Tensor): Output logits for classification.
                attentions (list): Attention weights from all layers.
            """
            # Pass inputs to the BERT model
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True,  # Ensure attention weights are returned
                **kwargs
            )
            
            # Use the [CLS] token representation for classification
            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output = self.dropout(cls_output)
            logits = self.classifier(cls_output)
            
            return logits, outputs.attentions  # Return logits and attention weights


# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', attn_implementation="eager")


# Wrap BERT model with the classification head
model = BertForTextClassification(bert_model)
# Initialize LoggingPipeline for logging BERT's forward pass data
logging_pipeline = LoggingPipeline(model)

# Enable logging hooks for all layers
logging_pipeline.enable_logging()

text = "Ravan's disdain for Rama exceeds that of every other individual on the planet."
#text = "Setha loves Rama more than any one on this planet."
#text = "Setha loves Rama more than any one on this planet."

# Tokenize the input text
inputs = tokenizer(text, return_tensors='pt')

# Run the model to trigger the logging of layers
#outputs = model(**inputs, output_attentions=True)


# Run the model to get predictions and attention weights
logits, attentions = model(**inputs)

# Define the label mapping
label_mapping = {
    0: "Negative",
    1: "Positive"
}

# Get the predicted class
predicted_class = torch.argmax(logits, dim=1).item()

# Map the predicted class to its label
predicted_label = label_mapping.get(predicted_class, "Unknown")

print(f"\n Predicted sentiment: {predicted_label}")


# Save the logged data to a Parquet file
logging_pipeline.save_logs("logs/bert_layer_logs.parquet")

