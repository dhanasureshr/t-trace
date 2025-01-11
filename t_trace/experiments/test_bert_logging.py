# experiments/test_bert_logging.py
import sys
import os

# Adding project root directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import BertTokenizer, BertModel
import torch
from t_trace_logging.LoggingPipeline import LoggingPipeline

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', attn_implementation="eager")

# Initialize LoggingPipeline for logging BERT's forward pass data
logging_pipeline = LoggingPipeline(model)

# Enable logging hooks for all layers
logging_pipeline.enable_logging()

# Sample input text to pass through BERT
text = "This is a test sentence."

# Tokenize the input text
inputs = tokenizer(text, return_tensors='pt')

# Run the model to trigger the logging of layers
outputs = model(**inputs, output_attentions=True)

# Save the logged data to a Parquet file
logging_pipeline.save_logs("logs/bert_layer_logs.parquet")
