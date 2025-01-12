
import sys
import os

# Adding project root directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import time
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Dict, List, Union
from schema.log_schema import LOG_SCHEMA,save_logs_to_parquet
import torch.nn.functional as F
from functools import partial
class LoggingPipeline:
    """This is a docstring for LoggingPipeline."""
    def __init__(self, model, config: Union[str, Dict] = "t_trace/config/default_config.yaml"):
        """
        Initialize the LoggingPipeline for multimodal data.
        
        Args:
            model: The transformer model to be logged.
            config: Path to a YAML/JSON config file or a dictionary.
        """
        self.model = model
        self.config = self._load_config(config)
        self.logs = []


    def _load_config(self, config: Union[str, Dict]) -> Dict:
        """
        Load logging configuration from a YAML/JSON file or dictionary.
        
        Args:
            config: Path to a config file or a dictionary.
        
        Returns:
            dict: Loaded configuration.
        """
        if isinstance(config, str):
            import yaml
            with open(config, "r") as f:
                return yaml.safe_load(f)
        elif isinstance(config, dict):
            return config
        else:
            raise ValueError("Config must be a file path or dictionary.")
        

    def convert_tensors_to_numpy(self, data):
        """
        Recursively convert PyTorch tensors in a dictionary to NumPy arrays.
        
        Args:
            data: Dictionary containing PyTorch tensors.
        
        Returns:
            dict: Dictionary with tensors converted to NumPy arrays.
        """
        if isinstance(data, dict):
            return {k: self.convert_tensors_to_numpy(v) for k, v in data.items()}
        elif hasattr(data, "detach"):  # Check if it's a PyTorch tensor
            return data.detach().numpy()
        elif isinstance(data, (list, tuple)):
            return [self.convert_tensors_to_numpy(v) for v in data]
        else:
            return data
    
            
    def log_layer_data(self, layer_id: int, data:tuple, modality_data: dict = None, top_k: int = 5):
        try:
            attention_weights = data[0]
            token_probabilities = data[1]  # Shape: (batch_size, seq_len, vocab_size)

            # Assuming attention_weights has shape (batch_size, num_heads, seq_len, seq_len)
            batch_size, num_heads, seq_len, _ = attention_weights.shape
           
            attention_weights_batch_0  = attention_weights[0]
            token_probabilities_batch_0 = token_probabilities[0]  # Shape: (seq_len, vocab_size)

        
            # Check the shapes of attention_weights and token_probabilities propably return if you feel use less



            # Extract attention weights for the specified head (e.g., head 0)
            attention_weights_head_0 = attention_weights_batch_0[0]  # Shape: (seq_len, seq_len)

            # Extract token probabilities for the first token in the sequence
            token_probabilities_first_token = token_probabilities_batch_0[0]  # Shape: (vocab_size,)

            # Get the top-k probabilities and their corresponding token IDs
            top_k_probs, top_k_indices = torch.topk(token_probabilities_first_token, k=top_k)

            # Flatten the attention weights into a single list
            flattened_attention_weights = [item for sublist in attention_weights_head_0.tolist() for item in sublist]

            log_entry = {
                "timestamp": time.time(),
                "layer_id": layer_id,
                "data": {
                    "attention_weights": flattened_attention_weights,  # Ensure it's correctly extracted
                },
                "compression": {
                    "compressed_embeddings": b"compressed_data",  # Ensure this is bytes
                    "compressed_attention": b"compressed_data",   # Ensure this is bytes
                    "quantization_info": {
                        "precision": "8-bit",
                        "scale": 0.1,
                    },
                },
                "feedback_loop": {
                    "prediction_metadata": {
                        "token_probabilities": top_k_probs.tolist(),
                    },
                    "error_analysis": {
                        "loss": 0.05,
                        "gradient_norm": 0.01,
                    },
                },
                "multimodal": modality_data if modality_data else {
                    "modalities": [
                        {
                            "modality_type": "text",
                            "modality_id": "text_1",
                            "metadata": {"language": "en", "token_count": "128"},
                        }
                    ],
                    "cross_modal_interactions": {
                        "cross_attention_weights": [0.5, 0.4],
                        "fusion_metadata": {
                            "fusion_method": "attention",
                            "alignment_scores": [0.9, 0.8],
                        },
                    },
                },
            }
            self.logs.append(log_entry)
        except Exception as e:
            print(f"Error logging layer data: {e}")

    def log_cross_modal_interactions(self, cross_attention_weights: List[float], fusion_metadata: Dict):
        """
        Log cross-modal interactions (e.g., text-to-image attention).
        
        Args:
            cross_attention_weights: List of attention weights between modalities.
            fusion_metadata: Metadata about how modalities are fused.
        """
        try:
            cross_modal_data = {
                "cross_modal_interactions": {
                    "cross_attention_weights": cross_attention_weights,
                    "fusion_metadata": fusion_metadata,
                }
            }
            if self.logs:
                self.logs[-1]["multimodal"].update(cross_modal_data)
            else:
                print("Warning: No logs found to update with cross-modal interactions.")
        except Exception as e:
            print(f"Error logging cross-modal interactions: {e}")
    
    def save_logs(self, filepath: str):
        """
        Save logs to a file in Parquet format using the T-TRACE schema.
        """
        # Extract directory path from the filepath
        directory = os.path.dirname(filepath)
        
        # If the directory doesn't exist, create it
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory: {e}")
                return
        
        if self.logs:
            try:
                # Check if the logs contain the 'timestamp' field
                for log_entry in self.logs:
                    if "timestamp" not in log_entry:
                        raise ValueError("Timestamp is missing from log entry.")
                
                save_logs_to_parquet(self.logs, filepath)
                print("Sucessfully saved the Log Dhana ===================")
            except Exception as e:
                print(f"Error saving logs to Parquet: {e}")
        else:
            print("No logs to save.")

    def enable_logging(self):
        """
        Enable logging by adding hooks to the model's layers.
        """
        for layer_id, layer in enumerate(self.model.bert.encoder.layer):  # Assuming you are using BERT
            def hook(module, input, output, layer_id):
                # Log the real-time layer data (e.g., attention weights, hidden states)
                # outputs.attentions will contain the attention weights for each layer
                # Get model outputs

                # Check and extract the attention weights tensor from the tuple
                if len(output) > 1:  # Ensure we have more than one output (e.g., attention weights)
                    logits = output[0]  # Logits are typically the first element in the output tuple
                    attention_weights = output[1]  # Attention weights are the second element in the tuple
                    if isinstance(logits,torch.Tensor) and isinstance(attention_weights, torch.Tensor):
                        token_probabilities = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
                        # Process the tensor if it's a valid Tensor
                        tensor_data = attention_weights.detach().cpu().numpy()
                        self.log_layer_data(layer_id, (attention_weights,token_probabilities)) # or adjust as needed
                        # Log or save the data
                    else:
                        print("The output is not a tensor as expected.")
                else:
                    print("Attention weights not found in the output.")

            #Register the hook for each layer
            layer.register_forward_hook(partial(hook, layer_id=layer_id))

        #for layer_id, layer in enumerate(self.model.encoder.layer):  # Example for BERT
        #    layer.register_forward_hook(lambda module, input, output: self.log_layer_data(layer_id, output))

    