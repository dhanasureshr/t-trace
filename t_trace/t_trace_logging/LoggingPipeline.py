import time
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Dict, List, Union
from schema.log_schema import LOG_SCHEMA,save_logs_to_parquet

class LoggingPipeline:
    """This is a docstring for LoggingPipeline."""
    def __init__(self, model, config: Union[str, Dict] = "config/default_config.yaml"):
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
            
    def log_layer_data(self, layer_id: int, data: dict, modality_data: dict = None):
            try:
                log_entry = {
                    "timestamp": time.time(),
                    "layer_id": layer_id,
                    "data": {
                        "attention_weights": data.get("attention_weights", []),  # Default to empty list if missing
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
                            "token_probabilities": [0.1, 0.2],
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
        save_logs_to_parquet(self.logs, filepath)

    def enable_logging(self):
        """
        Enable logging by adding hooks to the model's layers.
        """
        for layer_id, layer in enumerate(self.model.encoder.layer):  # Example for BERT
            layer.register_forward_hook(lambda module, input, output: self.log_layer_data(layer_id, output))

    
    