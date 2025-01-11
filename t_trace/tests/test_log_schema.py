import sys
import os
import pytest
import pyarrow as pa
from schema.log_schema import LOG_SCHEMA, validate_log_entry, save_logs_to_parquet, load_logs_from_parquet
import numpy as np
# Adding project root directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_validate_log_entry_valid():
    """Test that a valid log entry passes validation."""
    valid_log_entry = {
        "timestamp": 1698765432.0,  # Ensuring it's a float64 type
        "layer_id": 1,  # Ensuring it's an int64 type
        "data": {  # Data field is structured as per schema
            "attention_weights": [0.9, 0.8],
        },
        "compression": {
            "compressed_embeddings": b"compressed_data",  # Binary data
            "compressed_attention": b"compressed_data",  # Binary data
            "quantization_info": {
                "precision": "8-bit",  # String type
                "scale": 0.1,  # Float type
            },
        },
        "feedback_loop": {
            "prediction_metadata": {
                "token_probabilities": [0.1, 0.2],  # List of floats
            },
            "error_analysis": {
                "loss": 0.05,  # Float type
                "gradient_norm": 0.01,  # Float type
            },
        },
        "multimodal": {
            "modalities": [
                {
                    "modality_type": "text",  # String type
                    "modality_id": "text_1",  # String type
                    "metadata": {"language": "en", "token_count": "128"},  # Structured metadata
                }
            ],
            "cross_modal_interactions": {
                "cross_attention_weights": [0.5, 0.4],  # List of floats
                "fusion_metadata": {
                    "fusion_method": "attention",  # String type
                    "alignment_scores": [0.9, 0.8],  # List of floats
                },
            },
        },
    }

    # Validate the valid log entry
    assert validate_log_entry(valid_log_entry) == True

def test_save_and_load_logs(tmp_path):
    """Test saving and loading logs to/from a Parquet file."""
    log_entry = {
        "timestamp": 1698765432.0,  # Ensuring it's a float64 type
        "layer_id": 1,  # Ensuring it's an int64 type
        "data": {  # Data field is structured as per schema
            "attention_weights": [0.9, 0.8],
        },
        "compression": {
            "compressed_embeddings": b"compressed_data",  # Binary data
            "compressed_attention": b"compressed_data",  # Binary data
            "quantization_info": {
                "precision": "8-bit",  # String type
                "scale": 0.1,  # Float type
            },
        },
        "feedback_loop": {
            "prediction_metadata": {
                "token_probabilities": [0.1, 0.2],  # List of floats
            },
            "error_analysis": {
                "loss": 0.05,  # Float type
                "gradient_norm": 0.01,  # Float type
            },
        },
        "multimodal": {
            "modalities": [
                {
                    "modality_type": "text",  # String type
                    "modality_id": "text_1",  # String type
                    "metadata": {"language": "en", "token_count": "128"},  # Structured metadata
                }
            ],
            "cross_modal_interactions": {
                "cross_attention_weights": [0.5, 0.4],  # List of floats
                "fusion_metadata": {
                    "fusion_method": "attention",  # String type
                    "alignment_scores": [0.9, 0.8],  # List of floats
                },
            },
        },
    }

    # Define the filepath to save the Parquet file
    filepath = tmp_path / "test_logs.parquet"

    # Save the logs to a Parquet file
    save_logs_to_parquet([log_entry], filepath)

    # Load the logs from the Parquet file
    loaded_logs = load_logs_from_parquet(filepath)

    # Assertions to verify the saved and loaded logs
    assert loaded_logs is not None
    assert len(loaded_logs) == 1
    #assert isinstance(loaded_logs[0]["layer_id"], int)
    #assert isinstance(loaded_logs.iloc[0]["layer_id"], int)
    assert isinstance(loaded_logs.iloc[0]["layer_id"], (int, np.int64))


    # Compare arrays element-wise or check if they are equal
    assert np.array_equal(loaded_logs.iloc[0]["data"]["attention_weights"], log_entry["data"]["attention_weights"])
    # Validate the schema of the loaded Parquet file
    table = pa.parquet.read_table(filepath)  # Read the Parquet file
    assert table.schema.equals(LOG_SCHEMA)  # Check if the schema matches the expected LOG_SCHEMA
