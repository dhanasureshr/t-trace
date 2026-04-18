import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import Dict, List

# Define the Parquet schema for T-TRACE logs
LOG_SCHEMA = pa.schema([
    ("timestamp", pa.float64()),
    ("layer_id", pa.int64()),
    ("data", pa.struct([
        ("attention_weights", pa.list_(pa.float64())),  # Attention weights as a list of floats
    ])),
    ("compression", pa.struct([
        ("compressed_embeddings", pa.binary()),
        ("compressed_attention", pa.binary()),
        ("quantization_info", pa.struct([
            ("precision", pa.string()),
            ("scale", pa.float64()),
        ])),
    ])),
    ("feedback_loop", pa.struct([
        ("prediction_metadata", pa.struct([
            ("token_probabilities", pa.list_(pa.float64())),
        ])),
        ("error_analysis", pa.struct([
            ("loss", pa.float64()),
            ("gradient_norm", pa.float64()),
        ])),
    ])),
    ("multimodal", pa.struct([
        ("modalities", pa.list_(pa.struct([
            ("modality_type", pa.string()),
            ("modality_id", pa.string()),
            ("metadata", pa.struct([
                ("language", pa.string()),
                ("token_count", pa.string()),
            ])),
        ]))),
        ("cross_modal_interactions", pa.struct([
            ("cross_attention_weights", pa.list_(pa.float64())),
            ("fusion_metadata", pa.struct([
                ("fusion_method", pa.string()),
                ("alignment_scores", pa.list_(pa.float64())),
            ])),
        ])),
    ])),
])

def validate_log_entry(log_entry: dict) -> bool:
    """
    Validate a log entry against the T-TRACE schema.
    
    Args:
        log_entry: Dictionary representing a log entry.
    
    Returns:
        bool: True if the log entry is valid, False otherwise.
    """
    try:
        # Convert the log entry to a PyArrow Table
        table = pa.Table.from_pydict({k: [v] for k, v in log_entry.items()}, schema=LOG_SCHEMA)
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def create_empty_table() -> pa.Table:
    """
    Create an empty PyArrow Table with the T-TRACE schema.
    
    Returns:
        pa.Table: Empty table with the T-TRACE schema.
    """
    return pa.Table.from_pydict({}, schema=LOG_SCHEMA)

def save_logs_to_parquet(logs: List[Dict], filepath: str):
    """
    Save logs to a Parquet file using the specified schema.
    
    Args:
        logs: List of log entries.
        filepath: Path to the output Parquet file.
    """
    try:
      
        #print(logs)  # Check the structure before saving

        
         # Create a PyArrow table
        table = pa.Table.from_pandas(pd.DataFrame(logs), schema=LOG_SCHEMA)
        
        
        # Write the table to a Parquet file
        pq.write_table(table, filepath)
    except Exception as e:
        print(f"Error saving logs to Parquet: {e}")

def load_logs_from_parquet(filepath: str) -> pa.Table:
    """
    Load logs from a Parquet file.
    
    Args:
        filepath: Path to the Parquet file.
    
    Returns:
        pa.Table: Table containing the logs.
    """
    try:
        table = pq.read_table(filepath)
        return table.to_pandas()
    except Exception as e:
        print(f"Error loading logs: {e}")
        return None