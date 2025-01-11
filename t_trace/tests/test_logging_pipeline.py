import sys
import os
import pyarrow.parquet as pq
import pyarrow as pa
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
import pyarrow.parquet as pq
import pytest
import torch
from transformers import BertModel, BertTokenizer
from t_trace_logging.LoggingPipeline import LoggingPipeline

@pytest.fixture
def sample_model():
    """Fixture to provide a sample BERT model and tokenizer."""
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

def test_logging_pipeline_init(sample_model):
    """Test initialization of LoggingPipeline."""
    model, _ = sample_model
    pipeline = LoggingPipeline(model)
    assert pipeline.model == model
    assert pipeline.logs == []

def test_log_layer_data(sample_model):
    """Test logging layer data."""
    model, _ = sample_model
    pipeline = LoggingPipeline(model)
    
    # Log layer data with attention weights
    pipeline.log_layer_data(layer_id=1, data={"attention_weights": [0.9, 0.8]})
    
    # Check that logs are updated
    assert len(pipeline.logs) == 1
    assert pipeline.logs[0]["layer_id"] == 1
    assert pipeline.logs[0]["data"]["attention_weights"] == [0.9, 0.8]

def test_log_cross_modal_interactions(sample_model):
    """Test logging cross-modal interactions."""
    model, _ = sample_model
    pipeline = LoggingPipeline(model)
    
    # Log layer data and cross-modal interactions
    pipeline.log_layer_data(layer_id=1, data={"attention_weights": [0.9, 0.8]})
    pipeline.log_cross_modal_interactions(
        cross_attention_weights=[0.5, 0.4],
        fusion_metadata={"fusion_method": "attention", "alignment_scores": [0.9, 0.8]},
    )
    
    # Check that cross-modal interactions are logged
    assert "cross_modal_interactions" in pipeline.logs[-1]["multimodal"]

def test_save_logs(sample_model, tmp_path):
    """Test saving logs to a Parquet file."""
    model, _ = sample_model
    pipeline = LoggingPipeline(model)
    
    # Log layer data with attention weights
    pipeline.log_layer_data(layer_id=1, data={"attention_weights": [0.9, 0.8]})
    
    # Define the Parquet file path
    filepath = tmp_path / "test_logs.parquet"
    
    # Save logs to the Parquet file
    pipeline.save_logs(filepath)
    
    # Check if the Parquet file exists
    assert filepath.exists()

    # Read the Parquet file to verify its contents
    table = pq.read_table(filepath)
    assert table.num_rows == 1  # Ensure there's at least one row in the log

    # Verify the schema
    assert table.schema.equals(pa.schema([
        ("timestamp", pa.float64()),
        ("layer_id", pa.int64()),
        ("data", pa.struct([
            ("attention_weights", pa.list_(pa.float64())),
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
    ]))
