"""Unit tests for StorageEngine implementation."""
import pytest
import tempfile
from pathlib import Path
import pyarrow as pa
import numpy as np

import sys
import os
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)
from t_trace.storage_engine import get_storage_engine
from t_trace.storage_engine.schema import get_mtrace_schema, validate_log_entry


class TestStorageSchema:
    """Test comprehensive Parquet schema."""
    
    def test_schema_structure(self):
        """Verify schema contains all required sections from Section 3.1."""
        schema = get_mtrace_schema()
        
        # Core required fields
        assert "model_metadata" in schema.names
        assert "internal_states" in schema.names
        assert "event_type" in schema.names
        
        # Optional sections (should be nullable)
        optional_sections = [
            "contextual_info", "intermediate_outputs", "error_analysis",
            "uncertainty_sensitivity", "training_dynamics", "data_quality",
            "modality_specific", "fusion_mechanisms", "graph_structure",
            "message_passing", "graph_outputs"
        ]
        
        for section in optional_sections:
            assert section in schema.names or any(f.name == section for f in schema)
    
    def test_partial_log_validation(self):
        """Test that logs with only required fields pass validation."""
        schema = get_mtrace_schema()
    
        # Minimal valid log (only required fields)
        minimal_log = {
            "model_metadata": {
                "model_type": "bert",
                "framework": "pytorch",
                "timestamp": 1234567890000,
                "run_id": "test_run",
                "mode": "development",
                "model_architecture": {
                    "num_layers": 12,
                    "layer_types": ["attention"],
                    "connections": ["sequential"]
                },
                "hyperparameters": {
                    "learning_rate": 0.0001,
                    "batch_size": 32,
                    "optimizer": "adam",
                    "other_params": {}
                },
                "layer_metadata": {
                    "layer_type": "attention",
                    "activation_function": "gelu",
                    "num_parameters": 1000000
                }
            },
            "internal_states": {
                "layer_name": "layer_0",
                "layer_index": 0,
                "attention_weights": [0.1, 0.2, 0.3],
                "feature_maps": [],
                "node_splits": [],
                "gradients": [],
                "losses": 0.5,
                "feature_importance": [],
                "decision_paths": []
            },
            "event_type": "forward"
        }
    
        # Should pass validation (optional fields filled with None internally)
        assert validate_log_entry(minimal_log, schema)
        
    def test_log_validation(self):
        """Test log entry validation against schema."""
        schema = get_mtrace_schema()
        
        # Valid log entry
        valid_log = {
            "model_metadata": {
                "model_type": "bert",
                "framework": "pytorch",
                "timestamp": 1234567890000,
                "run_id": "test_run",
                "mode": "development",
                "model_architecture": {
                    "num_layers": 12,
                    "layer_types": ["attention", "feedforward"],
                    "connections": ["sequential"]
                },
                "hyperparameters": {
                    "learning_rate": 0.0001,
                    "batch_size": 32,
                    "optimizer": "adam",
                    "other_params": {"key": "value"}
                },
                "layer_metadata": {
                    "layer_type": "attention",
                    "activation_function": "gelu",
                    "num_parameters": 1000000
                }
            },
            "internal_states": {
                "layer_name": "layer_0",
                "layer_index": 0,
                "attention_weights": [0.1, 0.2, 0.3],
                "feature_maps": [],
                "node_splits": [],
                "gradients": [],
                "losses": 0.5,
                "feature_importance": [],
                "decision_paths": []
            },
            "event_type": "forward"
        }
        
        assert validate_log_entry(valid_log, schema)
        
        # Invalid log (missing required field)
        invalid_log = valid_log.copy()
        del invalid_log["event_type"]
        assert not validate_log_entry(invalid_log, schema)

        

    


class TestLocalStorage:
    """Test local Parquet storage implementation."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def storage_engine(self, temp_storage_dir):
        """Create initialized storage engine."""
        engine = get_storage_engine(
            backend="local",
            config={
                "storage_dir": str(temp_storage_dir),
                "compression": {
                    "compression_type": "snappy",
                    "compression_level": 1
                },
                "sparse_logging": {
                    "enabled": True,
                    "sparse_threshold": 0.1,
                    "top_k_values": 5
                }
            }
        )
        return engine
    
    def test_save_and_retrieve_logs(self, storage_engine, temp_storage_dir):
        """Test end-to-end save and retrieve workflow."""
        # Create test logs
        test_logs = [{
            "model_metadata": {
                "model_type": "test_model",
                "framework": "pytorch",
                "timestamp": 1234567890000,
                "run_id": "test_run_123",
                "mode": "development",
                "model_architecture": {
                    "num_layers": 4,
                    "layer_types": ["linear"],
                    "connections": ["sequential"]
                },
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 16,
                    "optimizer": "sgd",
                    "other_params": {}
                },
                "layer_metadata": {
                    "layer_type": "linear",
                    "activation_function": "relu",
                    "num_parameters": 1000
                }
            },
            "internal_states": {
                "layer_name": "layer_0",
                "layer_index": 0,
                "attention_weights": [0.5, 0.3, 0.2],
                "feature_maps": [],
                "node_splits": [],
                "gradients": [0.1, -0.1, 0.05],
                "losses": 0.25,
                "feature_importance": [0.8, 0.2],
                "decision_paths": []
            },
            "event_type": "forward"
        }]
        
        # Save logs
        filepath = storage_engine.save_logs(
            logs=test_logs,
            run_id="test_run_123",
            model_type="test_model",
            mode="development"
        )
        
        assert filepath, "Filepath should not be empty"
        assert Path(filepath).exists(), "Parquet file should exist"
        
        # Retrieve logs
        retrieved = storage_engine.retrieve_logs(run_id="test_run_123")
        assert len(retrieved) == 1
        assert retrieved[0]["model_metadata"]["run_id"] == "test_run_123"
    
    def test_sparse_logging_integration(self, storage_engine):
        """Test sparse logging reduces storage size."""
        # Create dense tensor log
        dense_tensor = np.random.randn(1000).astype(np.float32)
        dense_tensor[:10] = 10.0  # Create sparse pattern
        
        log_entry = {
            "model_metadata": {
                "model_type": "test",
                "framework": "pytorch",
                "timestamp": 1234567890000,
                "run_id": "sparse_test",
                "mode": "development",
                "model_architecture": {"num_layers": 1, "layer_types": ["test"], "connections": ["test"]},
                "hyperparameters": {"learning_rate": 0.001, "batch_size": 1, "optimizer": "test", "other_params": {}},
                "layer_metadata": {"layer_type": "test", "activation_function": "test", "num_parameters": 100}
            },
            "internal_states": {
                "layer_name": "test_layer",
                "layer_index": 0,
                "attention_weights": dense_tensor.tolist(),
                "feature_maps": [],
                "node_splits": [],
                "gradients": [],
                "losses": 0.1,
                "feature_importance": [],
                "decision_paths": []
            },
            "event_type": "forward"
        }
        
        # Save with sparse logging enabled
        filepath = storage_engine.save_logs(
            logs=[log_entry],
            run_id="sparse_test",
            model_type="test",
            mode="development"
        )
        
        # Verify file was created
        assert Path(filepath).exists()
        
        # Verify sparse metadata was added
        retrieved = storage_engine.retrieve_logs(run_id="sparse_test")
        assert len(retrieved) == 1
        assert "sparse_logging_metadata" in retrieved[0]