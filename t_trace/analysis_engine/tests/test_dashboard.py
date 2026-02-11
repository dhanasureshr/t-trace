"""Unit tests for AnalysisEngine dashboard."""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)

from t_trace.analysis_engine.data_loader import DataLoader
from t_trace.analysis_engine.visualizations import Visualizations


class TestVisualizations:
    """Test visualization components."""
    
    def test_create_attention_heatmap(self):
        """Test attention heatmap creation."""
        viz = Visualizations()
        
        # Test with 2D array
        weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        fig = viz.create_attention_heatmap(weights, tokens=["A", "B", "C"])
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) == 1
    
    def test_create_loss_curve(self):
        """Test loss curve creation."""
        viz = Visualizations()
        
        # Create mock DataFrame with losses
        df = pd.DataFrame({
            "timestamp": [1, 2, 3, 4, 5],
            "internal_states": [
                {"losses": 0.5},
                {"losses": 0.4},
                {"losses": 0.3},
                {"losses": 0.25},
                {"losses": 0.2}
            ]
        })
        
        fig = viz.create_loss_curve(df)
        assert fig is not None
        assert hasattr(fig, 'data')


class TestAnalysisDashboardIntegration:
    """Integration tests for AnalysisEngine components."""
    
    def test_list_runs_with_real_files(self):
        """Test listing runs with actual Parquet files in temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create development directory
            dev_dir = Path(tmpdir) / "development"
            dev_dir.mkdir()
            
            # Create a valid Parquet file with correct naming
            test_file = dev_dir / "model_run_testmodel_20260211_164344_a1b2c3d4.parquet"
            test_file.touch()
            
            # Initialize loader with temp directory
            loader = DataLoader(storage_config={"directory": tmpdir})
            
            # List runs
            runs = loader.list_runs()
            
            assert len(runs) == 1
            assert runs[0]["run_id"] == "a1b2c3d4"
            assert runs[0]["model_type"] == "testmodel"
            assert runs[0]["mode"] == "development"
    
    def test_load_run_logs_with_mock_data(self):
        """Test loading logs with mock Parquet data."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dev_dir = Path(tmpdir) / "development"
            dev_dir.mkdir()
            
            # Create minimal valid Parquet file
            schema = pa.schema([
                ("model_metadata", pa.struct([
                    ("model_type", pa.string()),
                    ("framework", pa.string()),
                    ("timestamp", pa.timestamp("ms")),
                    ("run_id", pa.string()),
                    ("mode", pa.string()),
                    ("model_architecture", pa.struct([
                        ("num_layers", pa.int32()),
                        ("layer_types", pa.list_(pa.string())),
                        ("connections", pa.list_(pa.string())),
                    ])),
                    ("hyperparameters", pa.struct([
                        ("learning_rate", pa.float64()),
                        ("batch_size", pa.int32()),
                        ("optimizer", pa.string()),
                        ("other_params", pa.map_(pa.string(), pa.string())),
                    ])),
                    ("layer_metadata", pa.struct([
                        ("layer_type", pa.string()),
                        ("activation_function", pa.string()),
                        ("num_parameters", pa.int64()),
                    ])),
                ])),
                ("internal_states", pa.struct([
                    ("layer_name", pa.string()),
                    ("layer_index", pa.int32()),
                    ("attention_weights", pa.list_(pa.float32())),
                    ("feature_maps", pa.list_(pa.float32())),
                    ("node_splits", pa.list_(pa.string())),
                    ("gradients", pa.list_(pa.float32())),
                    ("losses", pa.float64()),
                    ("feature_importance", pa.list_(pa.float32())),
                    ("decision_paths", pa.list_(pa.string())),
                ])),
                ("event_type", pa.string()),
            ])
            
            # Create minimal record
            record = [{
                "model_metadata": {
                    "model_type": "test",
                    "framework": "pytorch",
                    "timestamp": 1234567890000,
                    "run_id": "test_run",
                    "mode": "development",
                    "model_architecture": {
                        "num_layers": 1,
                        "layer_types": ["linear"],
                        "connections": ["sequential"]
                    },
                    "hyperparameters": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "optimizer": "adam",
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
                    "gradients": [],
                    "losses": 0.25,
                    "feature_importance": [],
                    "decision_paths": []
                },
                "event_type": "forward"
            }]
            
            table = pa.Table.from_pylist(record, schema=schema)
            filepath = dev_dir / "model_run_test_20260211_164344_test_run.parquet"
            pq.write_table(table, filepath)
            
            # Load logs
            loader = DataLoader(storage_config={"directory": tmpdir})
            df = loader.load_run_logs("test_run")
            
            assert df is not None
            assert len(df) == 1
            assert df.iloc[0]["model_metadata"]["run_id"] == "test_run"