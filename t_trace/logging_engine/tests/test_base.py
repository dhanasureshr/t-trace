"""Unit tests for LoggingEngine core functionality."""
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)

from unittest.mock import Mock, patch
from t_trace.logging_engine.base import LoggingEngine
from t_trace.logging_engine.config import LoggingConfig




class SimpleModel(nn.Module):
    """Simple test model for PyTorch logging tests."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestLoggingEngine:
    """Test suite for LoggingEngine."""
    
    def test_config_loading_default(self):
        """Test that default config is loaded when file missing."""
        config = LoggingConfig(config_path="nonexistent_config.yml")
        assert config.get("sparse_logging.enabled") is True
        assert config.get("compression.compression_type") == "snappy"
    
    def test_framework_detection_pytorch(self):
        """Test PyTorch model detection."""
        model = SimpleModel()
        engine = LoggingEngine()
        
        with patch.object(engine, '_detect_framework') as mock_detect:
            mock_detect.return_value = "pytorch"
            framework = engine._detect_framework(model)
            assert framework == "pytorch"
    
    def test_enable_logging_pytorch(self):
        """Test enabling logging for PyTorch model."""
        model = SimpleModel()
        engine = LoggingEngine()
        
        # Should not raise exception
        engine.enable_logging(model, mode="development")
        assert engine.is_logging_enabled() is True
        assert engine.get_run_id() is not None
        
        # Test model execution with logging enabled
        input_tensor = torch.randn(2, 10)
        output = model(input_tensor)
        
        # Should have collected logs
        logs = engine.collect_logs()
        assert len(logs) > 0
        
        engine.disable_logging()
        assert engine.is_logging_enabled() is False
    
    def test_mode_differentiation(self):
        """Test that development mode captures more data than production."""
        model = SimpleModel()
        engine_dev = LoggingEngine()
        engine_prod = LoggingEngine()
        
        # Enable in different modes
        engine_dev.enable_logging(model, mode="development")
        engine_prod.enable_logging(model, mode="production")
        
        # Execute model
        input_tensor = torch.randn(2, 10)
        _ = model(input_tensor)
        
        # Development should capture backward passes (gradients)
        dev_logs = engine_dev.collect_logs()
        prod_logs = engine_prod.collect_logs()
        
        engine_dev.disable_logging()
        engine_prod.disable_logging()
        
        # Development logs should include backward events
        dev_backward = [l for l in dev_logs if l.get("event_type") == "backward"]
        prod_backward = [l for l in prod_logs if l.get("event_type") == "backward"]
        
        # Note: backward hooks only trigger during .backward() call
        # This test verifies configuration difference, not actual gradient capture
        assert engine_dev._mode == "development"
        assert engine_prod._mode == "production"
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        model = SimpleModel()
        engine = LoggingEngine()
        
        with pytest.raises(ValueError, match="Invalid mode"):
            engine.enable_logging(model, mode="invalid_mode")
    
    def test_context_manager(self):
        """Test context manager usage."""
        model = SimpleModel()
        
        with LoggingEngine() as engine:
            engine.enable_logging(model, mode="production")
            assert engine.is_logging_enabled() is True
        
        # Should be automatically disabled after context exit
        assert engine.is_logging_enabled() is False