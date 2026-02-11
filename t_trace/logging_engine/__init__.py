"""Public API for M-TRACE LoggingEngine."""
from typing import Any, Optional
from .base import LoggingEngine

__all__ = ["enable_logging", "LoggingEngine"]

# Global engine instance for convenience API
_GLOBAL_ENGINE: LoggingEngine = LoggingEngine()


def enable_logging(model: Any, mode: str = "production", config_path: Optional[str] = None) -> LoggingEngine:
    """
    Public API: Enable M-TRACE logging for any machine learning model.
    
    This function automatically detects the model framework (PyTorch, TensorFlow, scikit-learn),
    attaches appropriate hooks/callbacks, and begins capturing internal states in real-time.
    
    Args:
        model: Machine learning model instance
        mode: Logging mode - "development" (detailed logging with gradients) 
              or "production" (lightweight inference-only logging)
        config_path: Optional path to YAML configuration file. If None, uses default location.
    
    Returns:
        Configured LoggingEngine instance
    
    Example:
        >>> from transformers import BertModel
        >>> model = BertModel.from_pretrained('bert-base-uncased')
        >>> engine = enable_logging(model, mode="development")
        >>> # Use model normally - logs are captured automatically
        >>> outputs = model(input_ids)
        >>> # Retrieve logs
        >>> logs = engine.collect_logs()
    """
    global _GLOBAL_ENGINE
    
    # Reinitialize engine if config path changed
    if config_path:
        _GLOBAL_ENGINE = LoggingEngine(config_path=config_path)
    
    _GLOBAL_ENGINE.enable_logging(model, mode=mode)
    return _GLOBAL_ENGINE