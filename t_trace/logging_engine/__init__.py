"""Public API for M-TRACE LoggingEngine."""
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from .base import LoggingEngine

__all__ = ["enable_logging", "LoggingEngine"]

# Global engine instance for convenience API
_GLOBAL_ENGINE: Optional[LoggingEngine] = None


def enable_logging(
    model: Any, 
    mode: str = "production", 
    config_path: Optional[str] = None
) -> Union[LoggingEngine, Tuple[LoggingEngine, tf.keras.callbacks.Callback]]:
    """
    Public API: Enable M-TRACE logging for any machine learning model.
    
    AUTOMATIC FRAMEWORK DETECTION:
    - PyTorch/scikit-learn: Hooks attached directly → returns LoggingEngine only
    - TensorFlow: Requires Keras callback → returns (LoggingEngine, callback) tuple
    
    Args:
        model: Machine learning model instance (PyTorch nn.Module, TensorFlow Keras Model, or sklearn BaseEstimator)
        mode: Logging mode - "development" (detailed logging with gradients) 
              or "production" (lightweight inference-only logging)
        config_path: Optional path to YAML configuration file. If None, uses default location.
    
    Returns:
        - For PyTorch/scikit-learn: LoggingEngine instance
        - For TensorFlow: Tuple of (LoggingEngine, tf.keras.callbacks.Callback)
    
    Examples:
        # PyTorch usage (no callback needed)
        >>> engine = enable_logging(pytorch_model, mode="development")
        >>> output = pytorch_model(input)
        
        # TensorFlow usage (callback required)
        >>> engine, callback = enable_logging(tf_model, mode="development")
        >>> tf_model.fit(x_train, y_train, callbacks=[callback])
    
    Raises:
        ValueError: If model framework is unsupported or mode is invalid
    """
    global _GLOBAL_ENGINE
    
    # Initialize or reinitialize engine if config path changed
    if _GLOBAL_ENGINE is None or config_path is not None:
        _GLOBAL_ENGINE = LoggingEngine(config_path=config_path)
    
    # Enable logging and capture framework-specific return value
    framework_return = _GLOBAL_ENGINE.enable_logging(model, mode=mode)
    
    # TensorFlow requires callback integration; PyTorch/scikit-learn do not
    try:
        import tensorflow as tf
        if isinstance(framework_return, tf.keras.callbacks.Callback):
            return (_GLOBAL_ENGINE, framework_return)
    except ImportError:
        pass
    
    # PyTorch/scikit-learn: hooks attached directly to model
    return _GLOBAL_ENGINE