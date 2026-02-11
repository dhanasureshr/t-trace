"""TensorFlow-specific callback implementation for M-TRACE LoggingEngine."""
from pyexpat import model
import tensorflow as tf
import numpy as np
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from weakref import ref as weakref_ref

logger = logging.getLogger(__name__)


class MTraceCallback(tf.keras.callbacks.Callback):
    """
    Keras callback for M-TRACE logging that captures model internals during execution.
    
    Implements Section 2.1.2 requirement: "For TensorFlow: Use tf.keras.callbacks.Callback"
    AND Section 3.1.2 schema compliance with nested log structure.
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any]):
        super().__init__()
        self.model_ref = weakref_ref(model)
        self.config = config
        self._mode = config.get("mode", "production")
        self._logs: List[Dict] = []
        self._loggable_layers: List[Tuple[str, tf.keras.layers.Layer, int]] = []
        self._enabled = True
        self._run_metadata = {
            "run_id": config.get("run_id", "unknown"),
            "start_time": None,
            "batch_count": 0
        }
        
        # Extract model metadata per Section 3.1.2 requirements (used in ALL log entries)
        self._base_model_metadata = self._extract_base_model_metadata(model)
        
        # Identify loggable layers at initialization
        self._loggable_layers = self._identify_loggable_layers(model)
        logger.debug(
            f"MTraceCallback initialized for {model.__class__.__name__} "
            f"with {len(self._loggable_layers)} loggable layers"
        )
        
        # For development mode: setup gradient capture via train_step monkey-patching
        if self._mode == "development":
            self._patch_train_step_for_gradients(model)
    
    def _extract_base_model_metadata(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Extract BASE model metadata (shared across all log entries)."""
        try:
            trainable_params = int(sum(tf.keras.backend.count_params(w) 
                                     for w in model.trainable_weights))
            non_trainable_params = int(sum(tf.keras.backend.count_params(w) 
                                         for w in model.non_trainable_weights))
            
            # Extract architecture details
            layer_types = []
            if hasattr(model, 'layers'):
                layer_types = [layer.__class__.__name__ for layer in model.layers 
                             if not isinstance(layer, tf.keras.layers.InputLayer)]
            
            return {
                "model_type": model.__class__.__name__.lower(),
                "framework": "tensorflow",
                "num_parameters": trainable_params + non_trainable_params,
                "num_trainable_parameters": trainable_params,
                "input_shape": getattr(model, 'input_shape', None),
                "output_shape": getattr(model, 'output_shape', None),
                "layer_count": len(layer_types),
                "layer_types": layer_types,
                "connections": ["sequential"]  # Simplified for Keras models
            }
        except Exception as e:
            logger.warning(f"Failed to extract base model metadata: {e}")
            return {
                "model_type": model.__class__.__name__.lower(),
                "framework": "tensorflow",
                "num_parameters": 0,
                "num_trainable_parameters": 0,
                "input_shape": None,
                "output_shape": None,
                "layer_count": 0,
                "layer_types": [],
                "connections": ["sequential"]
            }
    
    def _identify_loggable_layers(self, model: tf.keras.Model) -> List[Tuple[str, tf.keras.layers.Layer, int]]:
        """Identify layers suitable for logging instrumentation."""
        loggable = []
        layer_index = 0
        
        if isinstance(model, tf.keras.Sequential):
            for layer in model.layers:
                if self._is_loggable_layer(layer):
                    loggable.append((layer.name, layer, layer_index))
                    layer_index += 1
            return loggable
        
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                if self._is_loggable_layer(layer):
                    loggable.append((layer.name, layer, layer_index))
                    layer_index += 1
        
        return loggable
    
    def _is_loggable_layer(self, layer: tf.keras.layers.Layer) -> bool:
        """Check if layer type should be logged per M-TRACE requirements."""
        loggable_types = (
            tf.keras.layers.Dense,
            tf.keras.layers.Conv1D,
            tf.keras.layers.Conv2D,
            tf.keras.layers.Conv3D,
            tf.keras.layers.LSTM,
            tf.keras.layers.GRU,
            tf.keras.layers.MultiHeadAttention,
            tf.keras.layers.Attention,
            tf.keras.layers.Embedding,
        )
        return isinstance(layer, loggable_types)
    
    def _patch_train_step_for_gradients(self, model: tf.keras.Model) -> None:
        """Monkey-patch train_step to capture gradients in development mode."""
        if not hasattr(model, '_original_train_step'):
            model._original_train_step = model.train_step
        
        original_train_step = model._original_train_step
        
        def wrapped_train_step(data):
            if len(data) == 2:
                x, y = data
            else:
                x, y, sample_weight = data
            
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = model.compiled_loss(
                    y, y_pred,
                    sample_weight=sample_weight if len(data) > 2 else None,
                    regularization_losses=model.losses
                )
            
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            
            if self._enabled:
                self._log_gradients(gradients, trainable_vars, loss)
            
            return original_train_step(data)
        
        model.train_step = wrapped_train_step
        logger.debug("Monkey-patched train_step for gradient capture (development mode)")
    
    def _log_gradients(
        self, 
        gradients: List[tf.Tensor], 
        variables: List[tf.Variable],
        loss: tf.Tensor
    ) -> None:
        """Log gradients with M-TRACE schema-compliant nested structure."""
        if not gradients or not variables:
            return
        
        # Group gradients by layer name
        layer_grads = {}
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            
            layer_name = self._extract_layer_name_from_variable(var.name)
            for idx, (lname, _, layer_idx) in enumerate(self._loggable_layers):
                if lname == layer_name:
                    if layer_idx not in layer_grads:
                        layer_grads[layer_idx] = []
                    layer_grads[layer_idx].append(grad)
                    break
        
        timestamp_ms = int(time.time() * 1000)  # Parquet-compatible timestamp
        
        for layer_idx, grads in layer_grads.items():
            layer_name, layer_obj, _ = self._loggable_layers[layer_idx]
            
            if len(grads) > 1:
                avg_grad = tf.add_n(grads) / len(grads)
            else:
                avg_grad = grads[0]
            
            # BUILD SCHEMA-COMPLIANT LOG ENTRY (nested structure)
            log_entry = {
                "model_metadata": {
                    **self._base_model_metadata,
                    "timestamp": timestamp_ms,
                    "run_id": self._run_metadata["run_id"],
                    "mode": self._mode,
                    "hyperparameters": {
                        "learning_rate": float(model.optimizer.learning_rate.numpy()) 
                                         if hasattr(model, 'optimizer') and model.optimizer else 0.001,
                        "batch_size": 32,  # Will be overridden by actual batch size
                        "optimizer": model.optimizer.__class__.__name__ 
                                    if hasattr(model, 'optimizer') and model.optimizer else "adam",
                        "other_params": {}
                    },
                    "layer_metadata": {
                        "layer_type": layer_obj.__class__.__name__,
                        "activation_function": self._get_activation_function(layer_obj),
                        "num_parameters": int(sum(tf.keras.backend.count_params(w) 
                                               for w in layer_obj.trainable_weights))
                    }
                },
                "internal_states": {
                    "layer_name": layer_name,
                    "layer_index": layer_idx,
                    "attention_weights": [],  # Empty for non-attention layers
                    "feature_maps": [],
                    "node_splits": [],
                    "gradients": self._sparse_filter(avg_grad),
                    "losses": float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss),
                    "feature_importance": [],
                    "decision_paths": [],
                    "output_activations": []  # Will be populated in forward pass
                },
                "event_type": "backward"
            }
            self._logs.append(log_entry)
    
    def _extract_layer_name_from_variable(self, variable_name: str) -> str:
        """Extract layer name from TensorFlow variable name using scope heuristics."""
        clean_name = variable_name.split(":")[0].split("@")[0]
        parts = clean_name.split("/")
        if len(parts) >= 2:
            return parts[-2]
        return parts[0].split("_")[0]
    
    def _get_activation_function(self, layer: tf.keras.layers.Layer) -> str:
        """Detect activation function type from layer configuration."""
        if hasattr(layer, 'activation'):
            act_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            return act_name.lower()
        elif isinstance(layer, tf.keras.layers.ReLU):
            return "relu"
        elif isinstance(layer, tf.keras.layers.GELU):
            return "gelu"
        return "none"
    
    def _sparse_filter(self, tensor: Union[tf.Tensor, np.ndarray]) -> Dict[str, Any]:
        """Apply sparse logging per Section 3.1.4 requirements."""
        np_tensor = tensor.numpy() if isinstance(tensor, tf.Tensor) else tensor
        
        if np_tensor.size == 0:
            return {"empty": True, "shape": list(np_tensor.shape)}
        if np_tensor.ndim == 0:
            return {"scalar": float(np_tensor)}
        
        sparse_config = self.config.get("sparse_logging", {})
        if not sparse_config.get("enabled", True):
            return {"full_tensor": np_tensor.tolist()}
        
        abs_values = np.abs(np_tensor)
        threshold = sparse_config.get("sparse_threshold", 0.1)
        top_k = sparse_config.get("top_k_values", 5)
        
        mask = abs_values > threshold
        indices = np.where(mask)
        
        if np.sum(mask) == 0:
            flat = abs_values.flatten()
            if len(flat) <= top_k:
                return {
                    "sparse_values": np_tensor.flatten().tolist(),
                    "sparse_indices": np.arange(len(flat)).tolist(),
                    "shape": list(np_tensor.shape),
                    "sparse_type": "all_values"
                }
            
            top_indices = np.argpartition(flat, -top_k)[-top_k:]
            return {
                "sparse_values": np_tensor.flatten()[top_indices].tolist(),
                "sparse_indices": top_indices.tolist(),
                "shape": list(np_tensor.shape),
                "threshold_applied": threshold,
                "sparse_type": "top_k"
            }
        
        values = np_tensor[indices]
        return {
            "sparse_values": values.tolist(),
            "sparse_indices": [idx.tolist() for idx in indices],
            "shape": list(np_tensor.shape),
            "threshold_applied": threshold,
            "sparse_type": "threshold"
        }
    
    # ===== Keras Callback Lifecycle Methods =====
    
    def on_train_begin(self, logs=None):
        self._run_metadata["start_time"] = time.time()
        self._run_metadata["batch_count"] = 0
        logger.info(
            f"M-TRACE logging started (run_id: {self._run_metadata['run_id']}, "
            f"mode: {self._mode})"
        )
    
    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if not self._enabled:
            return
        
        self._run_metadata["batch_count"] += 1
        
        if logs:
            self._log_model_metrics(batch, logs, training=True)
        
        if self._mode == "production":
            return
    
    def on_test_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if not self._enabled:
            return
        
        if logs:
            self._log_model_metrics(batch, logs, training=False)
    
    def on_predict_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if not self._enabled:
            return
        
        if logs:
            self._log_model_metrics(batch, logs, training=False)
    
    def _log_model_metrics(self, batch: int, logs: Dict, training: bool):
        """Log model metrics with M-TRACE schema-compliant nested structure."""
        timestamp_ms = int(time.time() * 1000)
        
        # BUILD SCHEMA-COMPLIANT LOG ENTRY (nested structure)
        log_entry = {
            "model_metadata": {
                **self._base_model_metadata,
                "timestamp": timestamp_ms,
                "run_id": self._run_metadata["run_id"],
                "mode": self._mode,
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": batch,
                    "optimizer": "adam",
                    "other_params": {}
                },
                "layer_metadata": {
                    "layer_type": "output_layer",
                    "activation_function": "softmax" if training else "linear",
                    "num_parameters": 0
                }
            },
            "internal_states": {
                "layer_name": "model_output",
                "layer_index": -1,
                "attention_weights": [],
                "feature_maps": [],
                "node_splits": [],
                "gradients": [],
                "losses": logs.get("loss", 0.0),
                "feature_importance": [],
                "decision_paths": [],
                "output_activations": []  # Not captured in callbacks - documented limitation
            },
            "event_type": "forward"
        }
        
        # Add attention weights if configured and available
        custom_fields = self.config.get("custom_fields", [])
        if "attention_weights" in custom_fields:
            model = self.model_ref()
            if model and hasattr(model, 'layers') and model.layers:
                last_layer = model.layers[-1]
                if isinstance(last_layer, (tf.keras.layers.MultiHeadAttention, tf.keras.layers.Attention)):
                    log_entry["internal_states"]["attention_layer_type"] = last_layer.__class__.__name__
        
        self._logs.append(log_entry)
    
    def on_train_end(self, logs=None):
        duration = time.time() - self._run_metadata["start_time"] if self._run_metadata["start_time"] else 0
        logger.info(
            f"M-TRACE logging ended (run_id: {self._run_metadata['run_id']}, "
            f"duration: {duration:.2f}s, batches: {self._run_metadata['batch_count']})"
        )
    
    # ===== Public API Methods =====
    
    def get_logs(self) -> List[Dict]:
        """Return collected logs in schema-compliant nested format."""
        logs = self._logs.copy()
        self._logs.clear()
        return logs
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False
        
        model = self.model_ref()
        if model and hasattr(model, '_original_train_step'):
            model.train_step = model._original_train_step
            delattr(model, '_original_train_step')
            logger.debug("Restored original train_step")
    
    def is_enabled(self) -> bool:
        return self._enabled


class TensorFlowLoggingEngine:
    """
    Framework-specific engine for TensorFlow/Keras models using standard callback pattern.
    Returns schema-compliant logs matching PyTorch implementation structure.
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any]):
        if not self._is_tensorflow_model(model):
            raise ValueError(
                f"Model must be a tf.keras.Model instance, got {type(model)}"
            )
        
        self.model_ref = weakref_ref(model)
        self.config = config
        self.callback: Optional[MTraceCallback] = None
        self._enabled = False
    
    @staticmethod
    def _is_tensorflow_model(model: Any) -> bool:
        try:
            import tensorflow as tf
            return isinstance(model, (tf.keras.Model, tf.Module))
        except ImportError:
            return False
    
    def enable(self) -> None:
        if self._enabled:
            logger.warning("TensorFlow logging already enabled")
            return
        
        model = self.model_ref()
        if model is None:
            raise RuntimeError("Model reference lost - cannot enable logging")
        
        callback_config = {**self.config, "run_id": self.config.get("run_id", "unknown")}
        self.callback = MTraceCallback(model, callback_config)
        self._enabled = True
        logger.info(
            f"Enabled TensorFlow logging in {self.config.get('mode', 'production')} "
            f"mode via Keras callback (run_id: {callback_config['run_id']})"
        )
    
    def disable(self) -> None:
        if not self._enabled:
            return
        
        if self.callback:
            self.callback.disable()
        
        self._enabled = False
        self.callback = None
        logger.info("Disabled TensorFlow logging")
    
    def get_callback(self) -> tf.keras.callbacks.Callback:
        if not self._enabled or self.callback is None:
            raise RuntimeError(
                "Logging not enabled. Call enable() before getting callback."
            )
        return self.callback
    
    def collect_logs(self) -> List[Dict]:
        if not self._enabled or self.callback is None:
            return []
        return self.callback.get_logs()
    
    def is_enabled(self) -> bool:
        return self._enabled


def is_tensorflow_model(model: Any) -> bool:
    try:
        import tensorflow as tf
        return isinstance(model, (tf.keras.Model, tf.Module))
    except ImportError:
        return False