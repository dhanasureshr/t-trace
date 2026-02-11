"""TensorFlow-specific hook implementation for M-TRACE LoggingEngine."""
import tensorflow as tf
import numpy as np
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from weakref import ref as weakref_ref

logger = logging.getLogger(__name__)


class TensorFlowHook:
    """
    Captures layer activations and gradients during TensorFlow model execution.
    
    Implements Section 3.1 requirements:
    - Sparse logging (threshold + top-k filtering)
    - Compression-ready tensor extraction
    - Development/production mode differentiation
    - Minimal overhead via conditional logging
    """
    
    def __init__(self, layer_name: str, layer_index: int, config: Dict[str, Any]):
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.config = config
        self._logs: List[Dict] = []
        self._enabled = True
        self._mode = config.get("mode", "development")
    
    def _sparse_filter(self, tensor: tf.Tensor) -> Dict[str, Any]:
        """
        Apply sparse logging per Section 3.1.4:
        - Threshold-based filtering (abs(value) > threshold)
        - Top-k preservation (always keep top-k values)
        - Shape preservation metadata
        """
        if not self.config.get("sparse_logging", {}).get("enabled", True):
            return {"full_tensor": tensor.numpy().tolist()}
        
        # Convert to numpy for processing
        np_tensor = tensor.numpy()
        abs_values = np.abs(np_tensor)
        threshold = self.config.get("sparse_logging", {}).get("sparse_threshold", 0.1)
        top_k = self.config.get("sparse_logging", {}).get("top_k_values", 5)
        
        # Get indices above threshold
        above_threshold = abs_values > threshold
        indices = np.where(above_threshold)
        
        if len(indices[0]) == 0:
            # Fallback to top-k if nothing above threshold
            flat = abs_values.flatten()
            if len(flat) <= top_k:
                # Return all values if tensor smaller than top_k
                return {
                    "sparse_values": np_tensor.flatten().tolist(),
                    "sparse_indices": np.arange(len(flat)).tolist(),
                    "shape": list(np_tensor.shape),
                    "threshold_applied": threshold,
                    "sparse_type": "all_values"
                }
            
            flat_indices = np.argpartition(flat, -top_k)[-top_k:]
            values = np_tensor.flatten()[flat_indices]
            return {
                "sparse_values": values.tolist(),
                "sparse_indices": flat_indices.tolist(),
                "shape": list(np_tensor.shape),
                "threshold_applied": threshold,
                "sparse_type": "top_k"
            }
        
        # Return sparse representation
        values = np_tensor[indices]
        sparse_indices = [idx.tolist() for idx in indices]
        
        return {
            "sparse_values": values.tolist(),
            "sparse_indices": sparse_indices,
            "shape": list(np_tensor.shape),
            "threshold_applied": threshold,
            "sparse_type": "threshold"
        }
    
    def log_forward(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]],
        outputs: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]],
        training: bool = False
    ) -> None:
        """
        Log forward pass activations with mode-aware filtering.
        
        Production mode: Only log during inference (training=False)
        Development mode: Log during both training and inference
        """
        if not self._enabled:
            return
        
        # Production mode optimization: skip training-time logging
        if self._mode == "production" and training:
            return
        
        try:
            log_entry = {
                "timestamp": time.time(),
                "layer_name": self.layer_name,
                "layer_index": self.layer_index,
                "mode": self._mode,
                "event_type": "forward",
                "training": training
            }
            
            # Capture inputs (if configured and not in production training)
            if "input_data" in self.config.get("custom_fields", []) or self._mode == "development":
                if isinstance(inputs, tf.Tensor):
                    log_entry["input"] = self._sparse_filter(inputs)
                elif isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                    if isinstance(inputs[0], tf.Tensor):
                        log_entry["input"] = self._sparse_filter(inputs[0])
            
            # Capture outputs (always logged)
            if isinstance(outputs, tf.Tensor):
                log_entry["output"] = self._sparse_filter(outputs)
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                if isinstance(outputs[0], tf.Tensor):
                    log_entry["output"] = self._sparse_filter(outputs[0])
            
            # Special handling for attention layers (transformers)
            if "attention_weights" in self.config.get("custom_fields", []):
                if hasattr(outputs, "attention_scores"):
                    log_entry["attention_weights"] = self._sparse_filter(outputs.attention_scores)
                elif isinstance(outputs, tuple) and len(outputs) > 1:
                    # Hugging Face style: (hidden_states, attentions)
                    if isinstance(outputs[1], tf.Tensor):
                        log_entry["attention_weights"] = self._sparse_filter(outputs[1])
            
            self._logs.append(log_entry)
            
        except Exception as e:
            logger.warning(f"Error in forward logging for {self.layer_name}: {e}")
    
    def log_backward(self, gradients: Union[tf.Tensor, List[tf.Tensor]]) -> None:
        """Log gradients during backpropagation (development mode only)."""
        if not self._enabled or self._mode != "development":
            return
        
        try:
            log_entry = {
                "timestamp": time.time(),
                "layer_name": self.layer_name,
                "layer_index": self.layer_index,
                "mode": "development",
                "event_type": "backward"
            }
            
            if isinstance(gradients, tf.Tensor):
                log_entry["gradients"] = self._sparse_filter(gradients)
            elif isinstance(gradients, (list, tuple)) and len(gradients) > 0:
                if isinstance(gradients[0], tf.Tensor):
                    log_entry["gradients"] = self._sparse_filter(gradients[0])
            
            self._logs.append(log_entry)
            
        except Exception as e:
            logger.warning(f"Error in backward logging for {self.layer_name}: {e}")
    
    def get_logs(self) -> List[Dict]:
        """Return collected logs and clear buffer."""
        logs = self._logs.copy()
        self._logs.clear()
        return logs
    
    def clear_logs(self) -> None:
        """Clear internal log buffer."""
        self._logs.clear()
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False


class LoggingModelWrapper(tf.keras.Model):
    """
    Wrapper model that intercepts layer executions for logging.
    
    Implements Section 3.1 requirements through:
    - Layer instrumentation without graph modification
    - Gradient capture via GradientTape integration
    - Minimal performance overhead via conditional execution
    - Support for both eager and graph execution modes
    
    Usage:
        model = YourModel()
        wrapped_model = LoggingModelWrapper(model, hooks, config)
        outputs = wrapped_model(inputs, training=True)  # Logs automatically captured
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        hooks: List[TensorFlowHook],
        config: Dict[str, Any],
        layer_mapping: Dict[str, int],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.wrapped_model = model
        self.hooks = hooks
        self.config = config
        self.layer_mapping = layer_mapping  # {layer_name: hook_index}
        self._mode = config.get("mode", "development")
        self._current_tape = None  # For gradient capture
        
        # Build the model if not already built
        if not hasattr(model, '_is_graph_network') or not model.built:
            logger.debug("Model not built - will build during first call")
    
    @tf.function
    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass with integrated logging.
        
        Strategy:
        1. In development mode: Use explicit layer iteration to capture intermediates
        2. In production mode: Minimal overhead path (only final output logging)
        3. Always preserve original model behavior and signatures
        """
        if self._mode == "production" and training:
            # Production training: minimal overhead - skip layer logging
            return self.wrapped_model(inputs, training=training, **kwargs)
        
        # Development mode or production inference: capture layer activations
        return self._call_with_logging(inputs, training=training, **kwargs)
    
    def _call_with_logging(self, inputs, training=None, **kwargs):
        """Execute model with layer-wise logging."""
        # Get layer execution order
        layers = self._get_executable_layers()
        
        # Execute layers sequentially with logging
        x = inputs
        for layer in layers:
            if layer.name in self.layer_mapping:
                hook_idx = self.layer_mapping[layer.name]
                hook = self.hooks[hook_idx]
                
                # Capture inputs before layer execution
                layer_inputs = x
                
                # Execute layer
                if hasattr(layer, 'call'):
                    x = layer(x, training=training, **kwargs)
                else:
                    x = layer(x, **kwargs)
                
                # Log forward pass
                hook.log_forward(layer_inputs, x, training=bool(training))
            else:
                # Execute without logging
                if hasattr(layer, 'call'):
                    x = layer(x, training=training, **kwargs)
                else:
                    x = layer(x, **kwargs)
        
        return x
    
    def _get_executable_layers(self) -> List[tf.keras.layers.Layer]:
        """Extract executable layers from model architecture."""
        if hasattr(self.wrapped_model, 'layers'):
            # Sequential/Functional API
            return [layer for layer in self.wrapped_model.layers 
                   if not isinstance(layer, tf.keras.layers.InputLayer)]
        
        # Subclassed models: cannot introspect layers reliably
        logger.warning(
            "Subclassed model detected - layer introspection limited. "
            "Only final output logging will be available."
        )
        return []
    
    def train_step(self, data):
        """
        Custom training step with gradient capture (development mode only).
        
        Overrides Keras default train_step to integrate with tf.GradientTape
        for gradient logging per Section 3.1 requirements.
        """
        if self._mode != "development":
            # Production mode: use standard train_step for performance
            return super().train_step(data)
        
        # Unpack data
        if len(data) == 2:
            x, y = data
        else:
            x, y, sample_weight = data
        
        # Forward pass with gradient tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Log gradients for relevant layers
        self._log_gradients(gradients, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def _log_gradients(self, gradients: List[tf.Tensor], variables: List[tf.Variable]) -> None:
        """Log gradients associated with each layer."""
        if not gradients or not variables:
            return
        
        # Group gradients by layer
        layer_gradients = {}
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            
            # Extract layer name from variable name (heuristic)
            var_name = var.name
            layer_name = self._extract_layer_name(var_name)
            
            if layer_name in self.layer_mapping:
                hook_idx = self.layer_mapping[layer_name]
                if hook_idx not in layer_gradients:
                    layer_gradients[hook_idx] = []
                layer_gradients[hook_idx].append(grad)
        
        # Log aggregated gradients per layer
        for hook_idx, grads in layer_gradients.items():
            # Average gradients for the layer
            if len(grads) > 1:
                avg_grad = tf.add_n(grads) / len(grads)
            else:
                avg_grad = grads[0]
            
            self.hooks[hook_idx].log_backward(avg_grad)
    
    def _extract_layer_name(self, variable_name: str) -> str:
        """
        Heuristic to extract layer name from variable name.
        
        Examples:
            "dense/kernel:0" -> "dense"
            "attention_layer/query/kernel:0" -> "attention_layer"
        """
        # Remove trailing :0 and split by /
        clean_name = variable_name.split(":")[0]
        parts = clean_name.split("/")
        
        if len(parts) > 1:
            return parts[-2]  # Parent directory is usually layer name
        return parts[0].split("_")[0]  # Fallback: first part before underscore
    
    def get_logs(self) -> List[Dict]:
        """Collect logs from all hooks."""
        all_logs = []
        for hook in self.hooks:
            all_logs.extend(hook.get_logs())
        return all_logs
    
    # Proxy methods to wrapped model
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)


class TensorFlowLoggingEngine:
    """
    Framework-specific engine for TensorFlow/Keras models.
    
    Implements M-TRACE Section 3.1 requirements:
    - Automatic layer instrumentation without model modification
    - Development/production mode differentiation
    - Sparse logging with threshold + top-k filtering
    - Gradient capture during training (development mode)
    - Minimal overhead via conditional execution paths
    - Support for Sequential, Functional, and limited Subclassed models
    
    Architecture decisions:
    1. Uses model wrapping instead of layer wrapping to avoid graph reconstruction issues
    2. Leverages Keras train_step override for gradient capture (cleaner than custom_gradient)
    3. Production mode skips training-time logging for maximum performance
    4. Development mode provides comprehensive introspection
    """
    
    SUPPORTED_LAYER_TYPES = (
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
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any]):
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                f"Model must be a tf.keras.Model instance, got {type(model)}"
            )
        
        self.model_ref = weakref_ref(model)
        self.original_model = model
        self.config = config
        self.hooks: List[TensorFlowHook] = []
        self.layer_mapping: Dict[str, int] = {}  # layer_name -> hook_index
        self.wrapped_model: Optional[LoggingModelWrapper] = None
        self._enabled = False
        self._mode = config.get("mode", "development")
    
    def _identify_loggable_layers(self, model: tf.keras.Model) -> List[Tuple[str, tf.keras.layers.Layer]]:
        """
        Identify layers suitable for logging instrumentation.
        
        Returns:
            List of (layer_name, layer) tuples for layers to instrument
        """
        loggable_layers = []
        
        # Sequential models
        if isinstance(model, tf.keras.Sequential):
            for layer in model.layers:
                if isinstance(layer, self.SUPPORTED_LAYER_TYPES):
                    loggable_layers.append((layer.name, layer))
            return loggable_layers
        
        # Functional models
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if isinstance(layer, self.SUPPORTED_LAYER_TYPES):
                    loggable_layers.append((layer.name, layer))
            return loggable_layers
        
        # Subclassed models: limited introspection
        logger.warning(
            "Subclassed tf.keras.Model detected. Layer introspection limited. "
            "Only layers accessible via .layers property will be instrumented."
        )
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if isinstance(layer, self.SUPPORTED_LAYER_TYPES):
                    loggable_layers.append((layer.name, layer))
        
        return loggable_layers
    
    def enable(self) -> None:
        """Enable logging by creating wrapped model with instrumentation."""
        if self._enabled:
            logger.warning("TensorFlow logging already enabled")
            return
        
        model = self.model_ref()
        if model is None:
            raise RuntimeError("Model reference lost - cannot enable logging")
        
        # Identify layers to instrument
        loggable_layers = self._identify_loggable_layers(model)
        
        if not loggable_layers and self._mode == "development":
            logger.warning(
                "No loggable layers found in model. "
                "Logging will only capture final outputs and losses."
            )
        
        # Create hooks for each loggable layer
        for layer_index, (layer_name, _) in enumerate(loggable_layers):
            hook = TensorFlowHook(
                layer_name=layer_name,
                layer_index=layer_index,
                config=self.config
            )
            self.hooks.append(hook)
            self.layer_mapping[layer_name] = layer_index
        
        # Create wrapped model
        self.wrapped_model = LoggingModelWrapper(
            model=model,
            hooks=self.hooks,
            config=self.config,
            layer_mapping=self.layer_mapping
        )
        
        # Transfer compiled state if model was compiled
        if hasattr(model, 'optimizer') and model.optimizer:
            self.wrapped_model.compile(
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
                loss_weights=model.loss_weights,
                weighted_metrics=model.weighted_metrics,
                run_eagerly=model.run_eagerly
            )
            logger.debug("Transferred compilation state to wrapped model")
        
        self._enabled = True
        logger.info(
            f"Enabled TensorFlow logging in {self._mode} mode for "
            f"{len(self.hooks)} layers (run_id: {self.config.get('run_id', 'unknown')})"
        )
    
    def disable(self) -> None:
        """Disable logging and cleanup resources."""
        if not self._enabled:
            return
        
        # Clear hooks
        for hook in self.hooks:
            hook.disable()
        self.hooks.clear()
        self.layer_mapping.clear()
        
        # Release wrapped model
        self.wrapped_model = None
        
        self._enabled = False
        logger.info("Disabled TensorFlow logging")
    
    def get_wrapped_model(self) -> tf.keras.Model:
        """
        Get the instrumented model for training/inference.
        
        IMPORTANT: Users MUST use this model instead of the original when logging is enabled.
        
        Returns:
            Wrapped model with logging instrumentation
        
        Raises:
            RuntimeError: If logging is not enabled
        """
        if not self._enabled:
            raise RuntimeError(
                "Logging not enabled. Call enable() before getting wrapped model."
            )
        
        if self.wrapped_model is None:
            raise RuntimeError("Wrapped model not available - instrumentation failed")
        
        return self.wrapped_model
    
    def collect_logs(self) -> List[Dict]:
        """Collect all buffered logs from hooks and wrapped model."""
        if not self._enabled or self.wrapped_model is None:
            return []
        
        # Collect from hooks
        logs = []
        for hook in self.hooks:
            logs.extend(hook.get_logs())
        
        # Collect from wrapped model (aggregated logs)
        logs.extend(self.wrapped_model.get_logs())
        
        return logs
    
    def is_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled


# Public utility functions
def is_tensorflow_model(model: Any) -> bool:
    """Check if object is a TensorFlow/Keras model."""
    try:
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)
    except ImportError:
        return False


def requires_custom_training_loop(model: tf.keras.Model) -> bool:
    """
    Check if model requires custom training loop for full logging support.
    
    Returns:
        True if model is subclassed and overrides train_step/test_step
    """
    # Check if model has custom train_step implementation
    if type(model).train_step != tf.keras.Model.train_step:
        return True
    
    # Check if model is subclassed (not Sequential/Functional)
    if not hasattr(model, '_is_graph_network') and not isinstance(model, tf.keras.Sequential):
        return True
    
    return False