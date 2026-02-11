"""PyTorch-specific hook implementation for M-TRACE LoggingEngine."""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Callable
import weakref
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class PyTorchHook:
    """Manages forward/backward hooks for PyTorch modules."""
    
    def __init__(self, layer_name: str, layer_index: int, config: Any):
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.config = config
        self._logs: List[Dict] = []
        self._hook_handles: List[Any] = []
        self._enabled = True
    
    def _sparse_filter(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Apply sparse logging: keep only top-k values or values above threshold."""
        if not self.config.get("sparse_logging.enabled", True):
            return {"full_tensor": tensor.detach().cpu().numpy()}
        
        # Convert to numpy for processing
        np_tensor = tensor.detach().cpu().numpy()
        abs_values = np.abs(np_tensor)
        threshold = self.config.get("sparse_logging.sparse_threshold", 0.1)
        top_k = self.config.get("sparse_logging.top_k_values", 5)
        
        # Get indices of values above threshold
        above_threshold = abs_values > threshold
        indices = np.where(above_threshold)
        
        if len(indices[0]) == 0:
            # Fallback to top-k if nothing above threshold
            flat_indices = np.argpartition(abs_values.flatten(), -top_k)[-top_k:]
            values = np_tensor.flatten()[flat_indices]
            return {
                "sparse_values": values.tolist(),
                "sparse_indices": flat_indices.tolist(),
                "shape": list(np_tensor.shape),
                "threshold_applied": threshold
            }
        
        # Return sparse representation
        values = np_tensor[indices]
        return {
            "sparse_values": values.tolist(),
            "sparse_indices": [idx.tolist() for idx in indices],
            "shape": list(np_tensor.shape),
            "threshold_applied": threshold
        }
    
    def _forward_hook(self, module: nn.Module, input: Any, output: Any) -> None:
        """Forward hook to capture layer activations with M-TRACE schema compliance."""
        if not self._enabled:
            return
        
        try:
            # EXTRACT MODEL METADATA (required by schema)
            model = self._get_model_ref()
            model_type = model.__class__.__name__ if model else "unknown"
            
            log_entry = {
                "model_metadata": {
                    "model_type": model_type.lower(),
                    "framework": "pytorch",
                    "timestamp": int(time.time() * 1000),  # Milliseconds for Parquet timestamp
                    "run_id": self.config.get("run_id", "unknown"),
                    "mode": self.config.get("mode", "development"),
                    "model_architecture": {
                        "num_layers": self._count_model_layers(model) if model else 1,
                        "layer_types": [type(m).__name__ for _, m in model.named_modules()] if model else ["linear"],
                        "connections": ["sequential"]  # Simplified for now
                    },
                    "hyperparameters": {
                        "learning_rate": 0.001,  # Default - will be overridden by actual training config
                        "batch_size": 32,
                        "optimizer": "adam",
                        "other_params": {}
                    },
                    "layer_metadata": {
                        "layer_type": module.__class__.__name__,
                        "activation_function": self._get_activation(module),
                        "num_parameters": sum(p.numel() for p in module.parameters()) if hasattr(module, 'parameters') else 0
                    }
                },
                "internal_states": {
                    "layer_name": self.layer_name,
                    "layer_index": self.layer_index,
                    "attention_weights": [],  # Empty for non-transformer layers
                    "feature_maps": [],
                    "node_splits": [],
                    "gradients": [],  # Populated in backward hook
                    "losses": 0.0,    # Will be populated during training loop
                    "feature_importance": [],
                    "decision_paths": []
                },
                "event_type": "forward"
            }
            
            # Capture output activations (apply sparse filtering)
            if isinstance(output, torch.Tensor):
                sparse_result = self._sparse_filter(output)
                log_entry["internal_states"]["output_activations"] = sparse_result.get("sparse_values", [])
                if "sparse_logging_metadata" not in log_entry:
                    log_entry["sparse_logging_metadata"] = {
                        "threshold_applied": sparse_result.get("threshold_applied", 0.0),
                        "top_k_values_logged": len(sparse_result.get("sparse_indices", [])),
                        "original_tensor_shape": sparse_result.get("shape", []),
                        "sparse_indices_count": len(sparse_result.get("sparse_indices", [])),
                        "sparse_type": sparse_result.get("sparse_type", "threshold"),
                        "sparse_indices": sparse_result.get("sparse_indices", [])
                    }

            # AFTER existing output capture logic in _forward_hook():

            # Special handling for transformer attention weights (Hugging Face)
            if "attention_weights" in self.config.get("custom_fields", []):
                # Case 1: Hugging Face output tuple (hidden_states, attentions)
                if isinstance(output, tuple) and len(output) > 1:
                    attentions = output[1]  # Second element is attention weights
                    
                    # Handle nested attention structures
                    if isinstance(attentions, (list, tuple)) and len(attentions) > 0:
                        # Take first layer's attention for logging (to avoid excessive data)
                        attentions = attentions[0]
                    
                    if isinstance(attentions, torch.Tensor):
                        # Extract attention for first batch item and first head
                        if attentions.ndim == 4:  # (batch, heads, seq_len, seq_len)
                            attn_weights = attentions[0, 0]  # First batch, first head
                        elif attentions.ndim == 3:  # (batch, seq_len, seq_len)
                            attn_weights = attentions[0]
                        else:
                            attn_weights = attentions
                        
                        log_entry["internal_states"]["attention_weights"] = self._sparse_filter(attn_weights)
                
                # Case 2: Direct attention weights in output dict (some custom models)
                elif isinstance(output, dict) and "attentions" in output:
                    attentions = output["attentions"]
                    if isinstance(attentions, torch.Tensor) and attentions.ndim == 4:
                        log_entry["internal_states"]["attention_weights"] = self._sparse_filter(attentions[0, 0])
            
            self._logs.append(log_entry)
            
        except Exception as e:
            logger.warning(f"Error in forward hook for {self.layer_name}: {e}", exc_info=True)

    
        # Add to PyTorchHook class in pytorch.py
    def _wrap_log_for_schema(self, flat_log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat hook logs into M-TRACE schema-compliant nested structure.
        
        Transforms:
        {'layer_name': 'linear1', 'timestamp': 12345, ...}
        Into:
        {
            'model_metadata': {...},
            'internal_states': {'layer_name': 'linear1', ...},
            'event_type': 'forward'
        }
        """
        # Extract model reference safely
        model = None
        if hasattr(self, '_model_ref') and self._model_ref is not None:
            model = self._model_ref()
        
        # Build model_metadata struct (required by schema)
        model_metadata = {
            "model_type": model.__class__.__name__.lower() if model else "unknown",
            "framework": "pytorch",
            "timestamp": int(flat_log.get("timestamp", time.time()) * 1000),  # ms for Parquet
            "run_id": self.config.get("run_id", "unknown"),
            "mode": self.config.get("mode", "development"),
            "model_architecture": {
                "num_layers": sum(1 for _ in model.modules()) - 1 if model else 1,
                "layer_types": [type(m).__name__ for _, m in model.named_modules()] if model else ["linear"],
                "connections": ["sequential"]
            },
            "hyperparameters": {
                "learning_rate": 0.001,  # Default - will be overridden by actual training config
                "batch_size": 32,
                "optimizer": "adam",
                "other_params": {}
            },
            "layer_metadata": {
                "layer_type": flat_log.get("layer_type", "linear"),
                "activation_function": flat_log.get("activation_function", "none"),
                "num_parameters": flat_log.get("num_parameters", 0)
            }
        }
        
        # Build internal_states struct (required by schema)
        internal_states = {
            "layer_name": flat_log.get("layer_name", "unknown"),
            "layer_index": flat_log.get("layer_index", 0),
            "attention_weights": flat_log.get("attention_weights", []),
            "feature_maps": flat_log.get("feature_maps", []),
            "node_splits": flat_log.get("node_splits", []),
            "gradients": flat_log.get("gradients", []),
            "losses": flat_log.get("losses", 0.0),
            "feature_importance": flat_log.get("feature_importance", []),
            "decision_paths": flat_log.get("decision_paths", [])
        }
        
        # Construct schema-compliant log
        return {
            "model_metadata": model_metadata,
            "internal_states": internal_states,
            "event_type": flat_log.get("event_type", "forward")
        }

    # Helper methods to add to PyTorchHook class
    def _get_model_ref(self):
        """Safely get model reference from weakref."""
        if hasattr(self, '_model_ref') and self._model_ref is not None:
            return self._model_ref()
        return None

    def _count_model_layers(self, model):
        """Count layers in model."""
        return sum(1 for _ in model.modules()) - 1  # Exclude root module

    def _get_activation(self, module):
        """Detect activation function type."""
        if isinstance(module, nn.ReLU):
            return "relu"
        elif isinstance(module, nn.GELU):
            return "gelu"
        elif isinstance(module, nn.Sigmoid):
            return "sigmoid"
        return "none"
    
    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """Backward hook to capture gradients (development mode only)."""
        if not self._enabled or self.config.get("mode", "development") != "development":
            return
        
        try:
            log_entry = {
                "timestamp": time.time(),
                "layer_name": self.layer_name,
                "layer_index": self.layer_index,
                "mode": "development",
                "event_type": "backward"
            }
            
            # Capture gradients
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                log_entry["grad_output"] = self._sparse_filter(grad_output[0])
            
            if grad_input and isinstance(grad_input[0], torch.Tensor):
                log_entry["grad_input"] = self._sparse_filter(grad_input[0])
            
            self._logs.append(log_entry)
            
        except Exception as e:
            logger.warning(f"Error in backward hook for {self.layer_name}: {e}")
    
    def attach(self, module: nn.Module) -> None:
        """Attach hooks to the module."""
        if not isinstance(module, nn.Module):
            raise ValueError(f"Expected nn.Module, got {type(module)}")
        
        # Always attach forward hook
        handle_fwd = module.register_forward_hook(self._forward_hook)
        self._hook_handles.append(handle_fwd)
        
        # Attach backward hook only in development mode
        if self.config.get("mode", "development") == "development":
            handle_bwd = module.register_full_backward_hook(self._backward_hook)
            self._hook_handles.append(handle_bwd)
        
        logger.debug(f"Attached hooks to layer: {self.layer_name}")
    
    def detach(self) -> None:
        """Remove all hooks from the module."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._enabled = False
        logger.debug(f"Detached hooks from layer: {self.layer_name}")
    
        # Replace existing get_logs() method in PyTorchHook class
    def get_logs(self) -> List[Dict]:
        """Return collected logs wrapped in M-TRACE schema structure."""
        wrapped_logs = []
        for log in self._logs:
            try:
                wrapped_logs.append(self._wrap_log_for_schema(log))
            except Exception as e:
                logger.warning(f"Failed to wrap log for schema: {e}")
        self._logs.clear()
        return wrapped_logs
    
    def clear_logs(self) -> None:
        """Clear internal log buffer."""
        self._logs.clear()


class PyTorchLoggingEngine:
    """Framework-specific engine for PyTorch models."""

    
    SUPPORTED_LAYERS = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU,
        nn.MultiheadAttention, nn.TransformerEncoderLayer
    )

    def _capture_model_attention(self, model_output: Any) -> Optional[Dict]:
        """
        Capture attention weights from transformer model outputs.
        
        Handles Hugging Face output formats:
        - BaseModelOutputWithAttentions (attentions attribute)
        - Tuple outputs (hidden_states, attentions)
        """
        try:
            # Case 1: Hugging Face output object with attentions attribute
            if hasattr(model_output, 'attentions') and model_output.attentions:
                # Extract first layer, first head, first batch item for visualization
                if isinstance(model_output.attentions[0], torch.Tensor):
                    attn = model_output.attentions[0][0, 0]  # [batch, head, seq, seq] → first head
                    return {
                        "attention_weights": attn.detach().cpu().numpy().flatten().tolist(),
                        "layer_name": "transformer_layer_0",
                        "layer_index": 0
                    }
            
            # Case 2: Tuple output (hidden_states, attentions)
            elif isinstance(model_output, tuple) and len(model_output) > 1:
                attentions = model_output[1]
                if isinstance(attentions, (list, tuple)) and attentions and isinstance(attentions[0], torch.Tensor):
                    attn = attentions[0][0, 0]
                    return {
                        "attention_weights": attn.detach().cpu().numpy().flatten().tolist(),
                        "layer_name": "transformer_layer_0",
                        "layer_index": 0
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Attention capture failed: {e}")
            return None
    
    def __init__(self, model: nn.Module, config: Any):
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")
        
        self.model = weakref.ref(model)  # Prevent reference cycles
        self.config = config
        self.hooks: List[PyTorchHook] = []
        self._enabled = False
    
    def enable(self) -> None:
        """Enable logging by attaching hooks to supported layers."""
        if self._enabled:
            logger.warning("Logging already enabled for this model")
            return
        
        model = self.model()
        if model is None:
            raise RuntimeError("Model reference lost")
        
        # Attach hooks to all supported layers
        layer_index = 0
        for name, module in model.named_modules():
            if isinstance(module, self.SUPPORTED_LAYERS):
                hook = PyTorchHook(
                    layer_name=name,
                    layer_index=layer_index,
                    config=self.config
                )
                hook.attach(module)
                self.hooks.append(hook)
                layer_index += 1
        
        self._enabled = True
        logger.info(f"Enabled logging for {len(self.hooks)} layers in PyTorch model")
    
    def disable(self) -> None:
        """Disable logging by detaching all hooks."""
        if not self._enabled:
            return
        
        for hook in self.hooks:
            hook.detach()
        
        self.hooks.clear()
        self._enabled = False
        logger.info("Disabled logging for PyTorch model")
    
    def collect_logs(self) -> List[Dict]:
        """Collect logs from all hooks."""
        if not self._enabled:
            return []
        
        all_logs = []
        for hook in self.hooks:
            all_logs.extend(hook.get_logs())
        
        return all_logs
    
    def is_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled
    

class AttentionCaptureWrapper(torch.nn.Module):
    """
    Minimal wrapper that captures attention weights from transformer model outputs.
    Avoids complex layer introspection - works at model output level.
    """
    def __init__(self, model, engine):
        super().__init__()
        self.model = model
        self.engine = engine
        self._mode = engine._mode
    
    def forward(self, *args, **kwargs):
        # Execute model
        output = self.model(*args, **kwargs)
        
        # Capture attention weights in development mode
        if self._mode == "development":
            attn_data = self.engine._capture_model_attention(output)
            if attn_data:
                # Create schema-compliant log entry
                log_entry = {
                    "model_metadata": {
                        "model_type": self.model.__class__.__name__.lower(),
                        "framework": "pytorch",
                        "timestamp": int(time.time() * 1000),
                        "run_id": self.engine.run_id,
                        "mode": self._mode,
                        "model_architecture": {
                            "num_layers": getattr(self.model, 'config', {}).get('num_hidden_layers', 12) 
                                          if hasattr(self.model, 'config') else 12,
                            "layer_types": ["transformer"],
                            "connections": ["sequential"]
                        },
                        "hyperparameters": {
                            "learning_rate": 0.0,
                            "batch_size": 1,
                            "optimizer": "n/a",
                            "other_params": {}
                        },
                        "layer_metadata": {
                            "layer_type": "transformer",
                            "activation_function": "gelu",
                            "num_parameters": sum(p.numel() for p in self.model.parameters())
                        }
                    },
                    "internal_states": {
                        "layer_name": attn_data["layer_name"],
                        "layer_index": attn_data["layer_index"],
                        "attention_weights": attn_data["attention_weights"],
                        "feature_maps": [],
                        "node_splits": [],
                        "gradients": [],
                        "losses": 0.0,
                        "feature_importance": [],
                        "decision_paths": []
                    },
                    "event_type": "forward"
                }
                self.engine._add_log(log_entry)
        
        return output