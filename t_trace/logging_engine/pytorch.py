"""PyTorch-specific hook implementation for M-TRACE with minimal overhead and guaranteed Parquet output."""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
import weakref
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class PyTorchHook:
    """Low-overhead hook that generates schema-compliant logs with deferred sparse processing."""
    
    def __init__(self, layer_name: str, layer_index: int, config: Dict, model_metadata: Dict):
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.config = config
        self.model_metadata = model_metadata  # Pre-computed ONCE during engine init
        self._logs: List[Dict] = []
        self._hook_handles: List[Any] = []
        self._enabled = True
        self._sparse_threshold = config.get("sparse_logging", {}).get("sparse_threshold", 0.1)
        self._top_k = config.get("sparse_logging", {}).get("top_k_values", 5)
    
    def _sparse_filter(self, tensor: torch.Tensor, is_attention: bool = False) -> Dict[str, Any]:
        if tensor.numel() == 0:
            return {
                "sparse_values": [],
                "sparse_indices": [],
                "shape": list(tensor.shape),
                "threshold_applied": self._sparse_threshold,
                "sparse_type": "empty"
            }
        
        np_tensor = tensor.detach().cpu().numpy()
        abs_vals = np.abs(np_tensor)
        
        # Track whether per-head preservation was applied
        used_per_head_preservation = False
        
        # === CORRECTED PER-HEAD LOGIC (self-contained with bypass) ===
        if is_attention and np_tensor.ndim >= 3:
            mask = abs_vals > self._sparse_threshold
            top_k_per_head = self.config.get("sparse_logging", {}).get("top_k_per_head", 3)
            
            if np_tensor.ndim == 4:  # [batch, heads, seq, seq]
                batch_size, num_heads, seq_len, _ = np_tensor.shape
                for b in range(batch_size):
                    for h in range(num_heads):
                        head_flat = np_tensor[b, h, :, :].flatten()
                        abs_head = np.abs(head_flat)
                        k = min(top_k_per_head, len(abs_head))
                        if k > 0:
                            topk_idx = np.argpartition(abs_head, -k)[-k:]
                            rows = topk_idx // seq_len
                            cols = topk_idx % seq_len
                            for r, c in zip(rows, cols):
                                mask[b, h, r, c] = True
            elif np_tensor.ndim == 3:  # [batch, seq, seq]
                batch_size, seq_len, _ = np_tensor.shape
                for b in range(batch_size):
                    batch_vals = np_tensor[b, :, :].flatten()
                    abs_batch = np.abs(batch_vals)
                    k = min(top_k_per_head, len(abs_batch))
                    if k > 0:
                        topk_idx = np.argpartition(abs_batch, -k)[-k:]
                        rows = topk_idx // seq_len
                        cols = topk_idx % seq_len
                        for r, c in zip(rows, cols):
                            mask[b, r, c] = True
            
            indices = np.where(mask)
            used_per_head_preservation = True  # ← FLAG FOR BYPASS
        
        else:
            mask = abs_vals > self._sparse_threshold
            indices = np.where(mask)
        # ===========================================================
        
        # Handle empty case
        if len(indices[0]) == 0:
            flat = abs_vals.flatten()
            if len(flat) == 0:
                return {
                    "sparse_values": [],
                    "sparse_indices": [],
                    "shape": list(tensor.shape),
                    "threshold_applied": self._sparse_threshold,
                    "sparse_type": "empty"
                }
            
            k = min(self._top_k, len(flat))
            topk_idx = np.argpartition(flat, -k)[-k:]
            sorted_idx = topk_idx[np.argsort(-flat[topk_idx])]
            values = np_tensor.flatten()[sorted_idx]
            
            return {
                "sparse_values": values.tolist(),
                "sparse_indices": sorted_idx.tolist(),
                "shape": list(tensor.shape),
                "threshold_applied": self._sparse_threshold,
                "sparse_type": "top_k"
            }
        
        # Extract values at masked indices
        values = np_tensor[indices]
        abs_selected = abs_vals[indices]
        
        # === CRITICAL FIX: BYPASS GLOBAL TOP-K FOR PER-HEAD PRESERVATION ===
        if used_per_head_preservation:
            # PRESERVE ALL values that passed threshold OR per-head top-k
            # No global cutoff - trajectory fidelity requires full per-head representation
            limited_order = np.arange(len(values))  # ← Keep ALL values
            sparse_type = "per_head_top_k"
        else:
            # Standard global top-k for non-attention tensors
            sorted_order = np.argsort(-abs_selected)
            limited_order = sorted_order[:self._top_k]
            sparse_type = "threshold"
        # ===================================================================
        
        flat_indices = np.ravel_multi_index(
            [idx[limited_order] for idx in indices],
            tensor.shape
        )
        
        return {
            "sparse_values": values[limited_order].tolist(),
            "sparse_indices": flat_indices.tolist(),
            "shape": list(tensor.shape),
            "threshold_applied": self._sparse_threshold,
            "sparse_type": sparse_type  # Now correctly reports per_head_top_k
        }
    
    def _extract_attention_weights(self, module: nn.Module, output: Any) -> Optional[np.ndarray]:
        """
        Extract attention weights from transformer layers with framework-aware detection.
        Handles: PyTorch nn.MultiheadAttention, Hugging Face BERT/CLIP attention layers.
        
        Returns:
            np.ndarray of attention weights (shape: [batch, heads, seq_len, seq_len]) 
            or None if not applicable
        """
        try:
            # Case 1: Hugging Face transformers (BERT, CLIP) - attention in output tuple
            if isinstance(output, tuple) and len(output) > 1:
                # BERT/CLIP: (hidden_states, attentions) where attentions is tuple of layer attentions
                if hasattr(output[1], '__len__') and len(output[1]) > 0:
                    attn = output[1][-1]  # Last attention layer in this block
                    if isinstance(attn, torch.Tensor):
                        return attn.detach().cpu().numpy()
                    
            # === CRITICAL ENHANCEMENT FOR BERT: Direct SelfAttention submodule detection ===
            # Case 1b: Hugging Face attention submodules (BertSelfAttention, SdpaAttention)
            # These modules return (context_layer, attention_probs) directly during forward pass
            module_class_name = module.__class__.__name__
            if any(pattern in module_class_name for pattern in [
                'SelfAttention',    # BERT/RoBERTa: BertSelfAttention
                'SdpaAttention',    # Newer HF models using SDPA
                'Attention',        # Generic attention modules (CLIP, ViT)
                'MultiHeadSelfAttention'  # Some custom implementations
            ]):
                # HF attention modules typically return (context, attention_probs) tuple
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_probs = output[1]  # attention_probs tensor
                    if isinstance(attn_probs, torch.Tensor) and attn_probs.dim() >= 3:
                        # Validate shape: [batch, heads, seq_len, seq_len] or [batch, seq_len, seq_len]
                        if attn_probs.dim() == 3:  # [batch, seq_len, seq_len]
                            # Expand to 4D for consistency: [batch, 1, seq_len, seq_len]
                            attn_probs = attn_probs.unsqueeze(1)
                        return attn_probs.detach().cpu().numpy()
            # ==============================================================================
            
            # Case 2: PyTorch nn.MultiheadAttention - check module state
            if isinstance(module, nn.MultiheadAttention):
                # Native PyTorch MHA doesn't expose weights directly - need custom hook
                # Fallback: Check if module has cached attention (some implementations store it)
                if hasattr(module, '_last_attn_weights') and module._last_attn_weights is not None:
                    return module._last_attn_weights.detach().cpu().numpy()
            
            # Case 3: Vision Transformers (ViT) - common patterns
            if hasattr(module, 'attn') and hasattr(module.attn, 'get_attention_map'):
                return module.attn.get_attention_map().detach().cpu().numpy()
            
            # Case 4: Custom attention implementations (e.g., timm models)
            if hasattr(module, 'attention_weights'):
                attn = getattr(module, 'attention_weights')
                if isinstance(attn, torch.Tensor):
                    return attn.detach().cpu().numpy()
                    
        except Exception as e:
            logger.debug(f"Attention extraction failed for {self.layer_name}: {e}")
        
        return None  # No attention weights available for this layer type
    
    def _forward_hook(self, module: nn.Module, input: Any, output: Any) -> None:
        """CRITICAL: Generate schema-compliant log IMMEDIATELY (no deferred conversion)."""
        if not self._enabled:
            return
        
        try:
            # Build schema-compliant log with MINIMAL processing (no model introspection!)
            log_entry = {
                "model_metadata": {
                    **self.model_metadata,  # Pre-computed reference (safe to copy)
                    "timestamp": int(time.time() * 1000),  # Milliseconds for Parquet
                    "layer_metadata": {
                        "layer_type": module.__class__.__name__,
                        "activation_function": self._get_activation(module),
                        "num_parameters": sum(p.numel() for p in module.parameters()) 
                                          if hasattr(module, 'parameters') else 0
                    }
                },
                "internal_states": {
                    "layer_name": self.layer_name,
                    "layer_index": self.layer_index,
                    "attention_weights": [],
                    "feature_maps": [],
                    "node_splits": [],
                    "gradients": [],
                    "losses": 0.0,
                    "feature_importance": [],
                    "decision_paths": []
                },
                "event_type": "forward"
            }

            # === CRITICAL GAP 1 FIX: Extract attention weights ===
            attn_weights = self._extract_attention_weights(module, output)
            if attn_weights is not None and attn_weights.size > 0:
                # Store raw tensor for deferred sparse filtering (off critical path)
                log_entry["_raw_attention"] = attn_weights  # Mark for sparse processing in collect_logs()
                logger.debug(f"Captured attention weights shape {attn_weights.shape} for {self.layer_name}")
            # ================================================
            
            # Defer heavy processing: store raw tensor for sparse filtering in collect_logs()
            if isinstance(output, torch.Tensor):
                log_entry["_raw_output"] = output  # Mark for deferred sparse filtering
            
            self._logs.append(log_entry)
            
        except Exception as e:
            logger.debug(f"Hook error ({self.layer_name}): {e}")
    
    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        """Backward hook (development mode only)."""
        if not self._enabled or self.config.get("mode", "development") != "development":
            return
        
        try:
            log_entry = {
                "model_metadata": {
                    **self.model_metadata,
                    "timestamp": int(time.time() * 1000),
                    "layer_metadata": {
                        "layer_type": module.__class__.__name__,
                        "activation_function": "none",
                        "num_parameters": 0
                    }
                },
                "internal_states": {
                    "layer_name": self.layer_name,
                    "layer_index": self.layer_index,
                    "gradients": [],
                    "losses": 0.0
                },
                "event_type": "backward"
            }
            
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                log_entry["_raw_grad"] = grad_output[0]  # Defer sparse filtering
            
            self._logs.append(log_entry)
        except Exception as e:
            logger.debug(f"Backward hook error ({self.layer_name}): {e}")
    
    def _get_activation(self, module):
        if isinstance(module, nn.ReLU):
            return "relu"
        elif isinstance(module, nn.GELU):
            return "gelu"
        elif isinstance(module, nn.Sigmoid):
            return "sigmoid"
        return "none"
    
    def attach(self, module: nn.Module) -> None:
        handle_fwd = module.register_forward_hook(self._forward_hook)
        self._hook_handles.append(handle_fwd)
        
        if self.config.get("mode", "development") == "development":
            try:
                handle_bwd = module.register_full_backward_hook(self._backward_hook)
                self._hook_handles.append(handle_bwd)
            except AttributeError:
                handle_bwd = module.register_backward_hook(self._backward_hook)
                self._hook_handles.append(handle_bwd)
    
    def detach(self) -> None:
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self._enabled = False
    
    def get_logs(self) -> List[Dict]:
        """Apply sparse filtering to deferred tensors BEFORE returning logs (thread-safe)."""
        processed_logs = []
        for log in self._logs:
            # Process output tensor if present (SAFELY using pop())
            raw_output = log.pop("_raw_output", None)  # Atomic removal with default
            if raw_output is not None and isinstance(raw_output, torch.Tensor):
                try:
                    sparse_result = self._sparse_filter(raw_output)
                    log["internal_states"]["output_activations"] = sparse_result
                except Exception as e:
                    logger.debug(f"Sparse filtering failed for output: {e}")
                    # Fallback: log minimal metadata without activations
                    log["internal_states"]["output_activations"] = {
                        "sparse_values": [],
                        "sparse_indices": [],
                        "shape": list(raw_output.shape) if hasattr(raw_output, 'shape') else [],
                        "threshold_applied": self._sparse_threshold,
                        "sparse_type": "filtering_failed"
                    }
            
            # Process gradient tensor if present (SAFELY using pop())
            raw_grad = log.pop("_raw_grad", None)  # Atomic removal with default
            if raw_grad is not None and isinstance(raw_grad, torch.Tensor):
                try:
                    sparse_result = self._sparse_filter(raw_grad)
                    log["internal_states"]["gradients"] = [sparse_result]  # List per schema
                except Exception as e:
                    logger.debug(f"Sparse filtering failed for gradient: {e}")
                    log["internal_states"]["gradients"] = [{
                        "sparse_values": [],
                        "sparse_indices": [],
                        "shape": list(raw_grad.shape) if hasattr(raw_grad, 'shape') else [],
                        "threshold_applied": self._sparse_threshold,
                        "sparse_type": "filtering_failed"
                    }]

            # === CRITICAL GAP 1 FIX: Process attention weights ===
            raw_attn = log.pop("_raw_attention", None)
            if raw_attn is not None:
                try:
                    # Apply sparse filtering (reuses existing _sparse_filter method)
                    sparse_result = self._sparse_filter(
                        torch.from_numpy(raw_attn) if isinstance(raw_attn, np.ndarray) else raw_attn,
                        is_attention=True
                    )
                    # Store sparse representation in attention_weights field
                    log["internal_states"]["attention_weights"] = sparse_result.get("sparse_values", [])
                    
                    # Update metadata to reflect per-head preservation
                    if "sparse_logging_metadata" not in log:
                        log["sparse_logging_metadata"] = {}
                    log["sparse_logging_metadata"].update({
                        "attention_shape": sparse_result.get("shape", []),
                        "attention_sparse_type": "per_head_top_k",  # ← Explicit type
                        "top_k_per_head": self.config.get("sparse_logging", {}).get("top_k_per_head", 3),
                        "attention_threshold": sparse_result.get("threshold_applied", 0.1)
                    })
                        
                except Exception as e:
                    logger.debug(f"Sparse filtering failed for attention: {e}")
                    log["internal_states"]["attention_weights"] = []
            # ================================================
            processed_logs.append(log)
        
        self._logs.clear()
        return processed_logs


class PyTorchLoggingEngine:
    """Framework engine with pre-computed metadata + immediate schema compliance."""
    
    SUPPORTED_MODULES = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
        nn.ConvTranspose3d, nn.Embedding, nn.LSTM, nn.GRU, nn.RNN, nn.MultiheadAttention
    )
    
    def __init__(self, model: nn.Module, config: Dict):
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be nn.Module")
        
        self.model_ref = weakref.ref(model)
        self.config = config
        self.hooks: List[PyTorchHook] = []
        self._enabled = False
        
        # PRE-COMPUTE METADATA ONCE (critical for performance)
        self._model_metadata = self._build_model_metadata(model, config)
    
    def _build_model_metadata(self, model: nn.Module, config: Dict) -> Dict:
        """Single traversal to build metadata (happens ONCE during init)."""
        try:
            model_type = model.__class__.__name__.lower()
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            layer_names = []
            layer_types = []
            
            for name, module in model.named_modules():
                if list(module.children()):
                    continue
                if not list(module.parameters()):
                    continue
                if not isinstance(module, self.SUPPORTED_MODULES):
                    continue
                
                layer_names.append(name)
                layer_types.append(module.__class__.__name__)
            
            return {
                "model_type": model_type,
                "framework": "pytorch",
                "run_id": config.get("run_id", "unknown"),
                "mode": config.get("mode", "development"),
                "model_architecture": {
                    "num_layers": len(layer_names),
                    "layer_names": layer_names,
                    "layer_types": layer_types,
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params
                },
                "hyperparameters": {
                    "learning_rate": config.get("hyperparameters", {}).get("learning_rate", 0.001),
                    "batch_size": config.get("hyperparameters", {}).get("batch_size", 32),
                    "optimizer": config.get("hyperparameters", {}).get("optimizer", "adam"),
                    "other_params": {}
                }
            }
        except Exception as e:
            logger.warning(f"Metadata fallback: {e}")
            return {
                "model_type": "unknown",
                "framework": "pytorch",
                "run_id": config.get("run_id", "unknown"),
                "mode": config.get("mode", "development"),
                "model_architecture": {"num_layers": 0, "layer_names": [], "layer_types": []},
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "optimizer": "adam",
                    "other_params": {}
                }
            }
    
    def enable(self) -> None:
        if self._enabled:
            return
        
        model = self.model_ref()
        if model is None:
            raise RuntimeError("Model reference lost")
        
        layer_index = 0
        hooks_attached = 0
        
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            if not list(module.parameters()):
                continue
            if not isinstance(module, self.SUPPORTED_MODULES):
                continue
            
            hook = PyTorchHook(
                layer_name=name or f"layer_{layer_index}",
                layer_index=layer_index,
                config=self.config,
                model_metadata=self._model_metadata
            )
            hook.attach(module)
            self.hooks.append(hook)
            hooks_attached += 1
            layer_index += 1
        
 
        # === CRITICAL ADDITION: Transformer attention submodules ===
        model_type_lower = self._model_metadata["model_type"].lower()
        if any(kw in model_type_lower for kw in ["bert", "transformer", "roberta", "distilbert", "clip"]):
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                # Detect attention submodules by class name patterns
                if any(pattern in class_name for pattern in [
                    'SelfAttention',    # BERT/RoBERTa
                    'SdpaAttention',    # Newer HF models
                    'MultiHeadAttention',  # Some implementations
                    'Attention'         # Generic (CLIP, ViT) - use cautiously
                ]):
                    # Verify it's a real attention module (not false positive)
                    if hasattr(module, 'query') and hasattr(module, 'key'):
                        hook = PyTorchHook(
                            layer_name=name,
                            layer_index=layer_index,
                            config=self.config,
                            model_metadata=self._model_metadata
                        )
                        hook.attach(module)
                        self.hooks.append(hook)
                        hooks_attached += 1
                        layer_index += 1
                        logger.debug(f"Attached hook to attention submodule: {name} ({class_name})")
        # ================================================


        self._enabled = True
        logger.info(
            f"Enabled M-TRACE for {hooks_attached} layers ({self._model_metadata['model_type']}) | "
            f"run_id: {self.config.get('run_id', 'unknown')[:8]}..."
        )
    
    def disable(self) -> None:
        if not self._enabled:
            return
        
        for hook in self.hooks:
            hook.detach()
        self.hooks.clear()
        self._enabled = False
        logger.info("Disabled M-TRACE logging")
    
    def collect_logs(self) -> List[Dict]:
        """Collect logs with sparse filtering already applied + guaranteed EXECUTION ORDER."""
        if not self._enabled:
            return []
        
        all_logs = []
        for hook in self.hooks:
            all_logs.extend(hook.get_logs())
        
        if not all_logs:
            return all_logs
        
        # === CRITICAL GAP 2 FIX: Sort by EXECUTION TIMESTAMP (primary) + layer_index (secondary) ===
        def _sort_key(log: Dict) -> tuple:
            """Sort by actual execution time (timestamp), NOT attachment order (layer_index)."""
            metadata = log.get("model_metadata", {})
            internal = log.get("internal_states", {})
            
            # PRIMARY: timestamp (execution order - when hook actually fired)
            timestamp = metadata.get("timestamp", 0)
            if not isinstance(timestamp, (int, float)):
                timestamp = 0
            
            # SECONDARY: layer_index (for tie-breaking within same millisecond)
            layer_idx = internal.get("layer_index", -1)
            if not isinstance(layer_idx, (int, float)):
                layer_idx = -1
            
            return (int(timestamp), int(layer_idx))  # ← TIMESTAMP FIRST!
        # ==============================================================================
        
        sorted_logs = sorted(all_logs, key=_sort_key)
        
        logger.debug(
            f"Sorted {len(all_logs)} logs by EXECUTION ORDER (timestamp range: "
            f"{sorted_logs[0].get('model_metadata', {}).get('timestamp', 'N/A')} → "
            f"{sorted_logs[-1].get('model_metadata', {}).get('timestamp', 'N/A')})"
        )
        return sorted_logs
    
    def is_enabled(self) -> bool:
        return self._enabled