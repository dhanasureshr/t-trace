"""scikit-learn-specific logging implementation with M-TRACE schema compliance."""
import numpy as np
import time
import logging
from typing import Any, Dict, List, Optional
from weakref import ref as weakref_ref

logger = logging.getLogger(__name__)

class SklearnLoggingEngine:
    """
    Framework-specific engine for scikit-learn estimators with M-TRACE schema compliance.
    CRITICAL FIX: All logs wrapped in nested model_metadata/internal_states structure
    per Section 3.1.2 of project m trace.pdf specification.
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        if not self._is_sklearn_model(model):
            raise ValueError(f"Model must be a scikit-learn BaseEstimator, got {type(model)}")
        
        self.model_ref = weakref_ref(model)
        self.config = config
        self._mode = config.get("mode", "production")
        self._logs: List[Dict] = []
        self.wrapped_model: Optional[Any] = None
        self._enabled = False
        self._run_id = config.get("run_id", "unknown")
        
        # Pre-compute BASE model metadata ONCE (critical for performance)
        self._base_model_metadata = self._extract_base_model_metadata(model)
        logger.debug(f"Initialized SklearnLoggingEngine for {model.__class__.__name__}")
    
    @staticmethod
    def _is_sklearn_model(model: Any) -> bool:
        try:
            from sklearn.base import BaseEstimator
            return isinstance(model, BaseEstimator)
        except ImportError:
            return False
    
    def _extract_base_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Pre-compute model metadata ONCE during init (not per-log) for performance."""
        try:
            n_params = self._count_parameters(model)
            return {
                "model_type": model.__class__.__name__.lower(),
                "framework": "sklearn",
                "num_parameters": n_params,
                "num_trainable_parameters": n_params,
                "input_shape": None,  # sklearn doesn't have fixed input shape
                "output_shape": None,
                "layer_count": 1,  # Estimator-level (not per-layer)
                "layer_types": [model.__class__.__name__],
                "connections": ["estimator"]
            }
        except Exception as e:
            logger.warning(f"Failed to extract base model metadata: {e}")
            return {
                "model_type": model.__class__.__name__.lower(),
                "framework": "sklearn",
                "num_parameters": 0,
                "num_trainable_parameters": 0,
                "input_shape": None,
                "output_shape": None,
                "layer_count": 1,
                "layer_types": [model.__class__.__name__],
                "connections": ["estimator"]
            }
    
    def _count_parameters(self, model: Any) -> int:
        """Count model parameters for metadata."""
        try:
            if hasattr(model, 'coef_') and model.coef_ is not None:
                return model.coef_.size
            elif hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                return len(model.feature_importances_)
            return 0
        except Exception:
            return 0
    
    def _sparse_filter(self, array: np.ndarray, field_name: str = "") -> Dict[str, Any]:
        """Apply sparse logging to high-dimensional arrays (Section 3.1.4)."""
        if not self.config.get("sparse_logging", {}).get("enabled", True):
            return {"full_array": array.tolist()}
        
        if array.size == 0:
            return {"empty": True, "shape": list(array.shape)}
        
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
            except:
                return {"raw_value": str(array)[:100]}
        
        # For 1D arrays (feature importances, coefficients)
        if array.ndim == 1:
            abs_values = np.abs(array)
            threshold = self.config.get("sparse_logging", {}).get("sparse_threshold", 0.1)
            top_k = self.config.get("sparse_logging", {}).get("top_k_values", 5)
            
            mask = abs_values > threshold
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                if len(array) <= top_k:
                    return {
                        "sparse_values": array.tolist(),
                        "sparse_indices": np.arange(len(array)).tolist(),
                        "shape": [len(array)],
                        "sparse_type": "all_values"
                    }
                top_indices = np.argpartition(abs_values, -top_k)[-top_k:]
                sorted_indices = top_indices[np.argsort(-abs_values[top_indices])]
                return {
                    "sparse_values": array[sorted_indices].tolist(),
                    "sparse_indices": sorted_indices.tolist(),
                    "shape": [len(array)],
                    "threshold_applied": float(threshold),
                    "sparse_type": "top_k"
                }
            
            sorted_indices = indices[np.argsort(-abs_values[indices])]
            return {
                "sparse_values": array[sorted_indices].tolist(),
                "sparse_indices": sorted_indices.tolist(),
                "shape": [len(array)],
                "threshold_applied": float(threshold),
                "sparse_type": "threshold"
            }
        
        # Fallback for higher dimensions
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sparse_type": "unsupported_ndim"
        }
    
    def _log_fit_state(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Capture internal state AFTER fit() with M-TRACE schema compliance."""
        if self._mode != "development":
            return
        
        timestamp_ms = int(time.time() * 1000)
        
        # Build schema-compliant log entry (CRITICAL FIX)
        log_entry = {
            "model_metadata": {
                **self._base_model_metadata,
                "timestamp": timestamp_ms,
                "run_id": self._run_id,
                "mode": self._mode,
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": X.shape[0],
                    "optimizer": "n/a",
                    "other_params": {}
                },
                "layer_metadata": {
                    "layer_type": model.__class__.__name__,
                    "activation_function": "n/a",
                    "num_parameters": self._count_parameters(model)
                }
            },
            "internal_states": {  # ← REQUIRED NESTED STRUCT
                "layer_name": model.__class__.__name__,  # ← REQUIRED FIELD
                "layer_index": 0,                         # ← REQUIRED FIELD (estimator-level)
                "attention_weights": [],
                "feature_maps": [],
                "node_splits": [],
                "gradients": [],
                "losses": 0.0,                            # ← REQUIRED FIELD (0.0 for sklearn)
                "feature_importance": [],
                "decision_paths": [],
                "output_activations": []
            },
            "event_type": "fit"
        }
        
        # Add estimator-specific internals WITH sparse filtering
        try:

            # WITH THIS (schema-compliant implementation):
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                sparse_result = self._sparse_filter(model.feature_importances_, "feature_importances")
                
                # CRITICAL FIX 1: Store ONLY sparse_values in feature_importance (flat list)
                log_entry["internal_states"]["feature_importance"] = sparse_result.get("sparse_values", [])
                
                # CRITICAL FIX 2: Store sparse metadata in top-level sparse_logging_metadata
                if "sparse_logging_metadata" not in log_entry:
                    log_entry["sparse_logging_metadata"] = {
                        "threshold_applied": float(sparse_result.get("threshold_applied", 0.0)),
                        "top_k_values_logged": int(len(sparse_result.get("sparse_indices", []))),
                        "original_tensor_shape": [int(x) for x in sparse_result.get("shape", [])],
                        "sparse_indices_count": int(len(sparse_result.get("sparse_indices", []))),
                        "sparse_type": str(sparse_result.get("sparse_type", "threshold")),
                        "sparse_indices": [int(x) for x in sparse_result.get("sparse_indices", [])]  # FLAT indices
                    }
            
            if hasattr(model, 'coef_') and model.coef_ is not None:
                sparse_result = self._sparse_filter(model.coef_, "coef")
                log_entry["internal_states"]["coefficients"] = sparse_result.get("sparse_values", [])
                # Append to existing sparse_logging_metadata if already created
                if "sparse_logging_metadata" in log_entry:
                    log_entry["sparse_logging_metadata"]["coefficients_shape"] = sparse_result.get("shape", [])
            
            if hasattr(model, 'intercept_'):
                log_entry["internal_states"]["intercept"] = (
                    model.intercept_.tolist() if hasattr(model.intercept_, 'tolist') 
                    else float(model.intercept_)
                )
            
            # Tree structure metadata
            if hasattr(model, 'tree_'):
                log_entry["internal_states"]["tree_metadata"] = {
                    "max_depth": int(model.tree_.max_depth),
                    "n_leaves": int(model.tree_.n_leaves),
                    "n_nodes": int(model.tree_.node_count),
                    "n_features": int(model.tree_.n_features)
                }
            
            # Ensemble metadata
            if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                log_entry["internal_states"]["ensemble_metadata"] = {
                    "n_estimators": len(model.estimators_),
                    "base_estimator_type": model.estimators_[0].__class__.__name__
                }
        
        except Exception as e:
            logger.warning(f"Error capturing fit state for {model.__class__.__name__}: {e}")
        
        self._logs.append(log_entry)
        logger.debug(f"Logged fit state for {model.__class__.__name__}")
    
    def _log_prediction(
        self, 
        model: Any, 
        X: np.ndarray, 
        predictions: np.ndarray, 
        decision_info: Optional[Any] = None
    ) -> None:
        """Log prediction with M-TRACE schema compliance."""
        timestamp_ms = int(time.time() * 1000)
        
        # Build schema-compliant log entry (CRITICAL FIX)
        log_entry = {
            "model_metadata": {
                **self._base_model_metadata,
                "timestamp": timestamp_ms,
                "run_id": self._run_id,
                "mode": self._mode,
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": X.shape[0],
                    "optimizer": "n/a",
                    "other_params": {}
                },
                "layer_metadata": {
                    "layer_type": model.__class__.__name__,
                    "activation_function": "n/a",
                    "num_parameters": self._count_parameters(model)
                }
            },
            "internal_states": {  # ← REQUIRED NESTED STRUCT
                "layer_name": model.__class__.__name__,  # ← REQUIRED FIELD
                "layer_index": 0,                         # ← REQUIRED FIELD
                "attention_weights": [],
                "feature_maps": [],
                "node_splits": [],
                "gradients": [],
                "losses": 0.0,                            # ← REQUIRED FIELD
                "feature_importance": [],
                "decision_paths": [],
                "output_activations": []
            },
            "event_type": "predict" if hasattr(model, 'predict') else "transform"
        }
        
        # Add prediction-specific details
        try:
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X[:5])  # Sample first 5 for efficiency
                    log_entry["internal_states"]["sample_confidence"] = proba[:5].tolist()
                except Exception:
                    pass
            
            if hasattr(model, 'decision_function'):
                try:
                    distances = model.decision_function(X[:5])
                    log_entry["internal_states"]["sample_distances"] = distances[:5].tolist()
                except Exception:
                    pass
        
        except Exception as e:
            logger.warning(f"Error capturing prediction details: {e}")
        
        self._logs.append(log_entry)
        logger.debug(f"Logged prediction for {model.__class__.__name__}")
    
    def _create_wrapped_estimator_class(self, base_class: type) -> type:
        """Dynamically create wrapped estimator with schema-compliant logging."""
        engine_ref = weakref_ref(self)
        
        class WrappedEstimator(base_class):
            _mtrace_wrapped = True
            
            def fit(self, X, y=None, **kwargs):
                # Call original fit
                result = super().fit(X, y, **kwargs)
                
                # Log fit state (schema-compliant)
                engine = engine_ref()
                if engine and engine._enabled:
                    try:
                        engine._log_fit_state(self, X, y)
                    except Exception as e:
                        logger.warning(f"Fit logging failed: {e}")
                return result
            
            def predict(self, X):
                # Call original predict
                predictions = super().predict(X)
                
                # Log prediction (schema-compliant)
                engine = engine_ref()
                if engine and engine._enabled:
                    try:
                        engine._log_prediction(self, X, predictions, None)
                    except Exception as e:
                        logger.warning(f"Prediction logging failed: {e}")
                return predictions
            
            def transform(self, X):
                result = super().transform(X)
                engine = engine_ref()
                if engine and engine._enabled:
                    try:
                        engine._log_prediction(self, X, result, None)
                    except Exception as e:
                        logger.warning(f"Transform logging failed: {e}")
                return result
        
        WrappedEstimator.__name__ = f"MTrace{base_class.__name__}"
        WrappedEstimator.__qualname__ = f"MTrace{base_class.__qualname__}"
        WrappedEstimator.__module__ = base_class.__module__
        return WrappedEstimator
    
    def enable(self) -> None:
        """Enable logging with schema-compliant wrapped estimator."""
        if self._enabled:
            logger.warning("scikit-learn logging already enabled")
            return
        
        model = self.model_ref()
        if model is None:
            raise RuntimeError("Model reference lost - cannot enable logging")
        
        try:
            WrappedClass = self._create_wrapped_estimator_class(model.__class__)
            from sklearn.base import clone
            self.wrapped_model = clone(model)
            self.wrapped_model.__class__ = WrappedClass
            
            self._enabled = True
            logger.info(
                f"Enabled scikit-learn logging in {self._mode} mode for "
                f"{model.__class__.__name__} (run_id: {self._run_id})"
            )
        except Exception as e:
            logger.error(f"Failed to wrap scikit-learn estimator: {e}")
            raise RuntimeError(f"scikit-learn instrumentation failed: {e}") from e
    
    def disable(self) -> None:
        """Disable logging and cleanup resources."""
        if not self._enabled:
            return
        self.wrapped_model = None
        self._enabled = False
        logger.info("Disabled scikit-learn logging")
    
    def get_wrapped_model(self) -> Any:
        """Get the schema-compliant wrapped estimator."""
        if not self._enabled:
            raise RuntimeError("Logging not enabled. Call enable() before getting wrapped model.")
        if self.wrapped_model is None:
            raise RuntimeError("Wrapped model not available - instrumentation failed")
        return self.wrapped_model
    
    def collect_logs(self) -> List[Dict]:
        """Return schema-compliant logs."""
        logs = self._logs.copy()
        self._logs.clear()
        return logs
    
    def is_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled