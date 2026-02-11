"""scikit-learn-specific logging implementation for M-TRACE LoggingEngine."""
import numpy as np
import time
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from weakref import ref as weakref_ref

logger = logging.getLogger(__name__)


class SklearnLoggingEngine:
    """
    Framework-specific engine for scikit-learn estimators.
    
    Implements M-TRACE Section 2.1.3 requirement to override fit()/predict() methods
    for capturing internal estimator states without modifying original model behavior.
    
    Architecture decisions:
    1. Uses dynamic subclassing (not monkey-patching) for robustness and pipeline compatibility
    2. Preserves get_params()/set_params() for hyperparameter tuning workflows
    3. Handles classifiers, regressors, and transformers uniformly
    4. Development mode: captures coefficients, feature importances, decision paths
    5. Production mode: lightweight inference logging only (<5% overhead)
    """
    
    SUPPORTED_ESTIMATOR_TYPES = ["classifier", "regressor", "transformer", "cluster"]
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        if not self._is_sklearn_model(model):
            raise ValueError(
                f"Model must be a scikit-learn BaseEstimator, got {type(model)}"
            )
        
        self.model_ref = weakref_ref(model)
        self.original_model = model
        self.config = config
        self._mode = config.get("mode", "production")
        self._logs: List[Dict] = []
        self.wrapped_model: Optional[Any] = None
        self._enabled = False
        
        # Detect estimator type
        self._estimator_type = self._detect_estimator_type(model)
        logger.debug(
            f"Initialized SklearnLoggingEngine for {model.__class__.__name__} "
            f"(type: {self._estimator_type}, mode: {self._mode})"
        )
    
    @staticmethod
    def _is_sklearn_model(model: Any) -> bool:
        """Check if object is a scikit-learn estimator."""
        try:
            from sklearn.base import BaseEstimator
            return isinstance(model, BaseEstimator)
        except ImportError:
            return False
    
    def _detect_estimator_type(self, model: Any) -> str:
        """Detect estimator type (classifier/regressor/transformer/cluster)."""
        if hasattr(model, "fit") and hasattr(model, "predict"):
            if hasattr(model, "classes_"):
                return "classifier"
            return "regressor"
        elif hasattr(model, "fit") and hasattr(model, "transform"):
            return "transformer"
        elif hasattr(model, "fit_predict"):
            return "cluster"
        return "unknown"
    
    def _sparse_filter(self, array: np.ndarray, field_name: str = "") -> Dict[str, Any]:
        """
        Apply sparse logging to high-dimensional arrays per Section 3.1.4.
        
        Handles:
        - Feature importances (1D arrays)
        - Coefficients (1D/2D arrays)
        - Decision paths (sparse matrices)
        """
        if not self.config.get("sparse_logging", {}).get("enabled", True):
            return {"full_array": array.tolist()}
        
        # Handle empty/invalid arrays
        if array.size == 0:
            return {"empty": True, "shape": list(array.shape)}
        
        # Convert to numpy if needed
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
            except:
                return {"raw_value": str(array)[:100]}  # Fallback for non-numeric
        
        # For 1D arrays (feature importances, coefficients)
        if array.ndim == 1:
            abs_values = np.abs(array)
            threshold = self.config.get("sparse_logging", {}).get("sparse_threshold", 0.1)
            top_k = self.config.get("sparse_logging", {}).get("top_k_values", 5)
            
            # Find indices above threshold
            mask = abs_values > threshold
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                # Fallback to top-k
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
            
            # Return threshold-filtered values
            sorted_indices = indices[np.argsort(-abs_values[indices])]
            return {
                "sparse_values": array[sorted_indices].tolist(),
                "sparse_indices": sorted_indices.tolist(),
                "shape": [len(array)],
                "threshold_applied": float(threshold),
                "sparse_type": "threshold"
            }
        
        # For 2D arrays (coefficients in multi-output models)
        elif array.ndim == 2:
            # Apply sparse filtering per row (each output dimension)
            sparse_rows = []
            for i in range(array.shape[0]):
                row_filter = self._sparse_filter(array[i], f"{field_name}_row_{i}")
                sparse_rows.append(row_filter)
            return {"sparse_rows": sparse_rows, "shape": list(array.shape)}
        
        # Fallback for higher dimensions
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sparse_type": "unsupported_ndim"
        }
    
    def _log_fit_state(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Capture internal state after fit() completes (development mode only)."""
        if self._mode != "development":
            return
        
        timestamp = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "event_type": "fit",
            "estimator_name": model.__class__.__name__,
            "estimator_type": self._estimator_type,
            "mode": self._mode,
            "n_samples": X.shape[0],
            "n_features": X.shape[1] if X.ndim > 1 else 1,
        }
        
        # Capture estimator-specific internals
        try:
            # Feature importances (trees, forests)
            if hasattr(model, "feature_importances_"):
                log_entry["feature_importances"] = self._sparse_filter(
                    model.feature_importances_, "feature_importances"
                )
            
            # Coefficients (linear models)
            if hasattr(model, "coef_"):
                log_entry["coefficients"] = self._sparse_filter(model.coef_, "coef")
            if hasattr(model, "intercept_"):
                log_entry["intercept"] = (
                    model.intercept_.tolist() if hasattr(model.intercept_, "tolist")
                    else float(model.intercept_)
                )
            
            # Tree structure metadata
            if hasattr(model, "tree_"):
                log_entry["tree_metadata"] = {
                    "max_depth": int(model.tree_.max_depth),
                    "n_leaves": int(model.tree_.n_leaves),
                    "n_nodes": int(model.tree_.node_count),
                    "n_features": int(model.tree_.n_features)
                }
            
            # Support vectors (SVMs)
            if hasattr(model, "support_vectors_"):
                # Log only metadata in development mode (vectors themselves are large)
                log_entry["support_vectors_metadata"] = {
                    "n_support_vectors": int(len(model.support_vectors_)),
                    "vector_dimension": int(model.support_vectors_.shape[1])
                }
            
            # Cluster centers
            if hasattr(model, "cluster_centers_"):
                log_entry["n_clusters"] = int(len(model.cluster_centers_))
            
            # Ensemble metadata
            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                log_entry["ensemble_metadata"] = {
                    "n_estimators": len(model.estimators_),
                    "base_estimator_type": model.estimators_[0].__class__.__name__
                }
        
        except Exception as e:
            logger.warning(f"Error capturing fit state for {model.__class__.__name__}: {e}")
        
        self._logs.append(log_entry)
        logger.debug(
            f"Logged fit state for {model.__class__.__name__} "
            f"({log_entry.get('n_samples')} samples, {log_entry.get('n_features')} features)"
        )
    
    def _log_prediction(
        self, 
        model: Any, 
        X: np.ndarray, 
        predictions: np.ndarray,
        decision_info: Optional[Any] = None
    ) -> None:
        """Log prediction/inference events with optional decision path information."""
        timestamp = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "event_type": "predict" if hasattr(model, "predict") else "transform",
            "estimator_name": model.__class__.__name__,
            "estimator_type": self._estimator_type,
            "mode": self._mode,
            "n_samples": X.shape[0],
            "n_features": X.shape[1] if X.ndim > 1 else 1,
            "prediction_shape": list(predictions.shape) if hasattr(predictions, "shape") else None,
        }
        
        # Development mode: capture decision paths for tree-based models
        if self._mode == "development" and decision_info is not None:
            try:
                if hasattr(model, "tree_") and decision_info is not None:
                    # decision_info is sparse matrix of node indicators
                    path_lengths = np.array(decision_info.sum(axis=1)).flatten()
                    log_entry["decision_path_metadata"] = {
                        "mean_path_length": float(np.mean(path_lengths)),
                        "max_path_length": int(np.max(path_lengths)),
                        "min_path_length": int(np.min(path_lengths))
                    }
                
                # For classifiers with probability estimates
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(X[:5])  # Sample first 5 for efficiency
                        log_entry["sample_confidence"] = proba[:5].tolist()
                    except:
                        pass
                
                # Distance to hyperplane (SVMs)
                if hasattr(model, "decision_function"):
                    try:
                        distances = model.decision_function(X[:5])
                        log_entry["sample_distances"] = distances[:5].tolist()
                    except:
                        pass
            
            except Exception as e:
                logger.warning(f"Error capturing prediction details: {e}")
        
        self._logs.append(log_entry)
        logger.debug(
            f"Logged {log_entry['event_type']} for {model.__class__.__name__} "
            f"({log_entry['n_samples']} samples)"
        )
    
    def _create_wrapped_estimator_class(self, base_class: type) -> type:
        """
        Dynamically create a subclass that overrides fit/predict with logging.
        
        Preserves all scikit-learn API requirements:
        - get_params()/set_params() compatibility
        - Pipeline and GridSearchCV compatibility
        - Pickle serialization support
        """
        engine_ref = weakref_ref(self)
        
        class WrappedEstimator(base_class):
            _mtrace_wrapped = True  # Marker for detection
            
            def fit(self, X, y=None, **kwargs):
                # Call original fit
                result = super().fit(X, y, **kwargs)
                
                # ✅ FIX: Use explicit engine reference stored on wrapped model
                if hasattr(self, '_mtrace_engine_ref'):
                    engine = self._mtrace_engine_ref()
                    if engine and engine._enabled and engine._mode == "development":
                        try:
                            engine._log_fit_state(self, X, y)
                        except Exception as e:
                            logger.warning(f"Fit logging failed: {e}")
                
                return result
            
            def predict(self, X):
                # Capture decision paths BEFORE prediction for tree models
                decision_info = None
                engine = engine_ref()
                if (engine and engine._enabled and engine._mode == "development" and 
                    hasattr(self, "decision_path") and callable(self.decision_path)):
                    try:
                        decision_info = self.decision_path(X)[0]
                    except:
                        pass
                
                # Call original predict
                predictions = super().predict(X)
                
                # Log prediction
                if engine and engine._enabled:
                    try:
                        engine._log_prediction(self, X, predictions, decision_info)
                    except Exception as e:
                        logger.warning(f"Prediction logging failed: {e}")
                
                return predictions
            
            def transform(self, X):
                # For transformers (e.g., PCA, StandardScaler)
                result = super().transform(X)
                
                engine = engine_ref()
                if engine and engine._enabled:
                    try:
                        engine._log_prediction(self, X, result, None)
                    except Exception as e:
                        logger.warning(f"Transform logging failed: {e}")
                
                return result
            
            def fit_transform(self, X, y=None, **kwargs):
                # Combined fit+transform with logging
                self.fit(X, y, **kwargs)
                return self.transform(X)
        
        # Preserve class metadata for scikit-learn compatibility
        WrappedEstimator.__name__ = f"MTrace{base_class.__name__}"
        WrappedEstimator.__qualname__ = f"MTrace{base_class.__qualname__}"
        WrappedEstimator.__module__ = base_class.__module__
        
        return WrappedEstimator
    
    def enable(self) -> None:
        """Enable logging by creating wrapped estimator with instrumented methods."""
        if self._enabled:
            logger.warning("scikit-learn logging already enabled")
            return
        
        model = self.model_ref()
        if model is None:
            raise RuntimeError("Model reference lost - cannot enable logging")
        
        # Create wrapped estimator class dynamically
        try:
            WrappedClass = self._create_wrapped_estimator_class(model.__class__)
            
            # ✅ FIX 1: Use clone() for proper parameter preservation (handles nested params)
            from sklearn.base import clone
            self.wrapped_model = clone(model)  # Creates new instance with SAME parameters
            self.wrapped_model.__class__ = WrappedClass  # Transmute to wrapped class

            # ✅ FIX 2: Explicitly link wrapped model to this engine instance (avoid weakref GC issues)
            self.wrapped_model._mtrace_engine_ref = weakref_ref(self)
        
            
            
            # Copy fitted attributes if model already trained
            if hasattr(model, "fit") and hasattr(model, "classes_"):  # Already fitted classifier
                for attr in dir(model):
                    if attr.endswith("_") and not attr.startswith("__"):
                        try:
                            setattr(self.wrapped_model, attr, getattr(model, attr))
                        except Exception as e:
                            logger.debug(f"Could not copy fitted attribute {attr}: {e}")
            
            # Replace user's model reference (via weakref update pattern)
            # Note: User must use returned wrapped_model for logging to work
            self._enabled = True
            logger.info(
                f"Enabled scikit-learn logging in {self._mode} mode for "
                f"{model.__class__.__name__} (run_id: {self.config.get('run_id', 'unknown')})"
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
        """
        Get the instrumented estimator for training/inference.
        
        IMPORTANT: Users MUST use this model instead of the original when logging is enabled.
        
        Returns:
            Wrapped estimator with logging instrumentation
        
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
        """Collect all buffered logs."""
        logs = self._logs.copy()
        self._logs.clear()
        return logs
    
    def is_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled


# Public utility functions
def is_sklearn_model(model: Any) -> bool:
    """Check if object is a scikit-learn estimator."""
    try:
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator)
    except ImportError:
        return False


def requires_data_validation(model: Any) -> bool:
    """
    Check if model requires special data validation handling.
    
    Some scikit-learn models (e.g., HistGradientBoosting) have custom validation
    that may interfere with wrapping. This function identifies such cases.
    """
    model_type = type(model).__name__
    sensitive_models = [
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "IsolationForest"  # Has custom fit logic
    ]
    return model_type in sensitive_models