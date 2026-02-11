"""scikit-learn-specific hook implementation for M-TRACE LoggingEngine."""
"""
scikit-learn-specific hook implementation for M-TRACE LoggingEngine.
Production-grade optional dependency handling.
"""

import logging
import time
import numpy as np

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from weakref import ref as weakref_ref

# ---------------------------------------------------------------------
# Optional scikit-learn dependency handling
# ---------------------------------------------------------------------

try:
    import sklearn  # Runtime availability check
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if TYPE_CHECKING:
    # These imports are ONLY for static type checking.
    # They are never executed at runtime.
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    )
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import (
        LogisticRegression,
        LinearRegression,
        Ridge,
        Lasso,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.cluster import KMeans, DBSCAN

else:
    # Runtime fallbacks to keep code valid when sklearn is not installed.
    BaseEstimator = object
    ClassifierMixin = object
    RegressorMixin = object
    ClusterMixin = object

    DecisionTreeClassifier = object
    DecisionTreeRegressor = object
    ExtraTreeClassifier = object
    ExtraTreeRegressor = object

    RandomForestClassifier = object
    RandomForestRegressor = object
    GradientBoostingClassifier = object
    GradientBoostingRegressor = object

    LogisticRegression = object
    LinearRegression = object
    Ridge = object
    Lasso = object
    SGDClassifier = object
    SGDRegressor = object

    SVC = object
    SVR = object

    KNeighborsClassifier = object
    KNeighborsRegressor = object

    MLPClassifier = object
    MLPRegressor = object

    KMeans = object
    DBSCAN = object


logger = logging.getLogger(__name__)



class SklearnHook:
    """
    Captures model internals during scikit-learn estimator execution.
    
    Implements Section 3.1 requirements:
    - Estimator-type-specific internal state capture
    - Sparse logging for feature importance/coefficients
    - Development/production mode differentiation
    - Minimal overhead via conditional logging
    """
    
    def __init__(self, estimator_name: str, estimator_type: str, config: Dict[str, Any]):
        self.estimator_name = estimator_name
        self.estimator_type = estimator_type
        self.config = config
        self._logs: List[Dict] = []
        self._enabled = True
        self._mode = config.get("mode", "development")
        self._fitted = False  # Track if estimator has been fitted
    
    def _sparse_filter(self, array: np.ndarray) -> Dict[str, Any]:
        """
        Apply sparse logging per Section 3.1.4 to coefficient/importance arrays.
        
        Handles 1D and 2D arrays (e.g., multi-class coefficients).
        """
        if not self.config.get("sparse_logging", {}).get("enabled", True):
            return {"full_array": array.tolist()}
        
        # Convert to numpy if needed
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        abs_values = np.abs(array)
        threshold = self.config.get("sparse_logging", {}).get("sparse_threshold", 0.1)
        top_k = self.config.get("sparse_logging", {}).get("top_k_values", 5)
        
        # Handle multi-dimensional arrays (flatten for thresholding)
        original_shape = array.shape
        flat_abs = abs_values.flatten()
        flat_array = array.flatten()
        
        # Get indices above threshold
        above_threshold = flat_abs > threshold
        indices = np.where(above_threshold)[0]
        
        if len(indices) == 0:
            # Fallback to top-k
            if len(flat_array) <= top_k:
                # Return all values if array smaller than top_k
                return {
                    "sparse_values": flat_array.tolist(),
                    "sparse_indices": np.arange(len(flat_array)).tolist(),
                    "shape": list(original_shape),
                    "threshold_applied": threshold,
                    "sparse_type": "all_values"
                }
            
            top_k_indices = np.argpartition(flat_abs, -top_k)[-top_k:]
            values = flat_array[top_k_indices]
            return {
                "sparse_values": values.tolist(),
                "sparse_indices": top_k_indices.tolist(),
                "shape": list(original_shape),
                "threshold_applied": threshold,
                "sparse_type": "top_k"
            }
        
        # Return sparse representation
        values = flat_array[indices]
        return {
            "sparse_values": values.tolist(),
            "sparse_indices": indices.tolist(),
            "shape": list(original_shape),
            "threshold_applied": threshold,
            "sparse_type": "threshold"
        }
    
    def log_fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        **fit_params
    ) -> None:
        """Log training metadata and model internals after fit() completes."""
        if not self._enabled:
            return
        
        try:
            log_entry = {
                "timestamp": time.time(),
                "estimator_name": self.estimator_name,
                "estimator_type": self.estimator_type,
                "mode": self._mode,
                "event_type": "fit",
                "training_samples": X.shape[0] if hasattr(X, 'shape') else len(X),
                "training_features": X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 1,
            }
            
            # Capture training metadata if in development mode
            if self._mode == "development" and "training_dynamics" in self.config.get("custom_fields", []):
                log_entry["training_dynamics"] = {
                    "sample_weight_present": sample_weight is not None,
                    "fit_params_keys": list(fit_params.keys()) if fit_params else []
                }
            
            self._logs.append(log_entry)
            self._fitted = True
            
        except Exception as e:
            logger.warning(f"Error in fit logging for {self.estimator_name}: {e}")
    
    def log_predict(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        decision_function: Optional[np.ndarray] = None
    ) -> None:
        """Log prediction metadata and model internals during inference."""
        if not self._enabled:
            return
        
        try:
            log_entry = {
                "timestamp": time.time(),
                "estimator_name": self.estimator_name,
                "estimator_type": self.estimator_type,
                "mode": self._mode,
                "event_type": "predict",
                "inference_samples": X.shape[0] if hasattr(X, 'shape') else len(X),
            }
            
            # Capture predictions (always)
            log_entry["predictions"] = predictions[:10].tolist() if len(predictions) > 10 else predictions.tolist()
            
            # Capture probabilities if available and configured
            if probabilities is not None and "uncertainty_estimates" in self.config.get("custom_fields", []):
                log_entry["uncertainty_estimates"] = {
                    "confidence_scores": probabilities.max(axis=1)[:10].tolist() if probabilities.ndim > 1 else probabilities[:10].tolist()
                }
            
            # Capture decision function if available
            if decision_function is not None and "decision_function" in self.config.get("custom_fields", []):
                log_entry["decision_function"] = decision_function[:10].tolist() if len(decision_function) > 10 else decision_function.tolist()
            
            # Capture model internals (only if fitted and in development mode or explicitly configured)
            if self._fitted:
                log_entry["internal_states"] = self._capture_internal_states()
            
            self._logs.append(log_entry)
            
        except Exception as e:
            logger.warning(f"Error in predict logging for {self.estimator_name}: {e}")
    
    def _capture_internal_states(self) -> Dict[str, Any]:
        """Capture estimator-specific internal states based on type."""
        internals = {}
        
        try:
            # Tree-based models
            if self.estimator_type in ["decision_tree", "random_forest", "gradient_boosting"]:
                internals["feature_importance"] = self._sparse_filter(
                    getattr(self._wrapped_estimator, "feature_importances_", np.array([]))
                )
                
                if self._mode == "development" and hasattr(self._wrapped_estimator, "tree_"):
                    # Capture tree structure for single trees
                    tree = self._wrapped_estimator.tree_
                    internals["node_splits"] = {
                        "feature_indices": tree.feature[tree.feature >= 0].tolist()[:10],  # Only non-leaf nodes
                        "thresholds": tree.threshold[tree.feature >= 0].tolist()[:10],
                        "n_nodes": tree.node_count
                    }
            
            # Linear models
            elif self.estimator_type in ["logistic_regression", "linear_regression", "ridge", "lasso", "sgd"]:
                if hasattr(self._wrapped_estimator, "coef_"):
                    internals["coefficients"] = self._sparse_filter(self._wrapped_estimator.coef_)
                if hasattr(self._wrapped_estimator, "intercept_"):
                    internals["intercept"] = self._wrapped_estimator.intercept_.tolist() if hasattr(self._wrapped_estimator.intercept_, 'tolist') else float(self._wrapped_estimator.intercept_)
            
            # SVMs
            elif self.estimator_type in ["svc", "svr"]:
                if hasattr(self._wrapped_estimator, "support_vectors_"):
                    internals["support_vectors_count"] = self._wrapped_estimator.support_vectors_.shape[0]
                if hasattr(self._wrapped_estimator, "dual_coef_"):
                    internals["dual_coefficients"] = self._sparse_filter(self._wrapped_estimator.dual_coef_)
            
            # Clustering
            elif self.estimator_type in ["kmeans", "dbscan"]:
                if hasattr(self._wrapped_estimator, "cluster_centers_"):
                    internals["cluster_centers"] = self._wrapped_estimator.cluster_centers_[:5].tolist()  # First 5 clusters
            
            # Neural networks (MLP)
            elif self.estimator_type in ["mlp_classifier", "mlp_regressor"]:
                if hasattr(self._wrapped_estimator, "coefs_"):
                    internals["layer_weights_count"] = [w.shape for w in self._wrapped_estimator.coefs_]
                if hasattr(self._wrapped_estimator, "loss_"):
                    internals["training_loss"] = self._wrapped_estimator.loss_
            
        except Exception as e:
            logger.debug(f"Error capturing internal states for {self.estimator_name}: {e}")
        
        return internals
    
    def set_wrapped_estimator(self, estimator: BaseEstimator) -> None:
        """Set reference to wrapped estimator for internal state access."""
        self._wrapped_estimator = estimator
    
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


class SklearnEstimatorWrapper:
    """
    Dynamic wrapper that overrides estimator methods to inject logging.
    
    Strategy:
    1. Preserve original estimator behavior and signatures
    2. Inject logging before/after method execution
    3. Handle both single estimators and pipelines
    4. Support development/production mode differentiation
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        hook: SklearnHook,
        config: Dict[str, Any],
        estimator_type: str
    ):
        self._original_estimator = estimator
        self._hook = hook
        self._config = config
        self._estimator_type = estimator_type
        self._mode = config.get("mode", "development")
        self._hook.set_wrapped_estimator(self)
        
        # Preserve original methods
        self._original_fit = estimator.fit
        self._original_predict = getattr(estimator, 'predict', None)
        self._original_predict_proba = getattr(estimator, 'predict_proba', None)
        self._original_decision_function = getattr(estimator, 'decision_function', None)
        self._original_transform = getattr(estimator, 'transform', None)
        
        # Override methods
        self.fit = self._wrap_fit(estimator.fit)
        if self._original_predict:
            self.predict = self._wrap_predict(self._original_predict)
        if self._original_predict_proba:
            self.predict_proba = self._wrap_predict_proba(self._original_predict_proba)
        if self._original_decision_function:
            self.decision_function = self._wrap_decision_function(self._original_decision_function)
        if self._original_transform:
            self.transform = self._wrap_transform(self._original_transform)
        
        # Copy estimator attributes for duck-typing compatibility
        self._copy_estimator_attributes()
    
    def _copy_estimator_attributes(self) -> None:
        """Copy key attributes from original estimator for compatibility."""
        attrs_to_copy = [
            'classes_', 'n_classes_', 'n_features_in_', 'feature_names_in_',
            'n_outputs_', 'n_iter_', 'loss_curve_', 'coefs_', 'intercepts_'
        ]
        for attr in attrs_to_copy:
            if hasattr(self._original_estimator, attr):
                setattr(self, attr, getattr(self._original_estimator, attr))
    
    def _wrap_fit(self, original_fit):
        """Wrap fit method to log training metadata."""
        def wrapped_fit(X, y=None, **fit_params):
            # Execute original fit
            result = original_fit(X, y, **fit_params)
            
            # Log after successful fit
            try:
                sample_weight = fit_params.get('sample_weight')
                self._hook.log_fit(X, y, sample_weight, **fit_params)
            except Exception as e:
                logger.debug(f"Fit logging skipped: {e}")
            
            # Update wrapper attributes after fit
            self._copy_estimator_attributes()
            return result
        return wrapped_fit
    
    def _wrap_predict(self, original_predict):
        """Wrap predict method to log inference metadata."""
        def wrapped_predict(X, **predict_params):
            # Execute original predict
            predictions = original_predict(X, **predict_params)
            
            # Skip logging in production mode during training (if applicable)
            if self._mode == "production" and not hasattr(self, '_in_training'):
                return predictions
            
            # Capture probabilities/decision function if available
            probabilities = None
            decision_function = None
            
            try:
                if self._original_predict_proba and "uncertainty_estimates" in self._config.get("custom_fields", []):
                    probabilities = self._original_predict_proba(X)
                elif self._original_decision_function:
                    decision_function = self._original_decision_function(X)
            except Exception:
                pass  # Not all estimators support these methods
            
            # Log prediction
            self._hook.log_predict(X, predictions, probabilities, decision_function)
            return predictions
        return wrapped_predict
    
    def _wrap_predict_proba(self, original_predict_proba):
        """Wrap predict_proba to ensure it remains available."""
        def wrapped_predict_proba(X):
            return original_predict_proba(X)
        return wrapped_predict_proba
    
    def _wrap_decision_function(self, original_decision_function):
        """Wrap decision_function to ensure it remains available."""
        def wrapped_decision_function(X):
            return original_decision_function(X)
        return wrapped_decision_function
    
    def _wrap_transform(self, original_transform):
        """Wrap transform method for transformers."""
        def wrapped_transform(X, **transform_params):
            # Execute original transform
            result = original_transform(X, **transform_params)
            
            # Log transform operation in development mode
            if self._mode == "development":
                try:
                    self._hook._logs.append({
                        "timestamp": time.time(),
                        "estimator_name": self._hook.estimator_name,
                        "estimator_type": self._estimator_type,
                        "mode": self._mode,
                        "event_type": "transform",
                        "samples_transformed": X.shape[0] if hasattr(X, 'shape') else len(X)
                    })
                except Exception as e:
                    logger.debug(f"Transform logging skipped: {e}")
            
            return result
        return wrapped_transform
    
    def __getattr__(self, name):
        """Delegate unknown attributes to original estimator."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original_estimator, name)
    
    def __setattr__(self, name, value):
        """Set attributes on appropriate object."""
        if name.startswith('_') or name in ['fit', 'predict', 'predict_proba', 'decision_function', 'transform']:
            object.__setattr__(self, name, value)
        else:
            setattr(self._original_estimator, name, value)


class SklearnLoggingEngine:
    """
    Framework-specific engine for scikit-learn estimators.
    
    Implements M-TRACE Section 3.1 requirements:
    - Estimator type detection and specialized logging
    - Method wrapping without modifying original estimator
    - Development/production mode differentiation
    - Sparse logging for coefficients/feature importance
    - Support for pipelines and composite estimators
    
    Architecture decisions:
    1. Uses dynamic method wrapping instead of inheritance to preserve estimator behavior
    2. Detects estimator type to capture appropriate internals (trees, linear models, etc.)
    3. Production mode minimizes overhead by skipping detailed internal capture
    4. Development mode provides comprehensive introspection including decision paths
    """
    
    ESTIMATOR_TYPE_MAP = {
        # Trees
        DecisionTreeClassifier: "decision_tree",
        DecisionTreeRegressor: "decision_tree",
        ExtraTreeClassifier: "decision_tree",
        ExtraTreeRegressor: "decision_tree",
        # Ensembles
        RandomForestClassifier: "random_forest",
        RandomForestRegressor: "random_forest",
        GradientBoostingClassifier: "gradient_boosting",
        GradientBoostingRegressor: "gradient_boosting",
        # Linear models
        LogisticRegression: "logistic_regression",
        LinearRegression: "linear_regression",
        Ridge: "ridge",
        Lasso: "lasso",
        SGDClassifier: "sgd",
        SGDRegressor: "sgd",
        # SVMs
        SVC: "svc",
        SVR: "svr",
        # Neighbors
        KNeighborsClassifier: "knn",
        KNeighborsRegressor: "knn",
        # Neural networks
        MLPClassifier: "mlp_classifier",
        MLPRegressor: "mlp_regressor",
        # Clustering
        KMeans: "kmeans",
        DBSCAN: "dbscan",
    }
    
    def __init__(self, model: BaseEstimator, config: Dict[str, Any]):
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )
        
        if not isinstance(model, BaseEstimator):
            raise ValueError(
                f"Model must be a scikit-learn BaseEstimator, got {type(model)}"
            )
        
        self.model_ref = weakref_ref(model)
        self.original_model = model
        self.config = config
        self.hook: Optional[SklearnHook] = None
        self.wrapped_model: Optional[SklearnEstimatorWrapper] = None
        self._enabled = False
        self._mode = config.get("mode", "development")
        self._estimator_type = self._detect_estimator_type(model)
    
    def _detect_estimator_type(self, estimator: BaseEstimator) -> str:
        """Detect specific estimator type for specialized logging."""
        for cls, est_type in self.ESTIMATOR_TYPE_MAP.items():
            if isinstance(estimator, cls):
                return est_type
        
        # Fallback detection
        cls_name = estimator.__class__.__name__.lower()
        if "tree" in cls_name:
            return "decision_tree"
        elif "forest" in cls_name or "boost" in cls_name:
            return "random_forest"
        elif "logistic" in cls_name or "linear" in cls_name:
            return "linear_regression"
        elif "svc" in cls_name or "svr" in cls_name:
            return "svc"
        elif "kmeans" in cls_name:
            return "kmeans"
        elif "mlp" in cls_name:
            return "mlp_classifier"
        
        return "unknown"
    
    def enable(self) -> None:
        """Enable logging by creating wrapped estimator with instrumentation."""
        if self._enabled:
            logger.warning("scikit-learn logging already enabled")
            return
        
        model = self.model_ref()
        if model is None:
            raise RuntimeError("Model reference lost - cannot enable logging")
        
        # Create hook
        self.hook = SklearnHook(
            estimator_name=model.__class__.__name__,
            estimator_type=self._estimator_type,
            config=self.config
        )
        
        # Create wrapped model
        self.wrapped_model = SklearnEstimatorWrapper(
            estimator=model,
            hook=self.hook,
            config=self.config,
            estimator_type=self._estimator_type
        )
        
        self._enabled = True
        logger.info(
            f"Enabled scikit-learn logging in {self._mode} mode for "
            f"{self._estimator_type} estimator (run_id: {self.config.get('run_id', 'unknown')})"
        )
    
    def disable(self) -> None:
        """Disable logging and cleanup resources."""
        if not self._enabled:
            return
        
        # Clear hook
        if self.hook:
            self.hook.disable()
            self.hook = None
        
        # Release wrapped model
        self.wrapped_model = None
        
        self._enabled = False
        logger.info("Disabled scikit-learn logging")
    
    def get_wrapped_model(self) -> BaseEstimator:
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
        """Collect all buffered logs from hook."""
        if not self._enabled or self.hook is None:
            return []
        
        return self.hook.get_logs()
    
    def is_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled


# Public utility functions
def is_sklearn_model(model: Any) -> bool:
    """Check if object is a scikit-learn estimator."""
    if not HAS_SKLEARN:
        return False
    try:
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator)
    except ImportError:
        return False


def requires_pipeline_handling(model: BaseEstimator) -> bool:
    """
    Check if model is a Pipeline or composite estimator requiring special handling.
    
    Note: Full pipeline support requires recursive wrapping of all steps.
    This implementation focuses on single estimators; pipeline support
    can be added as a future enhancement.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        return isinstance(model, (Pipeline, ColumnTransformer))
    except ImportError:
        return False