"""Core LoggingEngine implementation with framework detection and log management."""
import os
import uuid
import time
import threading
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .config import LoggingConfig
from .compression import CompressionEngine
from .pytorch import PyTorchLoggingEngine
from .tensorflow import TensorFlowLoggingEngine  
from .sklearn import SklearnLoggingEngine  # To be implemented

logger = logging.getLogger(__name__)


class LoggingEngine:
    """
    Core LoggingEngine for M-TRACE: captures model internal states in real-time.
    
    This engine automatically detects the model framework, attaches appropriate hooks/callbacks,
    and manages log buffering with batched writes to minimize performance overhead.
    """
    
    SUPPORTED_MODES = ["development", "production"]
    SUPPORTED_FRAMEWORKS = ["pytorch", "tensorflow", "sklearn"]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the LoggingEngine.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default location.
        """
        self.config_manager = LoggingConfig(
            Path(config_path) if config_path else None
        )
        self.config = self.config_manager.config
        self.run_id = str(uuid.uuid4())
        self._framework_engine: Optional[Any] = None
        self._model_ref: Optional[Any] = None
        self._log_buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._last_write_time = time.time()
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_writer = threading.Event()
        self._enabled = False
        self._mode = "production"
        self._storage_engine = None  # Lazy-initialized storage engine
        self._storage_retry_count = 0  # Track consecutive storage failures
        self._max_storage_retries = 3  # Max retries before dropping logs
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure internal logging."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def _detect_framework(self, model: Any) -> str:
        """
        Detect the framework of the provided model.
        
        Args:
            model: Machine learning model instance
            
        Returns:
            Framework name ("pytorch", "tensorflow", or "sklearn")
            
        Raises:
            ValueError: If framework cannot be detected or is unsupported
        """
        # PyTorch detection
        try:
            import torch
            import torch.nn as nn
            if isinstance(model, nn.Module):
                return "pytorch"
        except ImportError:
            pass
        
        # TensorFlow/Keras detection
        try:
            import tensorflow as tf
            if isinstance(model, (tf.keras.Model, tf.Module)):
                return "tensorflow"
        except ImportError:
            pass
        
        # scikit-learn detection
        try:
            from sklearn.base import BaseEstimator
            if isinstance(model, BaseEstimator):
                return "sklearn"
        except ImportError:
            pass
        
        # Fallback detection via module inspection
        model_type = type(model)
        module_name = model_type.__module__.lower()
        
        if "torch" in module_name:
            return "pytorch"
        elif "tensorflow" in module_name or "keras" in module_name:
            return "tensorflow"
        elif "sklearn" in module_name:
            return "sklearn"
        
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported frameworks: {self.SUPPORTED_FRAMEWORKS}"
        )
    
    def _create_framework_engine(self, model: Any, framework: str, mode: str) -> Any:
        """Create framework-specific logging engine."""
        config_with_mode = {**self.config, "mode": mode}
        
        if framework == "pytorch":
            return PyTorchLoggingEngine(model, config_with_mode)
        elif framework == "tensorflow":
            # Lazy import to avoid hard dependency
            from .tensorflow import TensorFlowLoggingEngine
            return TensorFlowLoggingEngine(model, config_with_mode)
        elif framework == "sklearn":
            # Lazy import to avoid hard dependency
            from .sklearn import SklearnLoggingEngine
            return SklearnLoggingEngine(model, config_with_mode)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _get_storage_engine(self) -> Any:
        """
        Lazily initialize and return the storage engine.
        
        Implements pluggable backends per Section 3.2 with proper error handling.
        """
        if self._storage_engine is not None:
            return self._storage_engine
        
        # Import here to avoid circular dependency
        try:
            from t_trace.storage_engine import get_storage_engine
        except ImportError as e:
            logger.error(
                f"StorageEngine import failed: {e}. "
                "Ensure mtrace package is properly installed. "
                "Falling back to in-memory buffering only."
            )
            raise RuntimeError(
                "StorageEngine unavailable. Install required dependencies: "
                "pip install pyarrow snappy python-snappy"
            ) from e
        
        # Build storage configuration from logging config
        storage_config = {
            "storage_dir": self.config.get("storage.directory", "mtrace_logs"),
            "backend": self.config.get("storage.backend", "local"),
            "compression": self.config.get("compression", {
                "compression_type": "snappy",
                "compression_level": 1,
                "enabled": True
            }),
            "sparse_logging": self.config.get("sparse_logging", {
                "enabled": True,
                "sparse_threshold": 0.1,
                "top_k_values": 5
            })
        }
        
        try:
            self._storage_engine = get_storage_engine(
                backend=storage_config["backend"],
                config=storage_config
            )
            logger.info(
                f"Storage engine initialized: {storage_config['backend']} "
                f"at {storage_config['storage_dir']}"
            )
            return self._storage_engine
            
        except Exception as e:
            logger.error(f"Failed to initialize storage engine: {e}")
            raise RuntimeError(f"Storage initialization failed: {e}") from e
    
    def _start_writer_thread(self) -> None:
        """Start background thread for batched log writes."""
        if self._writer_thread and self._writer_thread.is_alive():
            return
        
        self._stop_writer.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="mtrace-log-writer",
            daemon=True
        )
        self._writer_thread.start()
        logger.debug("Started log writer thread")
    
    def _writer_loop(self) -> None:
        """Background loop for periodic log writes per Section 3.1.5."""
        batch_size = self.config.get("logging_frequency.batch_size", 1000)
        time_interval = self.config.get("logging_frequency.time_interval", 60)
        
        while not self._stop_writer.is_set():
            time.sleep(1)  # Check every second
            
            with self._buffer_lock:
                buffer_size = len(self._log_buffer)
                time_since_last_write = time.time() - self._last_write_time
            
            # Write if batch is full or time interval exceeded
            if buffer_size >= batch_size or time_since_last_write >= time_interval:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """
        Flush logs from framework engine to storage engine.
        
        CRITICAL FIX: Logs are stored in framework engine hooks (not LoggingEngine._log_buffer).
        This method MUST pull logs directly from framework engine via collect_logs().
        """
        # Primary source: framework engine hooks (where PyTorch/TensorFlow hooks store logs)
        logs_to_write = []
        if self._enabled and self._framework_engine and self._framework_engine.is_enabled():
            framework_logs = self._framework_engine.collect_logs()
            if framework_logs:
                logs_to_write.extend(framework_logs)
                logger.debug(f"Collected {len(framework_logs)} logs from framework engine")
        
        # Secondary source: direct log buffer (for rare cases where _add_log() was called)
        with self._buffer_lock:
            if self._log_buffer:
                logs_to_write.extend(self._log_buffer.copy())
                self._log_buffer.clear()
                logger.debug(f"Collected {len(self._log_buffer)} logs from direct buffer")
        
        # Exit early ONLY if truly no logs to write
        if not logs_to_write:
            logger.debug("No logs to flush - skipping storage write")
            return
        
        self._last_write_time = time.time()
        
        try:
            # Lazy initialize storage engine if needed
            if self._storage_engine is None:
                from t_trace.storage_engine import get_storage_engine
                
                storage_config = {
                    "storage_dir": self.config.get("storage.directory", "mtrace_logs"),
                    "backend": self.config.get("storage.backend", "local"),
                    "compression": self.config.get("compression", {
                        "compression_type": "snappy",
                        "compression_level": 1,
                        "enabled": True
                    }),
                    "sparse_logging": self.config.get("sparse_logging", {
                        "enabled": True,
                        "sparse_threshold": 0.1,
                        "top_k_values": 5
                    })
                }
                
                self._storage_engine = get_storage_engine(
                    backend=storage_config["backend"],
                    config=storage_config
                )
                logger.info(f"✓ Storage engine initialized at {storage_config['storage_dir']}")
            
            # Extract model type for filename generation
            model_type = "unknown"
            if self._model_ref is not None:
                try:
                    model = self._model_ref()
                    if model is not None:
                        model_type = model.__class__.__name__.lower()
                except Exception:
                    pass
            
            # CRITICAL: Ensure directory exists BEFORE write attempt
            mode_dir = "development" if self._mode == "development" else "production"
            storage_dir = Path(self.config.get("storage.directory", "mtrace_logs"))
            (storage_dir / mode_dir).mkdir(parents=True, exist_ok=True)
            
            # WRITE TO STORAGE (this is the critical operation that was missing)
            filepath = self._storage_engine.save_logs(
                logs=logs_to_write,
                run_id=self.run_id,
                model_type=model_type,
                mode=self._mode
            )
            
            if filepath:
                logger.info(f"✓ Saved {len(logs_to_write)} logs to {filepath}")
            else:
                logger.warning("Storage engine returned empty filepath - logs may not have been saved")
                
        except Exception as e:
            # CRITICAL: Log full error details with stack trace for debugging
            logger.error(
                f"✗ STORAGE WRITE FAILED (run_id={self.run_id[:8]}): {type(e).__name__}: {e}",
                exc_info=True  # ← SHOWS FULL STACK TRACE
            )
            # Safety: re-buffer a subset of logs for retry (prevent memory explosion)
            with self._buffer_lock:
                if len(self._log_buffer) < 1000:
                    self._log_buffer.extend(logs_to_write[:100])
                    logger.warning(f"Re-buffered {min(100, len(logs_to_write))} logs for retry")
    
    def _add_log(self, log_entry: Dict) -> None:
        """Add log entry to buffer with metadata enrichment."""
        # Enrich with run metadata
        log_entry.update({
            "run_id": self.run_id,
            "mode": self._mode,
            "timestamp": log_entry.get("timestamp", time.time()),
        })
        
        # Apply compression if enabled (field-level compression per Section 3.1.3)
        if self.config.get("compression.enabled", True):
            try:
                compression = CompressionEngine(
                    algorithm=self.config.get("compression.compression_type", "snappy"),
                    level=self.config.get("compression.compression_level", 1)
                )
                
                # Compress large fields
                for field in ["attention_weights", "feature_maps", "input", "output"]:
                    if field in log_entry and log_entry[field] is not None:
                        compressed, dtype = compression.compress(log_entry[field])
                        log_entry[f"compressed_{field}"] = compressed
                        log_entry[f"{field}_dtype"] = dtype
                        # Keep original for development mode debugging
                        if self._mode != "development":
                            del log_entry[field]
            except Exception as e:
                logger.warning(f"Compression failed for log entry: {e}")
        
        with self._buffer_lock:
            self._log_buffer.append(log_entry)
    
    def enable_logging(self, model: Any, mode: str = "production") -> None:
        """
        Public API: Enable logging for the provided model.
        
        Args:
            model: Machine learning model (PyTorch, TensorFlow, or scikit-learn)
            mode: Logging mode - "development" (detailed) or "production" (lightweight)
        
        Raises:
            ValueError: If mode is invalid or model/framework unsupported
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes: {self.SUPPORTED_MODES}"
            )
        
        # Disable existing logging if active
        if self._enabled:
            self.disable_logging()
        
        # Detect framework
        framework = self._detect_framework(model)
        logger.info(f"Detected framework: {framework}")
        
        # Create framework-specific engine
        self._framework_engine = self._create_framework_engine(model, framework, mode)
        self._model_ref = model
        self._mode = mode
        
        # Enable framework-specific logging
        self._framework_engine.enable()
        
        # Start writer thread
        self._start_writer_thread()
        
        self._enabled = True
        logger.info(
            f"M-TRACE logging enabled in {mode} mode for {framework} model "
            f"(run_id: {self.run_id})"
        )
    
    def disable_logging(self) -> None:
        """Disable logging and cleanup resources."""
        if not self._enabled:
            return
        
        # Stop writer thread
        self._stop_writer.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
        
        # Flush remaining logs (final write attempt)
        self._flush_buffer()
        
        # Disable framework-specific engine
        if self._framework_engine:
            self._framework_engine.disable()
        
        self._enabled = False
        self._framework_engine = None
        self._model_ref = None
        logger.info("M-TRACE logging disabled")
    
    def collect_logs(self) -> List[Dict]:
        """
        Collect all buffered logs from the current session.
        
        Returns:
            List of log entries with metadata
        """
        if not self._enabled or not self._framework_engine:
            return []
        
        # Collect framework-specific logs
        framework_logs = self._framework_engine.collect_logs()
        
        # Return combined logs (buffered + framework-specific)
        with self._buffer_lock:
            return self._log_buffer.copy() + framework_logs
    
    def is_logging_enabled(self) -> bool:
        """Check if logging is currently enabled."""
        return self._enabled
    
    def get_run_id(self) -> str:
        """Get the current run identifier."""
        return self.run_id
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure logging is disabled on context exit."""
        self.disable_logging()