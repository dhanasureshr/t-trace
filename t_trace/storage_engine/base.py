"""Abstract base class for storage engines with pluggable backends."""
import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pyarrow as pa

logger = logging.getLogger(__name__)


class StorageEngine(abc.ABC):
    """
    Abstract base class for M-TRACE storage engines.
    
    All storage backends must implement this interface to ensure
    consistent behavior across local/cloud/distributed storage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize storage engine with optional configuration.
        
        Args:
            config: Storage-specific configuration (e.g., bucket name for S3)
        """
        self.config = config or {}
        self._schema = None
        self._initialized = False
    
    @property
    def schema(self) -> pa.Schema:
        """Get the M-TRACE Parquet schema."""
        if self._schema is None:
            from .schema import get_mtrace_schema
            self._schema = get_mtrace_schema()
        return self._schema
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the storage backend (e.g., create bucket, verify permissions).
        
        Returns:
            True if initialization succeeded
        """
        pass
    
    @abc.abstractmethod
    def save_logs(
        self,
        logs: List[Dict[str, Any]],
        run_id: str,
        model_type: str,
        mode: str = "production"
    ) -> str:
        """
        Save logs to persistent storage.
        
        Args:
            logs: List of log entries to save
            run_id: Unique identifier for the model run
            model_type: Type of model (e.g., "bert", "resnet")
            mode: "development" or "production"
        
        Returns:
            Storage path/identifier where logs were saved
        
        Raises:
            StorageError: If saving fails
        """
        pass
    
    @abc.abstractmethod
    def retrieve_logs(
        self,
        run_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs for a specific run with optional time filtering.
        
        Args:
            run_id: Unique identifier for the model run
            start_time: Optional start timestamp (Unix epoch)
            end_time: Optional end timestamp (Unix epoch)
        
        Returns:
            List of log entries
        
        Raises:
            StorageError: If retrieval fails
        """
        pass
    
    @abc.abstractmethod
    def list_runs(
        self,
        model_type: Optional[str] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available model runs with optional filtering.
        
        Args:
            model_type: Filter by model type (e.g., "bert")
            mode: Filter by mode ("development" or "production")
        
        Returns:
            List of run metadata dictionaries
        """
        pass
    
    @abc.abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """
        Delete all logs associated with a run ID.
        
        Args:
            run_id: Unique identifier for the model run
        
        Returns:
            True if deletion succeeded
        
        Raises:
            StorageError: If deletion fails
        """
        pass
    
    def validate_logs(self, logs: List[Dict[str, Any]]) -> bool:
        """
        Validate logs against M-TRACE schema before saving.
        
        Args:
            logs: List of log entries to validate
        
        Returns:
            True if all logs are valid
        """
        from .schema import validate_log_entry
        
        for i, log in enumerate(logs):
            if not validate_log_entry(log, self.schema):
                logger.error(f"Log entry {i} failed schema validation")
                return False
        return True
    
    def _generate_filename(
        self,
        run_id: str,
        model_type: str,
        mode: str,
        timestamp: Optional[float] = None
    ) -> str:
        """
        Generate standardized filename per Section 3.1.1:
        model_run_<timestamp>_<run_id>.parquet
        
        Args:
            run_id: Unique run identifier
            model_type: Model type (e.g., "bert")
            mode: "development" or "production"
            timestamp: Optional timestamp (defaults to current time)
        
        Returns:
            Generated filename
        """
        import time
        from datetime import datetime
        
        if timestamp is None:
            timestamp = time.time()
        
        # Format: YYYYMMDD_HHMMSS
        dt = datetime.fromtimestamp(timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")
        
        # Sanitize model type for filename
        model_type_safe = "".join(c if c.isalnum() else "_" for c in model_type.lower())
        
        return f"model_run_{model_type_safe}_{timestamp_str}_{run_id[:8]}.parquet"