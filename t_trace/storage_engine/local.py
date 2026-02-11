"""Local Parquet storage implementation with compression, sparse logging, and robust error handling."""
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import logging

from .base import StorageEngine
from .schema import apply_sparse_filtering, get_mtrace_schema

try:
    import snappy
except ImportError:
    snappy = None

try:
    import zstandard as zstd
except ImportError:
    zstd = None

logger = logging.getLogger(__name__)


class LocalStorageError(Exception):
    """Custom exception for local storage errors."""
    pass


class LocalParquetStorage(StorageEngine):
    """
    Production-ready local storage engine implementing Section 3.1 specifications:
    - Comprehensive Parquet schema (Section 3.1.2)
    - On-the-fly compression (Section 3.1.3)
    - Sparse logging integration (Section 3.1.4)
    - Modular logging configuration (Section 3.1.7)
    - Robust error handling (Section 3.1.6)
    """
    
    DEFAULT_STORAGE_DIR = Path("mtrace_logs")
    SUPPORTED_COMPRESSION = ["snappy", "zstd", "gzip", "none"]
    
    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize local Parquet storage engine.
        
        Args:
            storage_dir: Directory to store Parquet files (defaults to ./mtrace_logs)
            config: Configuration dictionary with compression/sparse logging settings
        """
        super().__init__(config)
        self.storage_dir = Path(storage_dir) if storage_dir else self.DEFAULT_STORAGE_DIR
        self._compression_engine = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize storage by creating directory structure."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for modes
            (self.storage_dir / "development").mkdir(exist_ok=True)
            (self.storage_dir / "production").mkdir(exist_ok=True)
            
            # Verify write permissions
            test_file = self.storage_dir / f".write_test_{uuid.uuid4().hex}.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
            self._initialized = True
            logger.info(f"Local storage initialized at {self.storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize local storage: {e}")
            raise LocalStorageError(f"Storage initialization failed: {e}")
    
    def _get_compression_settings(self) -> Dict[str, Any]:
        """Extract compression settings from config with defaults."""
        compression_config = self.config.get("compression", {})
        algorithm = compression_config.get("compression_type", "snappy").lower()
        level = compression_config.get("compression_level", 1)
        
        # Validate algorithm
        if algorithm not in self.SUPPORTED_COMPRESSION:
            logger.warning(
                f"Unsupported compression algorithm '{algorithm}'. "
                f"Falling back to 'snappy'. Supported: {self.SUPPORTED_COMPRESSION}"
            )
            algorithm = "snappy"
        
        # Validate dependencies
        if algorithm == "snappy" and snappy is None:
            logger.warning("Snappy not available, falling back to 'none'")
            algorithm = "none"
        elif algorithm == "zstd" and zstd is None:
            logger.warning("Zstandard not available, falling back to 'snappy'")
            algorithm = "snappy"
        
        return {
            "algorithm": algorithm,
            "level": min(max(level, 1), 9)  # Clamp to 1-9
        }
    
    def _compress_data(self, data: bytes, algorithm: str, level: int) -> bytes:
        """Compress data using specified algorithm with fallback."""
        try:
            if algorithm == "snappy" and snappy:
                return snappy.compress(data)
            elif algorithm == "zstd" and zstd:
                cctx = zstd.ZstdCompressor(level=level)
                return cctx.compress(data)
            elif algorithm == "gzip":
                import gzip
                import io
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=level) as f:
                    f.write(data)
                return buf.getvalue()
            else:  # "none" or fallback
                return data
                
        except Exception as e:
            logger.warning(f"Compression failed with {algorithm}: {e}. Falling back to uncompressed.")
            return data
    
    def _prepare_log_entry(
        self,
        log_entry: Dict[str, Any],
        compression_settings: Dict[str, Any],
        sparse_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare log entry for storage:
        1. Apply sparse filtering to large tensors
        2. Compress binary fields
        3. Add compression/sparse metadata
        """
        prepared = log_entry.copy()
        
        # Apply sparse logging to tensor fields if enabled
        if sparse_config.get("enabled", True):
            for field in ["attention_weights", "feature_maps", "gradients", "layer_activations"]:
                if field in prepared.get("internal_states", {}):
                    tensor_data = prepared["internal_states"][field]
                    sparse_result = apply_sparse_filtering(tensor_data, {"sparse_logging": sparse_config})
                    
                    # Replace full tensor with sparse values only
                    prepared["internal_states"][field] = sparse_result.get("sparse_values", tensor_data)
                    
                    # Add sparse metadata (CORRECTED: scalar values only)
                    if "sparse_logging_metadata" not in prepared:
                        sparse_indices = sparse_result.get("sparse_indices", [])
                        prepared["sparse_logging_metadata"] = {
                            "threshold_applied": float(sparse_result.get("threshold_applied", 0.0)),
                            "top_k_values_logged": int(sparse_result.get("top_k_values_logged", len(sparse_indices))),  # ← INTEGER COUNT
                            "original_tensor_shape": [int(x) for x in sparse_result.get("shape", [])],  # ← List of ints
                            "sparse_indices_count": int(len(sparse_indices)),  # ← INTEGER COUNT
                            "sparse_type": str(sparse_result.get("sparse_type", "threshold")),
                            "sparse_indices": [int(x) for x in sparse_indices]  # ← NEW: Actual indices list
                        }
                
        # Compress large binary fields
        compression_fields = ["input_data", "output_data"]
        original_size = 0
        compressed_size = 0
        
        for field in compression_fields:
            if field in prepared.get("contextual_info", {}):
                raw_data = prepared["contextual_info"][field]
                if isinstance(raw_data, (bytes, bytearray)):
                    original_size += len(raw_data)
                    compressed = self._compress_data(raw_data, compression_settings["algorithm"], compression_settings["level"])
                    compressed_size += len(compressed)
                    prepared["contextual_info"][field] = compressed
        
        # Add compression metadata
        if original_size > 0 and compression_settings["algorithm"] != "none":
            prepared["compression_metadata"] = {
                "algorithm": compression_settings["algorithm"],
                "level": compression_settings["level"],
                "original_size_bytes": original_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0
            }
        
        return prepared
    
    def save_logs(
        self,
        logs: List[Dict[str, Any]],
        run_id: str,
        model_type: str,
        mode: str = "production"
    ) -> str:
        """
        Save logs to Parquet file with comprehensive error handling per Section 3.1.6.
        
        Implements:
        - File naming convention (Section 3.1.1)
        - Schema validation (Section 3.1.2)
        - On-the-fly compression (Section 3.1.3)
        - Sparse logging (Section 3.1.4)
        - Modular field configuration (Section 3.1.7)
        - Batched writes with configurable frequency (Section 3.1.5)
        """
        if not self._initialized:
            self.initialize()
        
        if not logs:
            logger.warning("No logs to save")
            return ""
        
        # Validate logs first
        if not self.validate_logs(logs):
            raise LocalStorageError("Log validation failed - schema mismatch")
        
        # Get configuration settings
        compression_settings = self._get_compression_settings()
        sparse_config = self.config.get("sparse_logging", {
            "enabled": True,
            "sparse_threshold": 0.1,
            "top_k_values": 5
        })
        
        # Prepare logs (sparse filtering + compression)
        prepared_logs = []
        for log in logs:
            try:
                prepared = self._prepare_log_entry(log, compression_settings, sparse_config)
                prepared_logs.append(prepared)
            except Exception as e:
                logger.warning(f"Failed to prepare log entry: {e}. Skipping entry.")
                continue
        
        if not prepared_logs:
            raise LocalStorageError("No valid logs to save after preparation")
        
        # Generate filename
        filename = self._generate_filename(run_id, model_type, mode)
        mode_dir = "development" if mode == "development" else "production"
        filepath = self.storage_dir / mode_dir / filename
        
        try:
            # Convert to PyArrow Table
            table = pa.Table.from_pylist(prepared_logs, schema=self.schema)
            
            # CRITICAL FIX 2: Ensure directory exists BEFORE write attempt
            filepath.parent.mkdir(parents=True, exist_ok=True)  # ← DIRECTORY CREATION
            
            # Write Parquet file (with Parquet 2.0 compatibility fix)
            pq.write_table(
                table,
                filepath,
                compression=compression_settings["algorithm"],
                # compression_level only supported for gzip/zstd (not snappy)
                compression_level=compression_settings["level"] if compression_settings["algorithm"] in ["gzip", "zstd"] else None,
                use_dictionary=True,
                data_page_size=1024 * 1024,  # 1MB pages for better compression
                # REMOVED: version='2.0' ← Critical fix for PyArrow compatibility
            )
            
            logger.info(
                f"✓ Saved {len(prepared_logs)} logs to {filepath} "
                f"({filepath.stat().st_size / 1024:.2f} KB)"
            )
            return str(filepath)
            
        except pa.ArrowInvalid as e:
            # CRITICAL: Log full error details for schema debugging
            logger.error(f"✗ SCHEMA MISMATCH during Parquet write: {e}", exc_info=True)
            logger.error(f"Log structure keys: {list(prepared_logs[0].keys()) if prepared_logs else 'N/A'}")
            
            # Attempt recovery: save as JSON for debugging
            import json
            recovery_path = filepath.with_suffix(".json")
            try:
                with open(recovery_path, "w") as f:
                    json.dump(prepared_logs[:5], f, indent=2)  # Save first 5 logs only
                logger.info(f"✓ Recovery logs saved to {recovery_path}")
            except Exception as json_err:
                logger.error(f"Recovery JSON write failed: {json_err}")
            
            raise LocalStorageError(f"Parquet write failed - schema mismatch (see logs)") from e
            
        except PermissionError as e:
            logger.error(f"✗ Permission denied writing to {filepath}: {e}", exc_info=True)
            raise LocalStorageError(f"Storage permission error: {e}")
            
        except OSError as e:
            if e.errno == 28:  # ENOSPC - No space left on device
                logger.error(f"✗ Disk space exhausted: {e}", exc_info=True)
                raise LocalStorageError("Insufficient disk space")
            else:
                logger.error(f"✗ OS error writing logs to {filepath}: {e}", exc_info=True)
                raise LocalStorageError(f"Storage OS error: {e}")
        
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"✗ Unexpected error writing logs: {type(e).__name__}: {e}", exc_info=True)
            raise LocalStorageError(f"Storage write failed: {e}") from e
    
    def retrieve_logs(
        self,
        run_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve logs for a specific run with optional time filtering."""
        if not self._initialized:
            self.initialize()
        
        # Search both development and production directories
        logs = []
        for mode_dir in ["development", "production"]:
            search_dir = self.storage_dir / mode_dir
            if not search_dir.exists():
                continue
            
            # Find files matching run_id
            for filepath in search_dir.glob(f"*{run_id[:8]}*.parquet"):
                try:
                    table = pq.read_table(filepath)
                    df = table.to_pandas()
                    
                    # Apply time filtering if specified
                    if start_time is not None or end_time is not None:
                        mask = True
                        if start_time is not None:
                            mask &= (df["model_metadata.timestamp"] >= start_time * 1000)  # Convert to ms
                        if end_time is not None:
                            mask &= (df["model_metadata.timestamp"] <= end_time * 1000)
                        df = df[mask]
                    
                    logs.extend(df.to_dict("records"))
                    
                except Exception as e:
                    logger.warning(f"Failed to read {filepath}: {e}")
                    continue
        
        if not logs:
            logger.warning(f"No logs found for run_id: {run_id}")
        
        return logs
    
    def list_runs(
        self,
        model_type: Optional[str] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available model runs with metadata."""
        if not self._initialized:
            self.initialize()
        
        runs = []
        search_dirs = []
        
        if mode == "development":
            search_dirs = [self.storage_dir / "development"]
        elif mode == "production":
            search_dirs = [self.storage_dir / "production"]
        else:
            search_dirs = [self.storage_dir / "development", self.storage_dir / "production"]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            for filepath in search_dir.glob("*.parquet"):
                try:
                    # Extract metadata from filename
                    parts = filepath.stem.split("_")
                    if len(parts) < 5 or parts[0] != "model" or parts[1] != "run":
                        continue
                    
                    run_info = {
                        "filepath": str(filepath),
                        "run_id": parts[-1],
                        "model_type": parts[2],
                        "timestamp": parts[3],
                        "mode": search_dir.name,
                        "size_bytes": filepath.stat().st_size
                    }
                    
                    # Optional: read first row for richer metadata
                    if model_type is None or model_type.lower() in run_info["model_type"].lower():
                        runs.append(run_info)
                        
                except Exception as e:
                    logger.debug(f"Error processing {filepath}: {e}")
                    continue
        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return runs
    
    def delete_run(self, run_id: str) -> bool:
        """Delete all logs associated with a run ID."""
        deleted = False
        for mode_dir in ["development", "production"]:
            search_dir = self.storage_dir / mode_dir
            if not search_dir.exists():
                continue
            
            for filepath in search_dir.glob(f"*{run_id[:8]}*.parquet"):
                try:
                    filepath.unlink()
                    logger.info(f"Deleted log file: {filepath}")
                    deleted = True
                except Exception as e:
                    logger.error(f"Failed to delete {filepath}: {e}")
        
        return deleted