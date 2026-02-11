"""Data loading and StorageEngine integration for AnalysisEngine."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import pyarrow.parquet as pq

try:
    from t_trace.storage_engine import get_storage_engine
    HAS_STORAGE_ENGINE = True
except ImportError:
    HAS_STORAGE_ENGINE = False
    get_storage_engine = None

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading from StorageEngine or local Parquet files.
    
    Provides unified interface for retrieving logs regardless of storage backend.
    Implements error handling and fallback mechanisms per Section 3.2.
    """
    
    def __init__(self, storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader with optional storage configuration.
        
        Args:
            storage_config: Configuration for StorageEngine (backend, directory, etc.)
        """
        self.storage_config = storage_config or {}
        self.storage_engine = None
        
        # Resolve storage directory with proper fallbacks
        self.storage_dir = Path(
            self.storage_config.get("directory") or 
            self.storage_config.get("storage_dir") or 
            "mtrace_logs"
        ).resolve()  # ← CRITICAL: Resolve to absolute path
        
        logger.info(f"DataLoader initialized with storage directory: {self.storage_dir}")
        
        # Initialize storage engine if available
        if HAS_STORAGE_ENGINE:
            try:
                backend = self.storage_config.get("backend", "local")
                self.storage_engine = get_storage_engine(
                    backend=backend,
                    config=self.storage_config
                )
                logger.info(f"Storage engine initialized: {backend}")
            except Exception as e:
                logger.warning(f"Failed to initialize storage engine: {e}. Falling back to local file access.")
                self.storage_engine = None
    
    def list_runs(
        self,
        model_type: Optional[str] = None,
        mode: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available model runs with metadata.
        
        Args:
            model_type: Filter by model type (e.g., "bert", "random_forest")
            mode: Filter by mode ("development" or "production")
        
        Returns:
            List of run metadata dictionaries with fields:
            - run_id: Unique identifier (full or truncated)
            - model_type: Model type/class name
            - timestamp: Run timestamp (YYYYMMDD_HHMMSS format)
            - mode: Environment mode
            - filepath: Absolute path to log file
            - size_bytes: File size
        """
        runs = []
        
        # Try StorageEngine first (if available)
        if self.storage_engine:
            try:
                runs = self.storage_engine.list_runs(model_type=model_type, mode=mode)
                logger.debug(f"Retrieved {len(runs)} runs from storage engine")
                return runs
            except Exception as e:
                logger.warning(f"Storage engine list_runs failed: {e}. Falling back to local scan.")
        
        # Fallback: scan local directory (with debug logging)
        logger.debug(f"Scanning local directory for runs: {self.storage_dir}")
        
        search_dirs = []
        if mode == "development":
            search_dirs = [self.storage_dir / "development"]
        elif mode == "production":
            search_dirs = [self.storage_dir / "production"]
        else:
            search_dirs = [
                self.storage_dir / "development",
                self.storage_dir / "production"
            ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                logger.debug(f"Directory does not exist (skipping): {search_dir}")
                continue
            
            logger.debug(f"Scanning directory: {search_dir}")
            parquet_files = list(search_dir.glob("*.parquet"))
            logger.debug(f"Found {len(parquet_files)} Parquet files in {search_dir}")
            
            for filepath in parquet_files:
                try:
                    # Extract metadata from filename
                    # Format: model_run_<model_type>_<timestamp>_<runid>.parquet
                    # Example: model_run_sequential_20260211_164344_d97e9c34.parquet
                    parts = filepath.stem.split("_")
                    
                    # Validate filename format (must have at least 5 parts: model, run, type, timestamp, runid)
                    if len(parts) < 5 or parts[0] != "model" or parts[1] != "run":
                        logger.warning(f"Skipping file with invalid name format: {filepath.name}")
                        continue
                    
                    # Extract components (handle truncated run IDs)
                    model_type_from_file = parts[2]
                    timestamp = parts[3] + "_" + parts[4] if len(parts) > 5 else parts[3]
                    run_id = parts[-1]  # Last part is always the run ID (possibly truncated)
                    
                    run_info = {
                        "filepath": str(filepath.resolve()),  # ← CRITICAL: Absolute path
                        "run_id": run_id,
                        "model_type": model_type_from_file,
                        "timestamp": timestamp,
                        "mode": search_dir.name,
                        "size_bytes": filepath.stat().st_size
                    }
                    
                    # Optional filter by model_type
                    if model_type is None or model_type.lower() in run_info["model_type"].lower():
                        runs.append(run_info)
                        logger.debug(f"Discovered run: {run_id} ({model_type_from_file})")
                        
                except Exception as e:
                    logger.debug(f"Error processing {filepath}: {e}")
                    continue
        
        # Sort by timestamp (newest first) - handle timestamp format variations
        try:
            runs.sort(key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.warning(f"Failed to sort runs by timestamp: {e}. Returning unsorted list.")
        
        logger.info(f"Discovered {len(runs)} runs in {self.storage_dir}")
        return runs
    
    def load_run_logs(self, run_id: str) -> Optional[pd.DataFrame]:
        """
        Load logs for a specific run ID.
        
        Args:
            run_id: Unique identifier for the model run (full or first 8 chars)
        
        Returns:
            DataFrame containing all logs for the run, or None if not found
        """
        if not run_id:
            logger.error("run_id cannot be empty")
            return None
        
        # Try StorageEngine first
        if self.storage_engine:
            try:
                logs = self.storage_engine.retrieve_logs(run_id=run_id)
                if logs:
                    df = pd.DataFrame(logs)
                    logger.info(f"Loaded {len(df)} logs for run {run_id} via StorageEngine")
                    return df
            except Exception as e:
                logger.warning(f"Storage engine retrieve_logs failed: {e}. Falling back to local file.")
        
        # Fallback: search local files (match first 8 chars of run_id)
        logger.debug(f"Searching local files for run_id: {run_id}")
        
        try:
            # Search both development and production directories
            for mode_dir in ["development", "production"]:
                search_dir = self.storage_dir / mode_dir
                if not search_dir.exists():
                    continue
                
                # Find files matching run_id (match first 8 chars for truncated IDs)
                search_pattern = f"*{run_id[:8]}*.parquet"
                for filepath in search_dir.glob(search_pattern):
                    try:
                        logger.info(f"Loading logs from: {filepath}")
                        table = pq.read_table(filepath)
                        df = table.to_pandas()
                        logger.info(f"Loaded {len(df)} logs from {filepath}")
                        return df
                    except Exception as e:
                        logger.warning(f"Failed to read {filepath}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error loading run logs: {e}")
        
        logger.warning(f"No logs found for run_id: {run_id}")
        return None
    
    def delete_run(self, run_id: str) -> bool:
        """Delete all logs associated with a run ID."""
        if self.storage_engine:
            try:
                return self.storage_engine.delete_run(run_id)
            except Exception as e:
                logger.error(f"Storage engine delete_run failed: {e}")
                return False
        else:
            # Local file deletion
            try:
                deleted = False
                for mode_dir in ["development", "production"]:
                    search_dir = self.storage_dir / mode_dir
                    if not search_dir.exists():
                        continue
                    
                    for filepath in search_dir.glob(f"*{run_id[:8]}*.parquet"):
                        filepath.unlink()
                        deleted = True
                        logger.info(f"Deleted log file: {filepath}")
                return deleted
            except Exception as e:
                logger.error(f"Error deleting local logs: {e}")
                return False
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary statistics for a run."""
        df = self.load_run_logs(run_id)
        if df is None or df.empty:
            return {}
        
        try:
            summary = {
                "total_logs": len(df),
                "unique_layers": df["internal_states"].apply(
                    lambda x: x.get("layer_name") if isinstance(x, dict) else None
                ).nunique() if "internal_states" in df.columns else 0,
                "framework": df["model_metadata"].apply(
                    lambda x: x.get("framework") if isinstance(x, dict) else None
                ).iloc[0] if "model_metadata" in df.columns else "unknown",
                "model_type": df["model_metadata"].apply(
                    lambda x: x.get("model_type") if isinstance(x, dict) else None
                ).iloc[0] if "model_metadata" in df.columns else "unknown",
                "mode": df["model_metadata"].apply(
                    lambda x: x.get("mode") if isinstance(x, dict) else None
                ).iloc[0] if "model_metadata" in df.columns else "unknown",
            }
            
            # Event type distribution
            if "event_type" in df.columns:
                summary["event_types"] = df["event_type"].value_counts().to_dict()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating run summary: {e}")
            return {"error": str(e)}