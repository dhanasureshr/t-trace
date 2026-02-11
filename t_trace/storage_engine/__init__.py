"""Public API for M-TRACE StorageEngine with pluggable backends."""
from typing import Dict, Optional
from .base import StorageEngine
from .local import LocalParquetStorage

# Optional imports for cloud backends (lazy-loaded to avoid dependencies)
_S3_AVAILABLE = False
try:
    from .s3 import S3Storage
    _S3_AVAILABLE = True
except ImportError:
    pass

_HDFS_AVAILABLE = False
try:
    from .hdfs import HDFStorage
    _HDFS_AVAILABLE = True
except ImportError:
    pass

__all__ = ["get_storage_engine", "StorageEngine"]


def get_storage_engine(
    backend: str = "local",
    config: Optional[Dict] = None
) -> StorageEngine:
    """
    Factory function to get storage engine instance.
    
    Supports pluggable backends per Section 3.2:
    - "local": Local Parquet storage (default)
    - "s3": AWS S3 storage (requires boto3)
    - "hdfs": Hadoop Distributed File System (requires pyarrow[hdfs])
    
    Args:
        backend: Storage backend type ("local", "s3", "hdfs")
        config: Backend-specific configuration
    
    Returns:
        Initialized StorageEngine instance
    
    Raises:
        ValueError: If backend is unsupported or dependencies missing
    """
    config = config or {}
    
    if backend == "local":
        storage_dir = config.get("storage_dir")
        engine = LocalParquetStorage(storage_dir=storage_dir, config=config)
        engine.initialize()
        return engine
    
    elif backend == "s3":
        if not _S3_AVAILABLE:
            raise ValueError(
                "S3 backend requires 'boto3' and 's3fs' packages. "
                "Install with: pip install boto3 s3fs"
            )
        bucket = config.get("bucket_name")
        if not bucket:
            raise ValueError("S3 backend requires 'bucket_name' in config")
        engine = S3Storage(bucket_name=bucket, config=config)
        engine.initialize()
        return engine
    
    elif backend == "hdfs":
        if not _HDFS_AVAILABLE:
            raise ValueError(
                "HDFS backend requires PyArrow with HDFS support. "
                "Install with: pip install pyarrow[hdfs]"
            )
        host = config.get("host", "localhost")
        port = config.get("port", 8020)
        engine = HDFStorage(host=host, port=port, config=config)
        engine.initialize()
        return engine
    
    else:
        raise ValueError(
            f"Unsupported storage backend: {backend}. "
            f"Supported: 'local', 's3', 'hdfs'"
        )