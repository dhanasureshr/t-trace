"""On-the-fly compression utilities for log data."""
import io
import logging
from typing import Any, Optional, Tuple
import numpy as np

try:
    import snappy
except ImportError:
    snappy = None

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import gzip
except ImportError:
    gzip = None

logger = logging.getLogger(__name__)


class CompressionEngine:
    """Handles on-the-fly compression of log data with multiple algorithm support."""
    
    SUPPORTED_ALGORITHMS = ["snappy", "zstd", "gzip", "none"]
    
    def __init__(self, algorithm: str = "snappy", level: int = 1):
        """Initialize compression engine with specified algorithm and level."""
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported compression algorithm: {algorithm}. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )
        
        self.algorithm = algorithm
        self.level = level
        
        # Validate dependencies
        if algorithm == "snappy" and snappy is None:
            raise ImportError("Snappy compression requires 'python-snappy' package")
        elif algorithm == "zstd" and zstd is None:
            raise ImportError("Zstandard compression requires 'zstandard' package")
        elif algorithm == "gzip" and gzip is None:
            raise ImportError("Gzip compression requires 'gzip' module (standard library)")
    
    def compress(self, data: Any) -> Tuple[bytes, str]:
        """
        Compress data using configured algorithm.
        
        Args:
            data: Serializable data (dict, list, numpy array, etc.)
        
        Returns:
            Tuple of (compressed_bytes, original_dtype)
        """
        try:
            # Convert numpy arrays to bytes for efficient compression
            if isinstance(data, np.ndarray):
                dtype_str = str(data.dtype)
                buffer = data.tobytes()
            elif isinstance(data, (list, tuple)):
                # Convert list of numbers to numpy array first
                arr = np.array(data)
                dtype_str = str(arr.dtype)
                buffer = arr.tobytes()
            else:
                # Fallback to pickle for complex objects
                import pickle
                buffer = pickle.dumps(data)
                dtype_str = "pickle"
            
            # Apply compression
            if self.algorithm == "snappy" and snappy:
                compressed = snappy.compress(buffer)
            elif self.algorithm == "zstd" and zstd:
                cctx = zstd.ZstdCompressor(level=self.level)
                compressed = cctx.compress(buffer)
            elif self.algorithm == "gzip" and gzip:
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=self.level) as f:
                    f.write(buffer)
                compressed = buf.getvalue()
            else:  # "none" or fallback
                compressed = buffer
                dtype_str = f"uncompressed_{dtype_str}"
            
            logger.debug(
                f"Compressed {len(buffer)} bytes to {len(compressed)} bytes "
                f"using {self.algorithm} (ratio: {len(compressed)/len(buffer):.2f})"
            )
            return compressed, dtype_str
            
        except Exception as e:
            logger.warning(f"Compression failed with {self.algorithm}: {e}. Falling back to uncompressed.")
            # Fallback to uncompressed
            if isinstance(data, np.ndarray):
                return data.tobytes(), f"uncompressed_{data.dtype}"
            else:
                import pickle
                return pickle.dumps(data), "uncompressed_pickle"
    
    def decompress(self, compressed_data: bytes, dtype_hint: str) -> Any:
        """Decompress data using stored dtype hint."""
        try:
            # Apply decompression
            if self.algorithm == "snappy" and snappy:
                buffer = snappy.decompress(compressed_data)
            elif self.algorithm == "zstd" and zstd:
                dctx = zstd.ZstdDecompressor()
                buffer = dctx.decompress(compressed_data)
            elif self.algorithm == "gzip" and gzip:
                buf = io.BytesIO(compressed_data)
                with gzip.GzipFile(fileobj=buf, mode='rb') as f:
                    buffer = f.read()
            else:
                buffer = compressed_data
            
            # Reconstruct based on dtype hint
            if dtype_hint.startswith("uncompressed_") or dtype_hint == "pickle":
                import pickle
                return pickle.loads(buffer)
            elif dtype_hint.startswith("float") or dtype_hint.startswith("int"):
                # Reconstruct numpy array
                dtype = np.dtype(dtype_hint)
                return np.frombuffer(buffer, dtype=dtype)
            else:
                # Fallback
                import pickle
                return pickle.loads(buffer)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise