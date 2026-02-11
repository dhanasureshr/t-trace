"""Configuration management for LoggingEngine with YAML support and dynamic reloading."""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class LoggingConfig:
    """Manages M-TRACE logging configuration with defaults and validation per Section 3.1.7."""
    
    # Support both .yml (preferred) and .yaml extensions per industry standards
    DEFAULT_CONFIG_PATHS = [
        Path("config.yml"),    # Preferred extension (shorter, ML ecosystem standard)
        Path("config.yaml")    # Fallback extension
    ]
    
    # Comprehensive default config aligned with Section 3.1 specification
    DEFAULT_CONFIG = {
        # Root-level mode differentiation (Section 3.1.7)
        "mode": "development",  # Options: "development" or "production"
        
        # Core logging behavior
        "logging": {
            "log_level": "detailed",  # Options: "minimal", "detailed", "debug"
            "log_type": "attention"   # Options: "attention", "embedding", "prediction", "all"
        },
        
        # Sparse Logging Configuration (Section 3.1.4)
        "sparse_logging": {
            "enabled": True,
            "sparse_threshold": 0.1,   # Log values with abs(value) > threshold
            "top_k_values": 5,         # Always log top-k values (even below threshold)
            "quantization": {
                "enabled": False,
                "precision": "8-bit",   # Options: "8-bit", "16-bit"
                "scale": 0.1
            }
        },
        
        # Compression Configuration (Section 3.1.3)
        "compression": {
            "enabled": True,
            "compression_type": "snappy",  # Options: "snappy", "zstd", "gzip", "none"
            "compression_level": 1         # 1=fastest, 9=max compression (ignored for snappy)
        },
        
        # Logging Frequency Configuration (Section 3.1.5)
        "logging_frequency": {
            "batch_size": 1000,    # Write to storage after N logs collected
            "time_interval": 60    # Max seconds between writes (even if batch not full)
        },
        
        # Modular Field Selection (Section 3.1.7)
        "default_fields": [
            "model_type",
            "framework",
            "timestamp",
            "run_id",
            "mode",
            "layer_name",
            "losses"
        ],
        "custom_fields": [
            "attention_weights",
            "feature_maps",
            "gradients",
            "feature_importance",
            "decision_paths",
            "layer_activations",
            "attention_distributions"
            # Note: contextual fields (input_data/output_data) disabled by default for privacy
        ],
        
        # Error Handling Configuration (Section 3.1.6)
        "error_handling": {
            "max_storage_retries": 3,
            "emergency_recovery": True,
            "fallback_to_minimal": True
        },
        
        # Storage Configuration (Section 3.2)
        "storage": {
            "backend": "local",        # Options: "local", "s3", "hdfs"
            "directory": "mtrace_logs",
            "format": "parquet",
            "compression": "snappy"    # Storage-level compression (separate from log compression)
        },
        
        # Feedback Loop Configuration
        "feedback_loop": {
            "enabled": True,
            "error_analysis": {
                "loss_threshold": 0.1,
                "gradient_norm_threshold": 0.01
            }
        },
        
        # Multimodal Configuration
        "multimodal": {
            "enabled": True,
            "modalities": ["text", "image", "audio"],
            "cross_modal_interactions": {
                "enabled": True,
                "fusion_method": "attention"
            }
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration with dual-extension support (.yml/.yaml).
        
        Args:
            config_path: Optional explicit path to config file. If None, auto-discovers
                         config.yml → config.yaml → creates config.yml
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Auto-discover: try .yml first, then .yaml
            for path in self.DEFAULT_CONFIG_PATHS:
                if path.exists():
                    self.config_path = path
                    logger.info(f"Discovered configuration at {self.config_path}")
                    break
            else:
                # Neither exists → create config.yml (preferred extension)
                self.config_path = self.DEFAULT_CONFIG_PATHS[0]
                logger.warning(
                    f"No config file found. Creating default config at {self.config_path}"
                )
        
        self._last_reload_time = 0.0
        self._config = self._load_or_create_default()
    
    def _load_or_create_default(self) -> Dict[str, Any]:
        """Load config from file or create default if missing with backward compatibility."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    raw_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
                return self._migrate_and_validate(raw_config)
            else:
                self._save_default_config()
                return self.DEFAULT_CONFIG.copy()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return self.DEFAULT_CONFIG.copy()
    
    def _migrate_and_validate(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate legacy configs and merge with defaults.
        
        Handles backward compatibility for:
        - 'top_k' → 'top_k_values'
        - 'algorithm' → 'compression_type'
        - Missing root-level 'mode' field
        """
        # Apply migrations for legacy configs
        if "sparse_logging" in user_config:
            # Migration: top_k → top_k_values
            if "top_k" in user_config["sparse_logging"] and "top_k_values" not in user_config["sparse_logging"]:
                user_config["sparse_logging"]["top_k_values"] = user_config["sparse_logging"].pop("top_k")
                logger.warning("Migrated legacy config: 'top_k' → 'top_k_values'")
        
        if "compression" in user_config:
            # Migration: algorithm → compression_type
            if "algorithm" in user_config["compression"] and "compression_type" not in user_config["compression"]:
                user_config["compression"]["compression_type"] = user_config["compression"].pop("algorithm")
                logger.warning("Migrated legacy config: 'algorithm' → 'compression_type'")
        
        # Ensure root-level mode exists (required by spec)
        if "mode" not in user_config:
            user_config["mode"] = self.DEFAULT_CONFIG["mode"]
            logger.warning("Added missing 'mode' field to config (default: 'development')")
        
        # Merge with defaults (user config overrides defaults)
        return self._deep_merge(self.DEFAULT_CONFIG.copy(), user_config)
    
    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config into defaults without overwriting nested structures."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _save_default_config(self) -> None:
        """Save comprehensive default configuration to file."""
        try:
            # Ensure parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, "w") as f:
                yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Created default config at {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not save default config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Examples:
            config.get("sparse_logging.sparse_threshold") → 0.1
            config.get("mode") → "development"
            config.get("non.existent", "fallback") → "fallback"
        """
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key '{key}' not found in {self.config_path}")
    
    def should_reload(self) -> bool:
        """Check if config file has been modified since last reload."""
        if not self.config_path.exists():
            return False
        
        last_modified = self.config_path.stat().st_mtime
        if last_modified > self._last_reload_time:
            self._last_reload_time = time.time()
            return True
        return False
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed. Returns True if reloaded."""
        if self.should_reload():
            self._config = self._load_or_create_default()
            logger.info(f"Configuration reloaded from {self.config_path}")
            return True
        return False
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration dictionary with auto-reload."""
        self.reload_if_changed()
        return self._config
    
    def is_development_mode(self) -> bool:
        """Convenience method to check if running in development mode."""
        return self.get("mode", "development") == "development"
    
    def is_production_mode(self) -> bool:
        """Convenience method to check if running in production mode."""
        return self.get("mode", "development") == "production"