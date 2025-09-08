"""Configuration management for Excel-to-CSV converter.

This module provides centralized configuration loading and management
with support for YAML files, environment variable overrides, and validation.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from excel_to_csv.models.data_models import (
    ArchiveConfig,
    Config,
    LoggingConfig,
    OutputConfig,
    RetryConfig,
)


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class ConfigManager:
    """Manages application configuration loading and validation.
    
    The ConfigManager handles:
    - Loading configuration from YAML files
    - Environment variable overrides
    - Configuration validation and defaults
    - Merging multiple configuration sources
    - Auto-loading config/default.yaml when no path specified
    
    Configuration Loading Order:
    1. If config_path is provided, load that file
    2. If config_path is None, try to load config/default.yaml
    3. If config/default.yaml doesn't exist, use built-in defaults
    
    Example:
        >>> config_manager = ConfigManager()
        >>> config = config_manager.load_config()  # Auto-loads config/default.yaml
        >>> print(config.confidence_threshold)
        0.8
    """
    
    # Environment variable prefix
    ENV_PREFIX = "EXCEL_TO_CSV_"
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "monitoring": {
            "folders": ["./input"],
            "file_patterns": ["*.xlsx", "*.xls"],
            "process_existing": True,
            "polling_interval": 5,
            "max_file_size": 100,
        },
        "confidence": {
            "threshold": 0.7,
            "weights": {
                "data_density": 0.4,
                "header_quality": 0.3,
                "consistency": 0.3,
            },
            "min_rows": 5,
            "min_columns": 2,
            "max_empty_percentage": 0.3,
        },
        "output": {
            "folder": "./output",
            "naming_pattern": "{filename}_{worksheet}.csv",
            "include_timestamp": True,
            "encoding": "utf-8",
            "delimiter": ",",
            "include_headers": True,
            "timestamp_format": "%Y%m%d_%H%M%S",
        },
        "processing": {
            "max_concurrent": 5,
            "retry": {
                "max_attempts": 3,
                "delay": 5,
                "backoff_factor": 2,
                "max_delay": 60,
            },
            "timeouts": {
                "file_processing": 300,
                "worksheet_analysis": 60,
            },
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": {
                "enabled": True,
                "path": "./logs/excel_to_csv.log",
                "max_size": 10,
                "backup_count": 5,
                "rotation": True,
            },
            "console": {
                "enabled": True,
                "level": "INFO",
            },
            "structured": {
                "enabled": False,
                "path": "./logs/excel_to_csv.json",
            },
        },
        "performance": {
            "memory": {
                "max_usage": 1024,
                "warning_threshold": 512,
            },
            "cache": {
                "enabled": True,
                "size": 100,
                "ttl": 3600,
            },
        },
        "security": {
            "validate_paths": True,
            "allowed_extensions": [".xlsx", ".xls"],
            "max_path_length": 260,
            "follow_symlinks": False,
        },
        "archiving": {
            "enabled": False,  # Disabled by default for backward compatibility
            "archive_folder_name": "archive",
            "timestamp_format": "%Y%m%d_%H%M%S",
            "handle_conflicts": True,
            "preserve_structure": True,
        },
    }
    
    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._config_cache: Dict[str, Config] = {}
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        use_env_overrides: bool = True
    ) -> Config:
        """Load configuration from file with optional environment overrides.
        
        Args:
            config_path: Path to configuration file. If None, will try to load
                        config/default.yaml, falling back to built-in defaults
            use_env_overrides: Whether to apply environment variable overrides
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        # Use cache key for memoization
        cache_key = f"{config_path}:{use_env_overrides}"
        if cache_key in self._config_cache:
            logger.debug(f"Using cached configuration for {cache_key}")
            return self._config_cache[cache_key]
        
        try:
            # Load base configuration
            config_dict = self._load_config_dict(config_path)
            
            # Apply environment overrides if requested
            if use_env_overrides:
                config_dict = self._apply_env_overrides(config_dict)
            
            # Convert to Config object
            config = self._dict_to_config(config_dict)
            
            # Validate configuration
            self._validate_config(config)
            
            # Cache and return
            self._config_cache[cache_key] = config
            logger.info(f"Configuration loaded successfully from {config_path or 'default'}")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e
    
    def _load_config_dict(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration dictionary from file or defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Try to load config/default.yaml first
            default_config_path = Path("config/default.yaml")
            if default_config_path.exists():
                logger.info(f"No config path provided, loading default config from {default_config_path}")
                config_path = default_config_path
            else:
                logger.info("No config path provided and config/default.yaml not found, using built-in defaults")
                return self.DEFAULT_CONFIG.copy()
        
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            logger.info("Using default configuration")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Merge with defaults
            config_dict = self._deep_merge(self.DEFAULT_CONFIG.copy(), file_config)
            logger.debug(f"Loaded configuration from {config_file}")
            return config_dict
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_file}: {e}")
        except (IOError, OSError) as e:
            raise ConfigurationError(f"Cannot read {config_file}: {e}")
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.
        
        Args:
            config_dict: Base configuration dictionary
            
        Returns:
            Configuration dictionary with environment overrides applied
        """
        # Define mappings from environment variables to config paths
        env_mappings = {
            f"{self.ENV_PREFIX}CONFIDENCE_THRESHOLD": ["confidence", "threshold"],
            f"{self.ENV_PREFIX}OUTPUT_FOLDER": ["output", "folder"],
            f"{self.ENV_PREFIX}LOG_LEVEL": ["logging", "level"],
            f"{self.ENV_PREFIX}MAX_CONCURRENT": ["processing", "max_concurrent"],
            f"{self.ENV_PREFIX}MAX_FILE_SIZE": ["monitoring", "max_file_size"],
            f"{self.ENV_PREFIX}INCLUDE_TIMESTAMP": ["output", "include_timestamp"],
            f"{self.ENV_PREFIX}ENCODING": ["output", "encoding"],
            f"{self.ENV_PREFIX}DELIMITER": ["output", "delimiter"],
            f"{self.ENV_PREFIX}ARCHIVE_ENABLED": ["archiving", "enabled"],
            f"{self.ENV_PREFIX}ARCHIVE_FOLDER_NAME": ["archiving", "archive_folder_name"],
            f"{self.ENV_PREFIX}ARCHIVE_TIMESTAMP_FORMAT": ["archiving", "timestamp_format"],
            f"{self.ENV_PREFIX}ARCHIVE_HANDLE_CONFLICTS": ["archiving", "handle_conflicts"],
            f"{self.ENV_PREFIX}ARCHIVE_PRESERVE_STRUCTURE": ["archiving", "preserve_structure"],
        }
        
        # Apply environment overrides
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value, config_path)
                self._set_nested_value(config_dict, config_path, converted_value)
                logger.debug(f"Applied environment override: {env_var}={converted_value}")
        
        # Handle special case for monitored folders (comma-separated)
        folders_env = os.getenv(f"{self.ENV_PREFIX}MONITORED_FOLDERS")
        if folders_env:
            folders = [folder.strip() for folder in folders_env.split(",")]
            config_dict["monitoring"]["folders"] = folders
            logger.debug(f"Applied monitored folders override: {folders}")
        
        return config_dict
    
    def _convert_env_value(self, value: str, config_path: List[str]) -> Any:
        """Convert environment variable string to appropriate type.
        
        Args:
            value: Environment variable string value
            config_path: Configuration path for type inference
            
        Returns:
            Converted value
        """
        # Boolean conversions
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Numeric conversions
        try:
            # Try integer first
            if "." not in value:
                return int(value)
            # Then float
            return float(value)
        except ValueError:
            pass
        
        # String value (default)
        return value
    
    def _set_nested_value(
        self, 
        dictionary: Dict[str, Any], 
        path: List[str], 
        value: Any
    ) -> None:
        """Set a nested dictionary value using a path list.
        
        Args:
            dictionary: Dictionary to modify
            path: List of keys representing the path
            value: Value to set
        """
        for key in path[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[path[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert configuration dictionary to Config object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        # Extract sections
        monitoring = config_dict.get("monitoring", {})
        confidence = config_dict.get("confidence", {})
        output = config_dict.get("output", {})
        processing = config_dict.get("processing", {})
        logging_config = config_dict.get("logging", {})
        archiving = config_dict.get("archiving", {})
        
        # Create sub-configurations
        retry_settings = RetryConfig(
            max_attempts=processing.get("retry", {}).get("max_attempts", 3),
            delay=processing.get("retry", {}).get("delay", 5),
            backoff_factor=processing.get("retry", {}).get("backoff_factor", 2),
            max_delay=processing.get("retry", {}).get("max_delay", 60),
        )
        
        output_config = OutputConfig(
            folder=Path(output["folder"]) if output.get("folder") else None,
            naming_pattern=output.get("naming_pattern", "{filename}_{worksheet}.csv"),
            encoding=output.get("encoding", "utf-8"),
            include_timestamp=output.get("include_timestamp", True),
            delimiter=output.get("delimiter", ","),
            include_headers=output.get("include_headers", True),
            timestamp_format=output.get("timestamp_format", "%Y%m%d_%H%M%S"),
        )
        
        logging_config_obj = LoggingConfig(
            level=logging_config.get("level", "INFO"),
            format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_enabled=logging_config.get("file", {}).get("enabled", True),
            file_path=Path(logging_config.get("file", {}).get("path", "./logs/excel_to_csv.log")),
            console_enabled=logging_config.get("console", {}).get("enabled", True),
            structured_enabled=logging_config.get("structured", {}).get("enabled", False),
        )
        
        archive_config = ArchiveConfig(
            enabled=archiving.get("enabled", False),
            archive_folder_name=archiving.get("archive_folder_name", "archive"),
            timestamp_format=archiving.get("timestamp_format", "%Y%m%d_%H%M%S"),
            handle_conflicts=archiving.get("handle_conflicts", True),
            preserve_structure=archiving.get("preserve_structure", True),
        )
        
        # Create main configuration
        return Config(
            monitored_folders=[Path(folder) for folder in monitoring.get("folders", ["./input"])],
            confidence_threshold=confidence.get("threshold", 0.7),
            output_folder=Path(output["folder"]) if output.get("folder") else None,
            file_patterns=monitoring.get("file_patterns", ["*.xlsx", "*.xls"]),
            logging=logging_config_obj,
            retry_settings=retry_settings,
            output_config=output_config,
            archive_config=archive_config,
            max_concurrent=processing.get("max_concurrent", 5),
            max_file_size_mb=monitoring.get("max_file_size", 100),
        )
    
    def _validate_config(self, config: Config) -> None:
        """Validate configuration object.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            # Validate monitored folders exist or can be created
            for folder in config.monitored_folders:
                if not folder.exists():
                    logger.warning(f"Monitored folder does not exist: {folder}")
                    # Try to create it
                    try:
                        folder.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created monitored folder: {folder}")
                    except (OSError, PermissionError) as e:
                        raise ConfigurationError(f"Cannot create monitored folder {folder}: {e}")
            
            # Validate output folder
            if config.output_folder:
                if not config.output_folder.exists():
                    try:
                        config.output_folder.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created output folder: {config.output_folder}")
                    except (OSError, PermissionError) as e:
                        raise ConfigurationError(f"Cannot create output folder {config.output_folder}: {e}")
            
            # Validate log directory
            if config.logging.file_enabled:
                log_dir = config.logging.file_path.parent
                if not log_dir.exists():
                    try:
                        log_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created log directory: {log_dir}")
                    except (OSError, PermissionError) as e:
                        raise ConfigurationError(f"Cannot create log directory {log_dir}: {e}")
            
            logger.debug("Configuration validation completed successfully")
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
    
    def save_config(self, config: Config, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration file
            
        Raises:
            ConfigurationError: If saving fails
        """
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert Config object back to dictionary
            config_dict = self._config_to_dict(config)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}") from e
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary for serialization.
        
        Args:
            config: Configuration object
            
        Returns:
            Configuration dictionary
        """
        return {
            "monitoring": {
                "folders": [str(folder) for folder in config.monitored_folders],
                "file_patterns": config.file_patterns,
                "max_file_size": config.max_file_size_mb,
            },
            "confidence": {
                "threshold": config.confidence_threshold,
            },
            "output": {
                "folder": str(config.output_folder) if config.output_folder else None,
                "naming_pattern": config.output_config.naming_pattern,
                "include_timestamp": config.output_config.include_timestamp,
                "encoding": config.output_config.encoding,
                "delimiter": config.output_config.delimiter,
                "include_headers": config.output_config.include_headers,
                "timestamp_format": config.output_config.timestamp_format,
            },
            "processing": {
                "max_concurrent": config.max_concurrent,
                "retry": {
                    "max_attempts": config.retry_settings.max_attempts,
                    "delay": config.retry_settings.delay,
                    "backoff_factor": config.retry_settings.backoff_factor,
                    "max_delay": config.retry_settings.max_delay,
                },
            },
            "logging": {
                "level": config.logging.level,
                "format": config.logging.format,
                "file": {
                    "enabled": config.logging.file_enabled,
                    "path": str(config.logging.file_path),
                },
                "console": {
                    "enabled": config.logging.console_enabled,
                },
                "structured": {
                    "enabled": config.logging.structured_enabled,
                },
            },
            "archiving": {
                "enabled": config.archive_config.enabled,
                "archive_folder_name": config.archive_config.archive_folder_name,
                "timestamp_format": config.archive_config.timestamp_format,
                "handle_conflicts": config.archive_config.handle_conflicts,
                "preserve_structure": config.archive_config.preserve_structure,
            },
        }
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        logger.debug("Configuration cache cleared")


# Global configuration manager instance
config_manager = ConfigManager()