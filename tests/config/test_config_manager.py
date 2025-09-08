"""Unit tests for configuration manager."""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import yaml

from excel_to_csv.config.config_manager import ConfigManager, ConfigurationError
from excel_to_csv.models.data_models import Config


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_init(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        assert config_manager._config_cache == {}
        assert hasattr(config_manager, 'DEFAULT_CONFIG')
        assert hasattr(config_manager, 'ENV_PREFIX')
    
    def test_load_config_with_defaults(self):
        """Test loading configuration with default values."""
        config_manager = ConfigManager()
        config = config_manager.load_config(None, use_env_overrides=False)
        
        assert isinstance(config, Config)
        # Should load config/default.yaml if it exists, otherwise built-in defaults
        # The default.yaml has threshold 0.8, built-in defaults have 0.7
        assert config.confidence_threshold in [0.7, 0.8]
        assert config.max_concurrent == 5
        assert len(config.monitored_folders) > 0
        assert len(config.file_patterns) > 0
    
    def test_load_config_from_file(self, sample_config_file: Path):
        """Test loading configuration from YAML file."""
        config_manager = ConfigManager()
        config = config_manager.load_config(sample_config_file, use_env_overrides=False)
        
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.9
        assert config.max_concurrent == 5
        assert any(str(folder).endswith('input') for folder in config.monitored_folders)
        assert any(str(folder).endswith('data') for folder in config.monitored_folders)
    
    def test_load_config_nonexistent_file(self, temp_dir: Path):
        """Test loading configuration from non-existent file falls back to defaults."""
        config_manager = ConfigManager()
        nonexistent_file = temp_dir / "nonexistent.yaml"
        
        config = config_manager.load_config(nonexistent_file, use_env_overrides=False)
        
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.7  # Built-in default value
    
    def test_load_config_invalid_yaml(self, temp_dir: Path):
        """Test loading configuration from invalid YAML file raises error."""
        config_manager = ConfigManager()
        invalid_yaml_file = temp_dir / "invalid.yaml"
        invalid_yaml_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            config_manager.load_config(invalid_yaml_file)
    
    def test_environment_variable_overrides(self, sample_config_file: Path, env_override):
        """Test environment variable overrides."""
        config_manager = ConfigManager()
        
        # Set environment overrides
        env_override.set("EXCEL_TO_CSV_CONFIDENCE_THRESHOLD", "0.85")
        env_override.set("EXCEL_TO_CSV_MAX_CONCURRENT", "10")
        env_override.set("EXCEL_TO_CSV_LOG_LEVEL", "DEBUG")
        env_override.set("EXCEL_TO_CSV_MONITORED_FOLDERS", "./test1,./test2")
        
        config = config_manager.load_config(sample_config_file, use_env_overrides=True)
        
        assert config.confidence_threshold == 0.85
        assert config.max_concurrent == 10
        assert config.logging.level == "DEBUG"
        assert len(config.monitored_folders) == 2
        assert any(str(folder).endswith('test1') for folder in config.monitored_folders)
        assert any(str(folder).endswith('test2') for folder in config.monitored_folders)
    
    def test_environment_variable_type_conversion(self, env_override):
        """Test environment variable type conversion."""
        config_manager = ConfigManager()
        
        # Test boolean conversion
        env_override.set("EXCEL_TO_CSV_INCLUDE_TIMESTAMP", "false")
        
        # Test float conversion
        env_override.set("EXCEL_TO_CSV_CONFIDENCE_THRESHOLD", "0.75")
        
        # Test integer conversion
        env_override.set("EXCEL_TO_CSV_MAX_CONCURRENT", "3")
        
        config = config_manager.load_config(use_env_overrides=True)
        
        assert config.output_config.include_timestamp is False
        assert config.confidence_threshold == 0.75
        assert config.max_concurrent == 3
    
    def test_config_validation_success(self, temp_dir: Path):
        """Test successful configuration validation."""
        config_manager = ConfigManager()
        
        # Create directories that will be validated
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        config_dict = {
            "monitoring": {"folders": [str(input_dir)]},
            "output": {"folder": str(output_dir)},
            "logging": {"file": {"path": str(temp_dir / "test.log")}},
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Should not raise an exception
        config = config_manager.load_config(config_file)
        assert isinstance(config, Config)
    
    def test_config_validation_missing_folders(self, temp_dir: Path):
        """Test configuration validation with missing folders."""
        config_manager = ConfigManager()
        
        nonexistent_dir = temp_dir / "nonexistent"
        config_dict = {
            "monitoring": {"folders": [str(nonexistent_dir)]},
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Should create the directory and succeed
        config = config_manager.load_config(config_file)
        assert isinstance(config, Config)
        assert nonexistent_dir.exists()
    
    def test_config_validation_permission_error(self, temp_dir: Path):
        """Test configuration validation with permission errors.""" 
        from unittest.mock import patch
        config_manager = ConfigManager()
        
        # Use a non-existent folder that will be mocked to fail
        restricted_dir = temp_dir / "nonexistent_folder"
        
        config_dict = {
            "monitoring": {"folders": [str(restricted_dir)]},
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Mock Path.mkdir to raise PermissionError
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            # Should raise ConfigurationError due to permission issues
            with pytest.raises(ConfigurationError, match="Cannot create monitored folder"):
                config_manager.load_config(config_file)
    
    def test_deep_merge(self):
        """Test deep merge functionality."""
        config_manager = ConfigManager()
        
        base = {
            "monitoring": {"folders": ["./base"], "patterns": ["*.xlsx"]},
            "confidence": {"threshold": 0.9},
        }
        
        override = {
            "monitoring": {"folders": ["./override"]},
            "output": {"folder": "./out"},
        }
        
        result = config_manager._deep_merge(base, override)
        
        assert result["monitoring"]["folders"] == ["./override"]
        assert result["monitoring"]["patterns"] == ["*.xlsx"]  # Preserved from base
        assert result["confidence"]["threshold"] == 0.9  # Preserved from base
        assert result["output"]["folder"] == "./out"  # Added from override
    
    def test_convert_env_value(self):
        """Test environment value conversion."""
        config_manager = ConfigManager()
        
        # Test boolean conversion
        assert config_manager._convert_env_value("true", ["test"]) is True
        assert config_manager._convert_env_value("false", ["test"]) is False
        assert config_manager._convert_env_value("True", ["test"]) is True
        assert config_manager._convert_env_value("FALSE", ["test"]) is False
        
        # Test integer conversion
        assert config_manager._convert_env_value("42", ["test"]) == 42
        
        # Test float conversion
        assert config_manager._convert_env_value("3.14", ["test"]) == 3.14
        
        # Test string fallback
        assert config_manager._convert_env_value("hello", ["test"]) == "hello"
    
    def test_set_nested_value(self):
        """Test setting nested dictionary values."""
        config_manager = ConfigManager()
        
        dictionary = {}
        config_manager._set_nested_value(dictionary, ["level1", "level2", "key"], "value")
        
        assert dictionary["level1"]["level2"]["key"] == "value"
    
    def test_dict_to_config(self, sample_config_dict: dict):
        """Test conversion from dictionary to Config object."""
        config_manager = ConfigManager()
        
        config = config_manager._dict_to_config(sample_config_dict)
        
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.9
        assert config.max_concurrent == 5
        assert len(config.monitored_folders) == 2
        assert config.logging.level == "INFO"
        assert config.output_config.encoding == "utf-8"
    
    def test_config_to_dict(self, sample_config_file: Path):
        """Test conversion from Config object to dictionary."""
        config_manager = ConfigManager()
        
        # Load config from file
        config = config_manager.load_config(sample_config_file, use_env_overrides=False)
        
        # Convert back to dictionary
        config_dict = config_manager._config_to_dict(config)
        
        assert isinstance(config_dict, dict)
        assert "monitoring" in config_dict
        assert "confidence" in config_dict
        assert "output" in config_dict
        assert "processing" in config_dict
        assert "logging" in config_dict
    
    def test_save_config(self, temp_dir: Path, sample_config_file: Path):
        """Test saving configuration to file."""
        config_manager = ConfigManager()
        
        # Load config
        config = config_manager.load_config(sample_config_file, use_env_overrides=False)
        
        # Save to new location
        new_config_file = temp_dir / "saved_config.yaml"
        config_manager.save_config(config, new_config_file)
        
        assert new_config_file.exists()
        
        # Load saved config and verify
        loaded_config = config_manager.load_config(new_config_file, use_env_overrides=False)
        assert loaded_config.confidence_threshold == config.confidence_threshold
        assert loaded_config.max_concurrent == config.max_concurrent
    
    def test_config_caching(self, sample_config_file: Path):
        """Test configuration caching functionality."""
        config_manager = ConfigManager()
        
        # Clear cache
        config_manager.clear_cache()
        assert len(config_manager._config_cache) == 0
        
        # Load config twice
        config1 = config_manager.load_config(sample_config_file)
        config2 = config_manager.load_config(sample_config_file)
        
        # Should be the same object (cached)
        assert config1 is config2
        assert len(config_manager._config_cache) == 1
        
        # Clear cache and load again
        config_manager.clear_cache()
        config3 = config_manager.load_config(sample_config_file)
        
        # Should be different object (cache cleared)
        assert config1 is not config3
    
    def test_invalid_config_structure(self, temp_dir: Path):
        """Test handling of invalid configuration structure."""
        config_manager = ConfigManager()
        
        # Create config with missing required sections
        invalid_config = {"invalid_key": "invalid_value"}
        
        config_file = temp_dir / "invalid_structure.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should still work with defaults
        config = config_manager.load_config(config_file)
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.7  # Should use built-in defaults when config file is invalid
    
    def test_edge_case_empty_config_file(self, temp_dir: Path):
        """Test handling of empty configuration file."""
        config_manager = ConfigManager()
        
        empty_config_file = temp_dir / "empty.yaml"
        empty_config_file.write_text("")
        
        # Should work with defaults
        config = config_manager.load_config(empty_config_file)
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.7  # Built-in default value
    
    def test_config_with_none_values(self, temp_dir: Path):
        """Test configuration with None values."""
        config_manager = ConfigManager()
        
        config_dict = {
            "output": {"folder": None},
            "confidence": {"threshold": None},
        }
        
        config_file = temp_dir / "none_values.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Should raise ConfigurationError for None values that aren't supported
        with pytest.raises(ConfigurationError):
            config_manager.load_config(config_file)


class TestGlobalConfigManager:
    """Test cases for global config_manager instance."""
    
    def test_global_instance_exists(self):
        """Test that global config_manager instance exists."""
        from excel_to_csv.config.config_manager import config_manager
        
        assert config_manager is not None
        assert isinstance(config_manager, ConfigManager)
    
    def test_global_instance_functionality(self, sample_config_file: Path):
        """Test that global config_manager instance works correctly."""
        from excel_to_csv.config.config_manager import config_manager
        
        # Clear cache to ensure clean test
        config_manager.clear_cache()
        
        config = config_manager.load_config(sample_config_file)
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.9
    
    def test_archiving_configuration_loading(self):
        """Test loading archiving configuration from defaults."""
        config_manager = ConfigManager()
        config = config_manager.load_config(use_env_overrides=False)
        
        assert hasattr(config, 'archive_config')
        assert config.archive_config is not None
        assert hasattr(config.archive_config, 'enabled')
        assert hasattr(config.archive_config, 'archive_folder_name')
        assert hasattr(config.archive_config, 'timestamp_format')
        assert hasattr(config.archive_config, 'handle_conflicts')
        assert hasattr(config.archive_config, 'preserve_structure')
        
        # Default archiving should be disabled
        assert config.archive_config.enabled is False
        assert config.archive_config.archive_folder_name == "archive"
        assert config.archive_config.timestamp_format == "%Y%m%d_%H%M%S"
        assert config.archive_config.handle_conflicts is True
        assert config.archive_config.preserve_structure is True
    
    def test_archiving_environment_variable_overrides(self, env_override):
        """Test archiving environment variable overrides."""
        config_manager = ConfigManager()
        
        # Set archiving environment overrides
        env_override.set("EXCEL_TO_CSV_ARCHIVE_ENABLED", "true")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_FOLDER_NAME", "processed")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_TIMESTAMP_FORMAT", "%Y-%m-%d_%H%M%S")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_HANDLE_CONFLICTS", "false")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_PRESERVE_STRUCTURE", "false")
        
        config = config_manager.load_config(use_env_overrides=True)
        
        assert config.archive_config.enabled is True
        assert config.archive_config.archive_folder_name == "processed"
        assert config.archive_config.timestamp_format == "%Y-%m-%d_%H%M%S"
        assert config.archive_config.handle_conflicts is False
        assert config.archive_config.preserve_structure is False
    
    def test_archiving_yaml_configuration(self, temp_dir: Path):
        """Test loading archiving configuration from YAML file."""
        config_manager = ConfigManager()
        
        config_dict = {
            "archiving": {
                "enabled": True,
                "archive_folder_name": "backup",
                "timestamp_format": "%Y%m%d",
                "handle_conflicts": False,
                "preserve_structure": False
            }
        }
        
        config_file = temp_dir / "archiving_test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        config = config_manager.load_config(config_file, use_env_overrides=False)
        
        assert config.archive_config.enabled is True
        assert config.archive_config.archive_folder_name == "backup"
        assert config.archive_config.timestamp_format == "%Y%m%d"
        assert config.archive_config.handle_conflicts is False
        assert config.archive_config.preserve_structure is False
    
    def test_archiving_config_validation(self, temp_dir: Path):
        """Test archiving configuration validation."""
        config_manager = ConfigManager()
        
        # Test valid archiving config
        valid_config_dict = {
            "archiving": {
                "enabled": True,
                "archive_folder_name": "valid_archive",
                "timestamp_format": "%Y%m%d_%H%M%S",
                "handle_conflicts": True,
                "preserve_structure": True
            }
        }
        
        valid_config_file = temp_dir / "valid_archiving.yaml"
        with open(valid_config_file, 'w') as f:
            yaml.dump(valid_config_dict, f)
        
        config = config_manager.load_config(valid_config_file, use_env_overrides=False)
        assert isinstance(config, Config)
        assert config.archive_config.enabled is True
        assert config.archive_config.archive_folder_name == "valid_archive"