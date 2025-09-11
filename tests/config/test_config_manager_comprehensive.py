"""Comprehensive tests for Config Manager with high coverage."""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import yaml
import logging

from excel_to_csv.config.config_manager import ConfigManager, ConfigurationError
from excel_to_csv.models.data_models import (
    Config, 
    ArchiveConfig, 
    LoggingConfig, 
    OutputConfig, 
    RetryConfig,
    ConfidenceConfig
)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config_yaml():
    """Sample configuration YAML content."""
    return """
monitoring:
  folders:
    - ./test_input
    - ./test_input2
  file_patterns:
    - "*.xlsx"
    - "*.xls"
  process_existing: true
  polling_interval: 10
  max_file_size: 200

confidence:
  threshold: 0.85
  weights:
    data_density: 0.5
    header_quality: 0.3
    consistency: 0.2
  min_rows: 10
  min_columns: 3
  max_empty_percentage: 0.2

output:
  folder: ./test_output
  naming_pattern: "test_{filename}_{worksheet}.csv"
  include_timestamp: false
  encoding: "utf-16"
  delimiter: ";"
  include_headers: false
  timestamp_format: "%Y-%m-%d_%H-%M-%S"

processing:
  max_concurrent: 10
  retry:
    max_attempts: 5
    delay: 10
    backoff_factor: 3
    max_delay: 120
  timeouts:
    file_processing: 600
    worksheet_analysis: 120

logging:
  level: DEBUG
  format: "%(levelname)s: %(message)s"
  file:
    enabled: false
    path: ./test_logs/test.log
    max_size: 20
    backup_count: 10
    rotation: false
  console:
    enabled: false
    level: DEBUG
  structured:
    enabled: true
    path: ./test_logs/test.json

archiving:
  enabled: false
  archive_folder_name: "test_archive"
  timestamp_format: "%Y%m%d_%H%M%S_%f"
  handle_conflicts: false
  preserve_structure: false
"""


@pytest.fixture
def invalid_yaml():
    """Invalid YAML content for testing error handling."""
    return """
invalid_yaml:
  - item1
  - item2:
    - nested_incorrectly: {
"""


class TestConfigManagerInitialization:
    """Test ConfigManager initialization and basic properties."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        
        assert manager is not None
        assert hasattr(manager, '_config_cache')
        assert isinstance(manager._config_cache, dict)
        assert len(manager._config_cache) == 0
        
        # Check constants
        assert manager.ENV_PREFIX == "EXCEL_TO_CSV_"
        assert isinstance(manager.DEFAULT_CONFIG, dict)
        assert "monitoring" in manager.DEFAULT_CONFIG
        assert "confidence" in manager.DEFAULT_CONFIG
        assert "output" in manager.DEFAULT_CONFIG
    
    def test_default_config_structure(self):
        """Test default configuration structure."""
        manager = ConfigManager()
        
        # Check all required sections exist
        required_sections = [
            "monitoring", "confidence", "output", "processing",
            "logging", "performance", "security", "archiving"
        ]
        
        for section in required_sections:
            assert section in manager.DEFAULT_CONFIG
        
        # Check specific values
        assert manager.DEFAULT_CONFIG["archiving"]["enabled"] is True
        assert manager.DEFAULT_CONFIG["confidence"]["min_columns"] == 2
        assert manager.DEFAULT_CONFIG["output"]["encoding"] == "utf-8"


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_config_with_defaults(self):
        """Test loading configuration with built-in defaults."""
        manager = ConfigManager()
        
        with patch.object(Path, 'exists', return_value=False):
            config = manager.load_config(config_path=None)
        
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.7  # From defaults
        assert config.max_concurrent == 5
        assert len(config.monitored_folders) == 1
        assert str(config.monitored_folders[0]) == "input"
    
    def test_load_config_from_file(self, temp_workspace, sample_config_yaml):
        """Test loading configuration from YAML file."""
        config_file = temp_workspace / "test_config.yaml"
        config_file.write_text(sample_config_yaml, encoding='utf-8')
        
        manager = ConfigManager()
        config = manager.load_config(config_path=config_file)
        
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.85
        assert config.max_concurrent == 10
        assert len(config.monitored_folders) == 2
        assert str(config.monitored_folders[0]) == "test_input"
        assert config.output_config.encoding == "utf-16"
        assert config.output_config.delimiter == ";"
    
    def test_load_config_auto_default_yaml(self, temp_workspace, sample_config_yaml):
        """Test auto-loading config/default.yaml when no path provided."""
        # Create config/default.yaml in current directory
        config_dir = Path.cwd() / "config"
        config_dir.mkdir(exist_ok=True)
        default_config = config_dir / "default.yaml"
        
        try:
            default_config.write_text(sample_config_yaml, encoding='utf-8')
            
            manager = ConfigManager()
            config = manager.load_config(config_path=None)
            
            assert isinstance(config, Config)
            assert config.confidence_threshold == 0.85
            
        finally:
            # Cleanup
            if default_config.exists():
                default_config.unlink()
            if config_dir.exists() and not any(config_dir.iterdir()):
                config_dir.rmdir()
    
    def test_load_config_caching(self, temp_workspace, sample_config_yaml):
        """Test configuration caching functionality."""
        config_file = temp_workspace / "test_config.yaml"
        config_file.write_text(sample_config_yaml, encoding='utf-8')
        
        manager = ConfigManager()
        
        # Load first time
        config1 = manager.load_config(config_path=config_file)
        assert len(manager._config_cache) == 1
        
        # Load second time (should use cache)
        config2 = manager.load_config(config_path=config_file)
        assert config1 is config2  # Same object reference
        assert len(manager._config_cache) == 1
        
        # Load with different env setting (new cache entry)
        config3 = manager.load_config(config_path=config_file, use_env_overrides=False)
        assert len(manager._config_cache) == 2
    
    def test_load_config_nonexistent_file(self):
        """Test loading configuration from non-existent file."""
        manager = ConfigManager()
        nonexistent_path = "/nonexistent/path/config.yaml"
        
        # Should fall back to defaults
        config = manager.load_config(config_path=nonexistent_path)
        assert isinstance(config, Config)
        assert config.confidence_threshold == 0.7  # From defaults


class TestConfigurationErrors:
    """Test error handling in configuration loading."""
    
    def test_invalid_yaml_syntax(self, temp_workspace, invalid_yaml):
        """Test handling of invalid YAML syntax."""
        config_file = temp_workspace / "invalid.yaml"
        config_file.write_text(invalid_yaml, encoding='utf-8')
        
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            manager.load_config(config_path=config_file)
    
    def test_permission_error_reading_config(self, temp_workspace, sample_config_yaml):
        """Test handling of permission errors when reading config."""
        config_file = temp_workspace / "readonly.yaml"
        config_file.write_text(sample_config_yaml, encoding='utf-8')
        
        manager = ConfigManager()
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(ConfigurationError, match="Cannot read"):
                manager.load_config(config_path=config_file)
    
    def test_general_exception_handling(self, temp_workspace):
        """Test handling of general exceptions during config loading."""
        config_file = temp_workspace / "test.yaml"
        config_file.write_text("valid: yaml", encoding='utf-8')
        
        manager = ConfigManager()
        
        # Mock unexpected exception during dict_to_config
        with patch.object(manager, '_dict_to_config', side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(ConfigurationError, match="Failed to load configuration"):
                manager.load_config(config_path=config_file)


class TestEnvironmentOverrides:
    """Test environment variable override functionality."""
    
    def test_env_overrides_basic(self, temp_workspace):
        """Test basic environment variable overrides."""
        config_file = temp_workspace / "base.yaml"
        config_file.write_text("confidence:\n  threshold: 0.5\n", encoding='utf-8')
        
        manager = ConfigManager()
        
        # Set environment variables
        env_vars = {
            "EXCEL_TO_CSV_CONFIDENCE_THRESHOLD": "0.9",
            "EXCEL_TO_CSV_OUTPUT_FOLDER": "/custom/output",
            "EXCEL_TO_CSV_LOG_LEVEL": "DEBUG",
            "EXCEL_TO_CSV_MAX_CONCURRENT": "15",
        }
        
        with patch.dict(os.environ, env_vars):
            config = manager.load_config(config_path=config_file)
        
        assert config.confidence_threshold == 0.9
        assert str(config.output_folder) == "/custom/output"
        assert config.logging.level == "DEBUG"
        assert config.max_concurrent == 15
    
    def test_env_overrides_boolean_conversion(self, temp_workspace):
        """Test boolean conversion in environment overrides."""
        config_file = temp_workspace / "base.yaml"
        config_file.write_text("archiving:\n  enabled: false\n", encoding='utf-8')
        
        manager = ConfigManager()
        
        env_vars = {
            "EXCEL_TO_CSV_ARCHIVE_ENABLED": "true",
            "EXCEL_TO_CSV_INCLUDE_TIMESTAMP": "false",
        }
        
        with patch.dict(os.environ, env_vars):
            config = manager.load_config(config_path=config_file)
        
        assert config.archive_config.enabled is True
        assert config.output_config.include_timestamp is False
    
    def test_env_overrides_monitored_folders(self, temp_workspace):
        """Test monitored folders environment override."""
        config_file = temp_workspace / "base.yaml"
        config_file.write_text("monitoring:\n  folders:\n    - ./default\n", encoding='utf-8')
        
        manager = ConfigManager()
        
        env_vars = {
            "EXCEL_TO_CSV_MONITORED_FOLDERS": "/folder1,/folder2, /folder3"
        }
        
        with patch.dict(os.environ, env_vars):
            config = manager.load_config(config_path=config_file)
        
        folder_paths = [str(f) for f in config.monitored_folders]
        assert "/folder1" in folder_paths
        assert "/folder2" in folder_paths
        assert "/folder3" in folder_paths
    
    def test_env_overrides_disabled(self, temp_workspace):
        """Test disabling environment overrides."""
        config_file = temp_workspace / "base.yaml"
        config_file.write_text("confidence:\n  threshold: 0.5\n", encoding='utf-8')
        
        manager = ConfigManager()
        
        env_vars = {
            "EXCEL_TO_CSV_CONFIDENCE_THRESHOLD": "0.9"
        }
        
        with patch.dict(os.environ, env_vars):
            config = manager.load_config(config_path=config_file, use_env_overrides=False)
        
        # Should use file value, not env override
        assert config.confidence_threshold == 0.5


class TestConfigValidation:
    """Test configuration validation functionality."""
    
    def test_validation_creates_missing_folders(self, temp_workspace):
        """Test that validation creates missing monitored and output folders."""
        config_file = temp_workspace / "test.yaml"
        
        # Create config with non-existent folders
        missing_input = temp_workspace / "missing_input"
        missing_output = temp_workspace / "missing_output"
        missing_logs = temp_workspace / "missing_logs"
        
        config_yaml = f"""
monitoring:
  folders:
    - {missing_input}
output:
  folder: {missing_output}
logging:
  file:
    enabled: true
    path: {missing_logs}/test.log
"""
        
        config_file.write_text(config_yaml, encoding='utf-8')
        
        manager = ConfigManager()
        config = manager.load_config(config_path=config_file)
        
        # Folders should be created
        assert missing_input.exists()
        assert missing_output.exists()
        assert missing_logs.exists()
    
    def test_validation_permission_error_monitored_folder(self, temp_workspace):
        """Test validation error when cannot create monitored folder."""
        config_file = temp_workspace / "test.yaml"
        
        config_yaml = """
monitoring:
  folders:
    - /proc/sys/kernel/read_only_folder
"""
        config_file.write_text(config_yaml, encoding='utf-8')
        
        manager = ConfigManager()
        
        # Mock the mkdir to simulate permission error
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(ConfigurationError, match="Cannot create monitored folder"):
                manager.load_config(config_path=config_file)
    


class TestConfigurationSaving:
    """Test configuration saving functionality."""
    
    def test_save_config_success(self, temp_workspace):
        """Test successful configuration saving."""
        manager = ConfigManager()
        config = manager.load_config(config_path=None)  # Load defaults
        
        save_path = temp_workspace / "saved_config.yaml"
        manager.save_config(config, save_path)
        
        assert save_path.exists()
        
        # Verify content by loading it back
        loaded_config = manager.load_config(config_path=save_path)
        assert isinstance(loaded_config, Config)
        assert loaded_config.confidence_threshold == config.confidence_threshold
    
    def test_save_config_creates_directory(self, temp_workspace):
        """Test that save_config creates parent directories."""
        manager = ConfigManager()
        config = manager.load_config(config_path=None)
        
        nested_path = temp_workspace / "nested" / "deep" / "config.yaml"
        manager.save_config(config, nested_path)
        
        assert nested_path.exists()
        assert nested_path.parent.exists()
    
    def test_save_config_permission_error(self, temp_workspace):
        """Test save_config error handling."""
        manager = ConfigManager()
        config = manager.load_config(config_path=None)
        
        # Mock permission error
        with patch('builtins.open', side_effect=PermissionError("Cannot write")):
            with pytest.raises(ConfigurationError, match="Failed to save configuration"):
                manager.save_config(config, temp_workspace / "readonly.yaml")


class TestUtilityMethods:
    """Test utility methods of ConfigManager."""
    
    def test_deep_merge_simple(self):
        """Test simple dictionary deep merge."""
        manager = ConfigManager()
        
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"d": 3}, "e": 4}
        
        result = manager._deep_merge(base, override)
        
        assert result["a"] == 1
        assert result["b"]["c"] == 2
        assert result["b"]["d"] == 3
        assert result["e"] == 4
    
    def test_deep_merge_override_values(self):
        """Test deep merge with value overrides."""
        manager = ConfigManager()
        
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"a": 10, "b": {"c": 20}}
        
        result = manager._deep_merge(base, override)
        
        assert result["a"] == 10  # Override
        assert result["b"]["c"] == 20  # Override
        assert result["b"]["d"] == 3  # Preserved
    
    def test_convert_env_value_types(self):
        """Test environment value type conversion."""
        manager = ConfigManager()
        
        # Boolean conversions
        assert manager._convert_env_value("true", []) is True
        assert manager._convert_env_value("false", []) is False
        assert manager._convert_env_value("True", []) is True
        assert manager._convert_env_value("FALSE", []) is False
        
        # Integer conversions
        assert manager._convert_env_value("42", []) == 42
        assert manager._convert_env_value("-10", []) == -10
        
        # Float conversions
        assert manager._convert_env_value("3.14", []) == 3.14
        assert manager._convert_env_value("-2.5", []) == -2.5
        
        # String (fallback)
        assert manager._convert_env_value("hello", []) == "hello"
        assert manager._convert_env_value("not-a-number", []) == "not-a-number"
    
    def test_set_nested_value(self):
        """Test setting nested dictionary values."""
        manager = ConfigManager()
        
        dictionary = {}
        manager._set_nested_value(dictionary, ["a", "b", "c"], "value")
        
        assert dictionary["a"]["b"]["c"] == "value"
        
        # Test setting another value in same nested structure
        manager._set_nested_value(dictionary, ["a", "b", "d"], "another_value")
        assert dictionary["a"]["b"]["d"] == "another_value"
        assert dictionary["a"]["b"]["c"] == "value"  # Preserved
    
    def test_config_to_dict_conversion(self):
        """Test converting Config object back to dictionary."""
        manager = ConfigManager()
        config = manager.load_config(config_path=None)
        
        config_dict = manager._config_to_dict(config)
        
        assert isinstance(config_dict, dict)
        assert "monitoring" in config_dict
        assert "confidence" in config_dict
        assert "output" in config_dict
        assert "archiving" in config_dict
        
        # Check some specific values
        assert config_dict["confidence"]["threshold"] == config.confidence_threshold
        assert config_dict["processing"]["max_concurrent"] == config.max_concurrent
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        manager = ConfigManager()
        
        # Load a config to populate cache
        config = manager.load_config(config_path=None)
        assert len(manager._config_cache) == 1
        
        # Clear cache
        manager.clear_cache()
        assert len(manager._config_cache) == 0


class TestGlobalConfigManager:
    """Test global config manager instance."""
    
    def test_global_instance_import(self):
        """Test that global config manager instance can be imported."""
        from excel_to_csv.config.config_manager import config_manager
        
        assert isinstance(config_manager, ConfigManager)
        assert hasattr(config_manager, 'load_config')
        assert hasattr(config_manager, 'save_config')


if __name__ == "__main__":
    pytest.main([__file__])