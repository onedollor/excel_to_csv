"""Comprehensive unit tests for Config Manager.

This test suite provides comprehensive coverage for:
- Configuration loading and validation
- YAML file parsing
- Environment variable overrides  
- Error handling scenarios
- Configuration caching
- File I/O operations
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any

import yaml

from excel_to_csv.config.config_manager import ConfigManager, ConfigurationError
from excel_to_csv.models.data_models import Config


class TestConfigManager:
    """Comprehensive test suite for ConfigManager."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager instance for testing."""
        manager = ConfigManager()
        manager.clear_cache()  # Ensure clean state
        return manager

    @pytest.fixture
    def mock_file_system(self):
        """Mock file system operations for configuration testing."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.stat') as mock_stat, \
             patch('builtins.open', mock_open()) as mock_file:
            
            # Default file exists behavior
            mock_exists.return_value = True
            mock_mkdir.return_value = None
            
            # Mock file stats
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = 1024
            mock_stat.return_value = mock_stat_obj
            
            yield {
                'exists': mock_exists,
                'mkdir': mock_mkdir,
                'stat': mock_stat,
                'open': mock_file
            }

    @pytest.fixture
    def invalid_yaml_content(self):
        """Sample invalid YAML content for error testing."""
        return """
        monitoring:
          folders: [
            invalid yaml syntax missing bracket
          file_patterns: ["*.xlsx"]
        """

    @pytest.fixture
    def valid_config_content(self, sample_config_dict):
        """Valid YAML configuration content."""
        return yaml.dump(sample_config_dict)

    def test_init(self, config_manager):
        """Test ConfigManager initialization."""
        assert config_manager is not None
        assert config_manager._config_cache == {}
        assert config_manager.ENV_PREFIX == "EXCEL_TO_CSV_"
        assert isinstance(config_manager.DEFAULT_CONFIG, dict)

    def test_default_config_structure(self, config_manager):
        """Test default configuration has required structure."""
        default_config = config_manager.DEFAULT_CONFIG
        
        # Check all required top-level sections exist
        required_sections = {
            'monitoring', 'confidence', 'output', 'processing', 
            'logging', 'performance', 'security', 'archiving'
        }
        assert set(default_config.keys()) == required_sections
        
        # Check monitoring section
        monitoring = default_config['monitoring']
        assert 'folders' in monitoring
        assert 'file_patterns' in monitoring
        assert 'process_existing' in monitoring
        assert 'max_file_size' in monitoring
        
        # Check confidence section
        confidence = default_config['confidence']
        assert 'threshold' in confidence
        assert 'weights' in confidence
        assert 'min_rows' in confidence
        
        # Check output section
        output = default_config['output']
        assert 'folder' in output
        assert 'naming_pattern' in output
        assert 'encoding' in output

    def test_config_cache_key_generation(self, config_manager):
        """Test configuration cache key generation."""
        # Test with path and env overrides
        key1 = "config.yaml:True"
        key2 = "config.yaml:False"
        key3 = "None:True"
        
        # Keys should be different for different combinations
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_clear_cache(self, config_manager):
        """Test configuration cache clearing."""
        # Add something to cache
        config_manager._config_cache["test_key"] = Mock()
        assert len(config_manager._config_cache) == 1
        
        # Clear cache
        config_manager.clear_cache()
        assert len(config_manager._config_cache) == 0

    @patch('pathlib.Path.exists')
    def test_load_config_default_path_not_found(self, mock_exists, config_manager):
        """Test loading config when default path doesn't exist."""
        # Mock that config/default.yaml doesn't exist
        mock_exists.return_value = False
        
        # Should use built-in defaults
        config = config_manager.load_config()
        
        assert config is not None
        assert isinstance(config, Config)
        # Should have default values
        assert config.confidence_threshold == 0.7  # From DEFAULT_CONFIG

    @patch('pathlib.Path.exists')
    @patch('builtins.open', mock_open())
    def test_load_config_caching(self, mock_exists, config_manager, valid_config_content):
        """Test configuration caching behavior."""
        mock_exists.return_value = False  # Use defaults
        
        # First load
        config1 = config_manager.load_config()
        
        # Second load should use cache
        config2 = config_manager.load_config()
        
        # Should be the same object from cache
        assert config1 is config2

    def test_deep_merge_functionality(self, config_manager):
        """Test deep merge functionality for configuration dictionaries."""
        base = {
            'level1': {
                'level2': {
                    'key1': 'base_value1',
                    'key2': 'base_value2'
                },
                'other_key': 'base_other'
            },
            'top_key': 'base_top'
        }
        
        override = {
            'level1': {
                'level2': {
                    'key1': 'override_value1',
                    'key3': 'override_value3'
                },
                'new_key': 'override_new'
            },
            'new_top_key': 'override_new_top'
        }
        
        result = config_manager._deep_merge(base, override)
        
        # Check deep merge worked correctly
        assert result['level1']['level2']['key1'] == 'override_value1'  # Overridden
        assert result['level1']['level2']['key2'] == 'base_value2'     # Preserved
        assert result['level1']['level2']['key3'] == 'override_value3'  # Added
        assert result['level1']['other_key'] == 'base_other'           # Preserved
        assert result['level1']['new_key'] == 'override_new'           # Added
        assert result['top_key'] == 'base_top'                         # Preserved
        assert result['new_top_key'] == 'override_new_top'             # Added

    def test_set_nested_value(self, config_manager):
        """Test setting nested dictionary values."""
        dictionary = {}
        
        # Test setting nested value
        config_manager._set_nested_value(dictionary, ['level1', 'level2', 'key'], 'value')
        assert dictionary['level1']['level2']['key'] == 'value'
        
        # Test overriding existing value
        config_manager._set_nested_value(dictionary, ['level1', 'level2', 'key'], 'new_value')
        assert dictionary['level1']['level2']['key'] == 'new_value'
        
        # Test setting at existing level
        config_manager._set_nested_value(dictionary, ['level1', 'other_key'], 'other_value')
        assert dictionary['level1']['other_key'] == 'other_value'
        assert dictionary['level1']['level2']['key'] == 'new_value'  # Preserved

    def test_convert_env_value_boolean(self, config_manager):
        """Test environment value conversion for booleans."""
        # Test true values
        assert config_manager._convert_env_value('true', ['test', 'path']) is True
        assert config_manager._convert_env_value('True', ['test', 'path']) is True
        assert config_manager._convert_env_value('TRUE', ['test', 'path']) is True
        
        # Test false values
        assert config_manager._convert_env_value('false', ['test', 'path']) is False
        assert config_manager._convert_env_value('False', ['test', 'path']) is False
        assert config_manager._convert_env_value('FALSE', ['test', 'path']) is False

    def test_convert_env_value_numeric(self, config_manager):
        """Test environment value conversion for numbers."""
        # Test integers
        assert config_manager._convert_env_value('42', ['test', 'path']) == 42
        assert config_manager._convert_env_value('0', ['test', 'path']) == 0
        assert config_manager._convert_env_value('-10', ['test', 'path']) == -10
        
        # Test floats
        assert config_manager._convert_env_value('3.14', ['test', 'path']) == 3.14
        assert config_manager._convert_env_value('0.0', ['test', 'path']) == 0.0
        assert config_manager._convert_env_value('-2.5', ['test', 'path']) == -2.5

    def test_convert_env_value_string(self, config_manager):
        """Test environment value conversion for strings."""
        # Test regular strings
        assert config_manager._convert_env_value('hello', ['test', 'path']) == 'hello'
        assert config_manager._convert_env_value('world', ['test', 'path']) == 'world'
        
        # Test strings that might look like numbers but aren't valid
        assert config_manager._convert_env_value('not_a_number', ['test', 'path']) == 'not_a_number'
        assert config_manager._convert_env_value('3.14.15', ['test', 'path']) == '3.14.15'
        
        # Test empty string
        assert config_manager._convert_env_value('', ['test', 'path']) == ''

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError is properly defined."""
        error = ConfigurationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    @patch.dict(os.environ, {}, clear=True)
    def test_clean_environment_state(self, config_manager):
        """Test that tests start with clean environment."""
        # Verify no EXCEL_TO_CSV_ variables exist
        env_vars = [key for key in os.environ.keys() if key.startswith('EXCEL_TO_CSV_')]
        assert len(env_vars) == 0

    def test_fixture_integration(self, config_manager, sample_config_dict, temp_dir):
        """Test integration with existing test fixtures."""
        # Verify fixtures are working
        assert config_manager is not None
        assert isinstance(sample_config_dict, dict)
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        # Test sample_config_dict has expected structure
        assert 'monitoring' in sample_config_dict
        assert 'confidence' in sample_config_dict
        assert 'output' in sample_config_dict


class TestConfigManagerYAMLLoading:
    """Test suite for YAML configuration loading functionality."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager instance for testing."""
        manager = ConfigManager()
        manager.clear_cache()
        return manager

    @pytest.fixture
    def yaml_config_content(self, sample_config_dict):
        """Valid YAML configuration content."""
        return yaml.dump(sample_config_dict)

    def test_load_config_dict_with_valid_path(self, config_manager, temp_dir, yaml_config_content):
        """Test loading configuration dictionary from valid YAML file."""
        # Create a valid config file
        config_file = temp_dir / "test_config.yaml"
        config_file.write_text(yaml_config_content)
        
        # Load configuration
        config_dict = config_manager._load_config_dict(config_file)
        
        # Verify configuration loaded correctly
        assert isinstance(config_dict, dict)
        assert 'monitoring' in config_dict
        assert 'confidence' in config_dict
        assert 'output' in config_dict
        
        # Verify defaults were merged
        assert 'security' in config_dict  # From defaults
        assert 'archiving' in config_dict  # From defaults

    def test_load_config_dict_file_not_found(self, config_manager, temp_dir):
        """Test loading configuration when file doesn't exist."""
        non_existent_file = temp_dir / "non_existent.yaml"
        
        # Should return default configuration
        config_dict = config_manager._load_config_dict(non_existent_file)
        
        assert isinstance(config_dict, dict)
        # Should be identical to default config
        assert config_dict == config_manager.DEFAULT_CONFIG

    def test_load_config_dict_with_none_path_default_exists(self, config_manager):
        """Test loading with None path when config/default.yaml exists."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data='monitoring:\n  folders: ["./test"]')):
            
            # Mock that config/default.yaml exists
            mock_exists.return_value = True
            
            config_dict = config_manager._load_config_dict(None)
            
            assert isinstance(config_dict, dict)
            # Should have loaded from default file and merged with defaults
            assert 'monitoring' in config_dict

    def test_load_config_dict_with_none_path_no_default(self, config_manager):
        """Test loading with None path when config/default.yaml doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            config_dict = config_manager._load_config_dict(None)
            
            # Should return default configuration
            assert config_dict == config_manager.DEFAULT_CONFIG

    def test_load_config_dict_invalid_yaml_syntax(self, config_manager, temp_dir):
        """Test handling of invalid YAML syntax."""
        invalid_yaml = "monitoring:\n  folders: [\n    invalid yaml"  # Missing closing bracket
        config_file = temp_dir / "invalid.yaml"
        config_file.write_text(invalid_yaml)
        
        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            config_manager._load_config_dict(config_file)
        
        assert "Invalid YAML" in str(exc_info.value)

    def test_load_config_dict_empty_yaml_file(self, config_manager, temp_dir):
        """Test loading empty YAML file."""
        config_file = temp_dir / "empty.yaml"
        config_file.write_text("")
        
        config_dict = config_manager._load_config_dict(config_file)
        
        # Should return default configuration (empty YAML becomes None, merged with defaults)
        assert isinstance(config_dict, dict)
        assert config_dict == config_manager.DEFAULT_CONFIG

    def test_load_config_dict_permission_error(self, config_manager, temp_dir):
        """Test handling of file permission errors."""
        config_file = temp_dir / "restricted.yaml"
        config_file.write_text("test: value")
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._load_config_dict(config_file)
            
            assert "Cannot read" in str(exc_info.value)
            assert "Access denied" in str(exc_info.value)

    def test_load_config_dict_io_error(self, config_manager, temp_dir):
        """Test handling of I/O errors."""
        config_file = temp_dir / "io_error.yaml"
        
        with patch('builtins.open', side_effect=IOError("Disk error")):
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._load_config_dict(config_file)
            
            assert "Cannot read" in str(exc_info.value)
            assert "Disk error" in str(exc_info.value)

    def test_load_config_dict_partial_config(self, config_manager, temp_dir):
        """Test loading partial configuration (only some sections)."""
        partial_yaml = """
        monitoring:
          folders: ["./custom"]
          max_file_size: 200
        confidence:
          threshold: 0.95
        """
        
        config_file = temp_dir / "partial.yaml"
        config_file.write_text(partial_yaml)
        
        config_dict = config_manager._load_config_dict(config_file)
        
        # Should have custom values
        assert config_dict['monitoring']['folders'] == ['./custom']
        assert config_dict['monitoring']['max_file_size'] == 200
        assert config_dict['confidence']['threshold'] == 0.95
        
        # Should have defaults for missing sections
        assert 'output' in config_dict
        assert 'logging' in config_dict
        assert config_dict['output']['encoding'] == 'utf-8'  # From defaults

    def test_load_config_dict_nested_merge(self, config_manager, temp_dir):
        """Test deep merging of nested configuration sections."""
        nested_yaml = """
        confidence:
          threshold: 0.85
          weights:
            data_density: 0.5
            # header_quality and consistency should come from defaults
        output:
          folder: "./custom_output"
          # Other output settings should come from defaults
        """
        
        config_file = temp_dir / "nested.yaml"
        config_file.write_text(nested_yaml)
        
        config_dict = config_manager._load_config_dict(config_file)
        
        # Check custom values
        assert config_dict['confidence']['threshold'] == 0.85
        assert config_dict['confidence']['weights']['data_density'] == 0.5
        assert config_dict['output']['folder'] == './custom_output'
        
        # Check merged defaults
        assert config_dict['confidence']['weights']['header_quality'] == 0.3  # From defaults
        assert config_dict['confidence']['weights']['consistency'] == 0.3    # From defaults
        assert config_dict['output']['encoding'] == 'utf-8'                  # From defaults
        assert config_dict['output']['include_timestamp'] is True           # From defaults

    def test_yaml_loading_with_unicode_content(self, config_manager, temp_dir):
        """Test loading YAML with Unicode characters."""
        unicode_yaml = """
        monitoring:
          folders: ["./测试目录", "./тест"]
        output:
          folder: "./输出文件夹"
          naming_pattern: "{filename}_中文_{worksheet}.csv"
        """
        
        config_file = temp_dir / "unicode.yaml"
        config_file.write_text(unicode_yaml, encoding='utf-8')
        
        config_dict = config_manager._load_config_dict(config_file)
        
        # Verify Unicode content was loaded correctly
        assert "./测试目录" in config_dict['monitoring']['folders']
        assert "./тест" in config_dict['monitoring']['folders']
        assert config_dict['output']['folder'] == "./输出文件夹"
        assert "中文" in config_dict['output']['naming_pattern']

    def test_yaml_loading_different_data_types(self, config_manager, temp_dir):
        """Test loading YAML with different data types."""
        mixed_types_yaml = """
        monitoring:
          folders: ["./input"]  # List
          max_file_size: 150    # Integer
          process_existing: true # Boolean
          polling_interval: 2.5  # Float
        confidence:
          threshold: 0.8        # Float
          min_rows: null        # None/null
        custom:
          tags: ["tag1", "tag2", "tag3"]  # List of strings
          metadata:
            created: 2023-01-01  # Date-like string
            version: 1.0         # Float that looks like version
        """
        
        config_file = temp_dir / "mixed_types.yaml"
        config_file.write_text(mixed_types_yaml)
        
        config_dict = config_manager._load_config_dict(config_file)
        
        # Verify data types are preserved
        assert isinstance(config_dict['monitoring']['folders'], list)
        assert isinstance(config_dict['monitoring']['max_file_size'], int)
        assert isinstance(config_dict['monitoring']['process_existing'], bool)
        assert isinstance(config_dict['monitoring']['polling_interval'], float)
        assert isinstance(config_dict['confidence']['threshold'], float)
        assert config_dict['confidence']['min_rows'] is None
        assert isinstance(config_dict['custom']['tags'], list)
        assert len(config_dict['custom']['tags']) == 3


class TestConfigManagerEnvironmentOverrides:
    """Test suite for environment variable override functionality."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager instance for testing."""
        manager = ConfigManager()
        manager.clear_cache()
        return manager

    @pytest.fixture
    def base_config_dict(self):
        """Base configuration dictionary for testing overrides."""
        return {
            "monitoring": {
                "folders": ["./input"],
                "max_file_size": 100
            },
            "confidence": {
                "threshold": 0.8
            },
            "output": {
                "folder": "./output",
                "encoding": "utf-8",
                "include_timestamp": True,
                "delimiter": ","
            },
            "processing": {
                "max_concurrent": 5
            },
            "logging": {
                "level": "INFO"
            },
            "archiving": {
                "enabled": True,
                "archive_folder_name": "archive",
                "timestamp_format": "%Y%m%d_%H%M%S",
                "handle_conflicts": True,
                "preserve_structure": True
            }
        }

    def test_apply_env_overrides_confidence_threshold(self, config_manager, base_config_dict, env_override):
        """Test environment override for confidence threshold."""
        env_override.set("EXCEL_TO_CSV_CONFIDENCE_THRESHOLD", "0.95")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['confidence']['threshold'] == 0.95
        # Other values should remain unchanged
        assert result['output']['folder'] == "./output"

    def test_apply_env_overrides_output_folder(self, config_manager, base_config_dict, env_override):
        """Test environment override for output folder."""
        env_override.set("EXCEL_TO_CSV_OUTPUT_FOLDER", "/custom/output")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['output']['folder'] == "/custom/output"
        # Other values should remain unchanged
        assert result['confidence']['threshold'] == 0.8

    def test_apply_env_overrides_log_level(self, config_manager, base_config_dict, env_override):
        """Test environment override for logging level."""
        env_override.set("EXCEL_TO_CSV_LOG_LEVEL", "DEBUG")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['logging']['level'] == "DEBUG"

    def test_apply_env_overrides_max_concurrent(self, config_manager, base_config_dict, env_override):
        """Test environment override for max concurrent processing."""
        env_override.set("EXCEL_TO_CSV_MAX_CONCURRENT", "10")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['processing']['max_concurrent'] == 10  # Should be converted to int

    def test_apply_env_overrides_max_file_size(self, config_manager, base_config_dict, env_override):
        """Test environment override for max file size."""
        env_override.set("EXCEL_TO_CSV_MAX_FILE_SIZE", "250")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['monitoring']['max_file_size'] == 250

    def test_apply_env_overrides_boolean_values(self, config_manager, base_config_dict, env_override):
        """Test environment override for boolean values."""
        env_override.set("EXCEL_TO_CSV_INCLUDE_TIMESTAMP", "false")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['output']['include_timestamp'] is False

    def test_apply_env_overrides_encoding(self, config_manager, base_config_dict, env_override):
        """Test environment override for encoding."""
        env_override.set("EXCEL_TO_CSV_ENCODING", "iso-8859-1")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['output']['encoding'] == "iso-8859-1"

    def test_apply_env_overrides_delimiter(self, config_manager, base_config_dict, env_override):
        """Test environment override for CSV delimiter."""
        env_override.set("EXCEL_TO_CSV_DELIMITER", ";")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['output']['delimiter'] == ";"

    def test_apply_env_overrides_archive_settings(self, config_manager, base_config_dict, env_override):
        """Test environment overrides for archive settings."""
        env_override.set("EXCEL_TO_CSV_ARCHIVE_ENABLED", "false")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_FOLDER_NAME", "backup")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_TIMESTAMP_FORMAT", "%Y-%m-%d")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_HANDLE_CONFLICTS", "false")
        env_override.set("EXCEL_TO_CSV_ARCHIVE_PRESERVE_STRUCTURE", "false")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['archiving']['enabled'] is False
        assert result['archiving']['archive_folder_name'] == "backup"
        assert result['archiving']['timestamp_format'] == "%Y-%m-%d"
        assert result['archiving']['handle_conflicts'] is False
        assert result['archiving']['preserve_structure'] is False

    def test_apply_env_overrides_monitored_folders(self, config_manager, base_config_dict, env_override):
        """Test environment override for monitored folders (comma-separated)."""
        env_override.set("EXCEL_TO_CSV_MONITORED_FOLDERS", "./input1, ./input2, ./input3")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        expected_folders = ["./input1", "./input2", "./input3"]
        assert result['monitoring']['folders'] == expected_folders

    def test_apply_env_overrides_monitored_folders_single(self, config_manager, base_config_dict, env_override):
        """Test environment override for single monitored folder."""
        env_override.set("EXCEL_TO_CSV_MONITORED_FOLDERS", "./single_input")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        assert result['monitoring']['folders'] == ["./single_input"]

    def test_apply_env_overrides_monitored_folders_with_spaces(self, config_manager, base_config_dict, env_override):
        """Test environment override for monitored folders with extra spaces."""
        env_override.set("EXCEL_TO_CSV_MONITORED_FOLDERS", " ./input1 , ./input2 , ./input3 ")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        expected_folders = ["./input1", "./input2", "./input3"]
        assert result['monitoring']['folders'] == expected_folders

    def test_apply_env_overrides_no_env_vars_set(self, config_manager, base_config_dict):
        """Test that configuration remains unchanged when no env vars are set."""
        original_config = base_config_dict.copy()
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        # Should be identical since no env vars are set
        assert result == original_config

    def test_apply_env_overrides_multiple_variables(self, config_manager, base_config_dict, env_override):
        """Test multiple environment overrides applied together."""
        env_override.set("EXCEL_TO_CSV_CONFIDENCE_THRESHOLD", "0.9")
        env_override.set("EXCEL_TO_CSV_OUTPUT_FOLDER", "./custom_output")
        env_override.set("EXCEL_TO_CSV_MAX_CONCURRENT", "8")
        env_override.set("EXCEL_TO_CSV_ENCODING", "utf-16")
        env_override.set("EXCEL_TO_CSV_INCLUDE_TIMESTAMP", "false")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        # All overrides should be applied
        assert result['confidence']['threshold'] == 0.9
        assert result['output']['folder'] == "./custom_output"
        assert result['processing']['max_concurrent'] == 8
        assert result['output']['encoding'] == "utf-16"
        assert result['output']['include_timestamp'] is False
        
        # Non-overridden values should remain unchanged
        assert result['output']['delimiter'] == ","
        assert result['monitoring']['max_file_size'] == 100

    def test_apply_env_overrides_type_conversion_edge_cases(self, config_manager, base_config_dict, env_override):
        """Test type conversion edge cases."""
        # Test different boolean representations
        env_override.set("EXCEL_TO_CSV_INCLUDE_TIMESTAMP", "TRUE")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        assert result['output']['include_timestamp'] is True
        
        # Test zero value
        env_override.set("EXCEL_TO_CSV_MAX_CONCURRENT", "0")
        result = config_manager._apply_env_overrides(base_config_dict)
        assert result['processing']['max_concurrent'] == 0
        
        # Test negative number
        env_override.set("EXCEL_TO_CSV_MAX_FILE_SIZE", "-1")
        result = config_manager._apply_env_overrides(base_config_dict)
        assert result['monitoring']['max_file_size'] == -1
        
        # Test float that looks like int
        env_override.set("EXCEL_TO_CSV_CONFIDENCE_THRESHOLD", "1.0")
        result = config_manager._apply_env_overrides(base_config_dict)
        assert result['confidence']['threshold'] == 1.0
        assert isinstance(result['confidence']['threshold'], float)

    def test_apply_env_overrides_invalid_boolean(self, config_manager, base_config_dict, env_override):
        """Test handling of invalid boolean values."""
        env_override.set("EXCEL_TO_CSV_INCLUDE_TIMESTAMP", "maybe")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        # Should be treated as string since it's not a valid boolean
        assert result['output']['include_timestamp'] == "maybe"

    def test_apply_env_overrides_empty_string_values(self, config_manager, base_config_dict, env_override):
        """Test handling of empty string environment values."""
        env_override.set("EXCEL_TO_CSV_OUTPUT_FOLDER", "")
        env_override.set("EXCEL_TO_CSV_ENCODING", "")
        
        result = config_manager._apply_env_overrides(base_config_dict)
        
        # Empty strings should be preserved
        assert result['output']['folder'] == ""
        assert result['output']['encoding'] == ""

    def test_env_mappings_completeness(self, config_manager):
        """Test that all expected environment mappings are defined."""
        # Get the mappings from the _apply_env_overrides method
        # This is a bit of a hack to test the internal mappings
        base_config = {"test": {}}
        
        with patch.dict(os.environ, {
            "EXCEL_TO_CSV_CONFIDENCE_THRESHOLD": "0.9",
            "EXCEL_TO_CSV_OUTPUT_FOLDER": "/test",
            "EXCEL_TO_CSV_LOG_LEVEL": "DEBUG",
            "EXCEL_TO_CSV_MAX_CONCURRENT": "10",
            "EXCEL_TO_CSV_MAX_FILE_SIZE": "200",
            "EXCEL_TO_CSV_INCLUDE_TIMESTAMP": "true",
            "EXCEL_TO_CSV_ENCODING": "utf-8",
            "EXCEL_TO_CSV_DELIMITER": ",",
            "EXCEL_TO_CSV_ARCHIVE_ENABLED": "true",
            "EXCEL_TO_CSV_ARCHIVE_FOLDER_NAME": "archive",
            "EXCEL_TO_CSV_ARCHIVE_TIMESTAMP_FORMAT": "%Y%m%d",
            "EXCEL_TO_CSV_ARCHIVE_HANDLE_CONFLICTS": "true",
            "EXCEL_TO_CSV_ARCHIVE_PRESERVE_STRUCTURE": "true",
            "EXCEL_TO_CSV_MONITORED_FOLDERS": "./input"
        }, clear=False):
            
            # Prepare a config with all the sections that should be modified
            full_config = {
                "confidence": {"threshold": 0.8},
                "output": {
                    "folder": "./output", 
                    "include_timestamp": False,
                    "encoding": "ascii",
                    "delimiter": ";"
                },
                "logging": {"level": "INFO"},
                "processing": {"max_concurrent": 1},
                "monitoring": {
                    "max_file_size": 50,
                    "folders": ["./old"]
                },
                "archiving": {
                    "enabled": False,
                    "archive_folder_name": "old_archive",
                    "timestamp_format": "%Y",
                    "handle_conflicts": False,
                    "preserve_structure": False
                }
            }
            
            result = config_manager._apply_env_overrides(full_config)
            
            # Verify all mappings work
            assert result['confidence']['threshold'] == 0.9
            assert result['output']['folder'] == "/test"
            assert result['logging']['level'] == "DEBUG"
            assert result['processing']['max_concurrent'] == 10
            assert result['monitoring']['max_file_size'] == 200
            assert result['output']['include_timestamp'] is True
            assert result['output']['encoding'] == "utf-8"
            assert result['output']['delimiter'] == ","
            assert result['archiving']['enabled'] is True
            assert result['archiving']['archive_folder_name'] == "archive"
            assert result['archiving']['timestamp_format'] == "%Y%m%d"
            assert result['archiving']['handle_conflicts'] is True
            assert result['archiving']['preserve_structure'] is True
            assert result['monitoring']['folders'] == ["./input"]


class TestConfigManagerValidationAndCaching:
    """Test suite for configuration validation and caching functionality."""

    @pytest.fixture
    def config_manager(self):
        """Create a fresh ConfigManager instance for testing."""
        manager = ConfigManager()
        manager.clear_cache()
        return manager

    @pytest.fixture
    def mock_config(self):
        """Mock Config object for testing validation."""
        config = Mock()
        config.monitored_folders = [Path("./input")]
        config.output_folder = Path("./output")
        config.logging = Mock()
        config.logging.file_enabled = True
        config.logging.file_path = Path("./logs/test.log")
        return config

    def test_validate_config_success(self, config_manager, mock_config):
        """Test successful configuration validation."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # Should not raise exception
            config_manager._validate_config(mock_config)
            
            # mkdir should not be called since directories exist
            mock_mkdir.assert_not_called()

    def test_validate_config_creates_missing_monitored_folders(self, config_manager, mock_config):
        """Test that missing monitored folders are created."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            config_manager._validate_config(mock_config)
            
            # mkdir should be called for monitored folders
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)

    def test_validate_config_creates_missing_output_folder(self, config_manager, mock_config):
        """Test that missing output folder is created."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # Mock that monitored folders exist but output folder doesn't
            mock_exists.side_effect = lambda: mock_exists.call_count <= len(mock_config.monitored_folders)
            
            config_manager._validate_config(mock_config)
            
            # mkdir should be called for output folder
            assert mock_mkdir.call_count >= 1

    def test_validate_config_creates_log_directory(self, config_manager, mock_config):
        """Test that missing log directory is created."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # Mock that all directories exist except log directory
            def exists_side_effect():
                path_obj = mock_exists.call_args[0][0] if mock_exists.call_args else None
                # Mock log directory doesn't exist
                if hasattr(path_obj, 'parent') and str(path_obj).endswith('test.log'):
                    return False  # Log directory doesn't exist
                return True  # Other directories exist
            
            mock_exists.side_effect = exists_side_effect
            
            config_manager._validate_config(mock_config)
            
            # mkdir should be called for log directory
            assert mock_mkdir.call_count >= 1

    def test_validate_config_permission_error_monitored_folder(self, config_manager, mock_config):
        """Test handling of permission errors when creating monitored folders."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._validate_config(mock_config)
            
            assert "Cannot create monitored folder" in str(exc_info.value)
            assert "Access denied" in str(exc_info.value)

    def test_validate_config_permission_error_output_folder(self, config_manager, mock_config):
        """Test handling of permission errors when creating output folder."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # Mock that monitored folders exist but output folder creation fails
            mock_exists.side_effect = lambda: mock_exists.call_count <= len(mock_config.monitored_folders)
            mock_mkdir.side_effect = PermissionError("Access denied")
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._validate_config(mock_config)
            
            assert "Cannot create output folder" in str(exc_info.value)
            assert "Access denied" in str(exc_info.value)

    def test_validate_config_permission_error_log_directory(self, config_manager, mock_config):
        """Test handling of permission errors when creating log directory."""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            # Mock that monitored and output folders exist, but log dir creation fails
            def exists_side_effect():
                # Only log directory doesn't exist
                if mock_exists.call_count > len(mock_config.monitored_folders) + 1:
                    return False  # Log directory doesn't exist
                return True
            
            mock_exists.side_effect = exists_side_effect
            mock_mkdir.side_effect = PermissionError("Access denied")
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._validate_config(mock_config)
            
            assert "Cannot create log directory" in str(exc_info.value)

    def test_validate_config_disabled_file_logging(self, config_manager, mock_config):
        """Test validation when file logging is disabled."""
        mock_config.logging.file_enabled = False
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            config_manager._validate_config(mock_config)
            
            # Should not attempt to create log directory
            # mkdir should only be called for folders that don't exist (none in this case)
            mock_mkdir.assert_not_called()

    def test_validate_config_os_error_handling(self, config_manager, mock_config):
        """Test handling of OS errors during validation."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir', side_effect=OSError("Disk full")):
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._validate_config(mock_config)
            
            assert "Cannot create monitored folder" in str(exc_info.value)
            assert "Disk full" in str(exc_info.value)

    def test_validate_config_unexpected_error(self, config_manager, mock_config):
        """Test handling of unexpected errors during validation."""
        with patch('pathlib.Path.exists', side_effect=ValueError("Unexpected error")):
            
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager._validate_config(mock_config)
            
            assert "Configuration validation failed" in str(exc_info.value)

    def test_load_config_caching_with_different_parameters(self, config_manager):
        """Test that different parameters result in different cache entries."""
        with patch('pathlib.Path.exists', return_value=False):
            
            # Load config with different parameters
            config1 = config_manager.load_config(config_path=None, use_env_overrides=True)
            config2 = config_manager.load_config(config_path=None, use_env_overrides=False)
            config3 = config_manager.load_config(config_path="custom.yaml", use_env_overrides=True)
            
            # Should have separate cache entries
            assert len(config_manager._config_cache) == 3
            
            # Load again with same parameters - should use cache
            config1_again = config_manager.load_config(config_path=None, use_env_overrides=True)
            
            # Should be the same object (from cache)
            assert config1 is config1_again
            assert len(config_manager._config_cache) == 3  # No new cache entries

    def test_load_config_cache_key_generation(self, config_manager):
        """Test proper cache key generation for different scenarios."""
        with patch('pathlib.Path.exists', return_value=False):
            
            # Test different cache key scenarios
            test_cases = [
                (None, True, "None:True"),
                (None, False, "None:False"),
                ("config.yaml", True, "config.yaml:True"),
                ("config.yaml", False, "config.yaml:False"),
                (Path("config.yaml"), True, "config.yaml:True"),  # Path object should work too
            ]
            
            for config_path, use_env_overrides, expected_key_part in test_cases:
                config_manager.clear_cache()
                config = config_manager.load_config(config_path, use_env_overrides)
                
                # Check that cache contains expected key structure
                cache_keys = list(config_manager._config_cache.keys())
                assert len(cache_keys) == 1
                assert expected_key_part in cache_keys[0]

    def test_config_to_dict_conversion(self, config_manager):
        """Test conversion of Config object back to dictionary."""
        # Create a mock config with all required attributes
        mock_config = Mock()
        mock_config.monitored_folders = [Path("./input1"), Path("./input2")]
        mock_config.file_patterns = ["*.xlsx", "*.xls"]
        mock_config.max_file_size_mb = 100
        mock_config.confidence_threshold = 0.8
        mock_config.output_folder = Path("./output")
        mock_config.max_concurrent = 5
        
        # Mock output config
        mock_output_config = Mock()
        mock_output_config.naming_pattern = "{filename}_{worksheet}.csv"
        mock_output_config.include_timestamp = True
        mock_output_config.encoding = "utf-8"
        mock_output_config.delimiter = ","
        mock_output_config.include_headers = True
        mock_output_config.timestamp_format = "%Y%m%d_%H%M%S"
        mock_config.output_config = mock_output_config
        
        # Mock retry settings
        mock_retry = Mock()
        mock_retry.max_attempts = 3
        mock_retry.delay = 5
        mock_retry.backoff_factor = 2
        mock_retry.max_delay = 60
        mock_config.retry_settings = mock_retry
        
        # Mock logging config
        mock_logging = Mock()
        mock_logging.level = "INFO"
        mock_logging.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        mock_logging.file_enabled = True
        mock_logging.file_path = Path("./logs/test.log")
        mock_logging.console_enabled = True
        mock_logging.structured_enabled = False
        mock_config.logging = mock_logging
        
        # Mock archive config
        mock_archive = Mock()
        mock_archive.enabled = True
        mock_archive.archive_folder_name = "archive"
        mock_archive.timestamp_format = "%Y%m%d_%H%M%S"
        mock_archive.handle_conflicts = True
        mock_archive.preserve_structure = True
        mock_config.archive_config = mock_archive
        
        # Convert to dictionary
        result_dict = config_manager._config_to_dict(mock_config)
        
        # Verify structure
        assert isinstance(result_dict, dict)
        assert "monitoring" in result_dict
        assert "confidence" in result_dict
        assert "output" in result_dict
        assert "processing" in result_dict
        assert "logging" in result_dict
        assert "archiving" in result_dict
        
        # Verify content
        assert result_dict["monitoring"]["folders"] == ["./input1", "./input2"]
        assert result_dict["monitoring"]["file_patterns"] == ["*.xlsx", "*.xls"]
        assert result_dict["confidence"]["threshold"] == 0.8
        assert result_dict["output"]["folder"] == "./output"
        assert result_dict["processing"]["max_concurrent"] == 5

    def test_save_config_functionality(self, config_manager, temp_dir):
        """Test saving configuration to YAML file."""
        # Create a mock config
        mock_config = Mock()
        mock_config.monitored_folders = [Path("./input")]
        mock_config.file_patterns = ["*.xlsx"]
        mock_config.max_file_size_mb = 50
        mock_config.confidence_threshold = 0.9
        mock_config.output_folder = Path("./output")
        mock_config.max_concurrent = 3
        
        # Setup mock objects for all required attributes
        mock_config.output_config = Mock()
        mock_config.output_config.naming_pattern = "test_pattern"
        mock_config.output_config.include_timestamp = False
        mock_config.output_config.encoding = "ascii"
        mock_config.output_config.delimiter = ";"
        mock_config.output_config.include_headers = False
        mock_config.output_config.timestamp_format = "%Y%m%d"
        
        mock_config.retry_settings = Mock()
        mock_config.retry_settings.max_attempts = 5
        mock_config.retry_settings.delay = 10
        mock_config.retry_settings.backoff_factor = 3
        mock_config.retry_settings.max_delay = 120
        
        mock_config.logging = Mock()
        mock_config.logging.level = "DEBUG"
        mock_config.logging.format = "test format"
        mock_config.logging.file_enabled = False
        mock_config.logging.file_path = Path("./test.log")
        mock_config.logging.console_enabled = False
        mock_config.logging.structured_enabled = True
        
        mock_config.archive_config = Mock()
        mock_config.archive_config.enabled = False
        mock_config.archive_config.archive_folder_name = "backup"
        mock_config.archive_config.timestamp_format = "%Y%m%d"
        mock_config.archive_config.handle_conflicts = False
        mock_config.archive_config.preserve_structure = False
        
        # Save config
        config_file = temp_dir / "test_save_config.yaml"
        config_manager.save_config(mock_config, config_file)
        
        # Verify file was created
        assert config_file.exists()
        
        # Verify content by loading it back
        with open(config_file, 'r') as f:
            saved_content = yaml.safe_load(f)
        
        assert saved_content["monitoring"]["folders"] == ["./input"]
        assert saved_content["confidence"]["threshold"] == 0.9
        assert saved_content["output"]["folder"] == "./output"

    def test_save_config_creates_directory(self, config_manager, temp_dir):
        """Test that save_config creates parent directories."""
        mock_config = Mock()
        # Setup minimal mock config
        for attr in ['monitored_folders', 'file_patterns', 'max_file_size_mb', 
                     'confidence_threshold', 'output_folder', 'max_concurrent',
                     'output_config', 'retry_settings', 'logging', 'archive_config']:
            setattr(mock_config, attr, Mock())
        
        # Set up mock attributes that are accessed during conversion
        mock_config.monitored_folders = [Path("./input")]
        mock_config.output_folder = Path("./output")
        
        # Create nested directory path
        nested_config_file = temp_dir / "nested" / "config" / "test.yaml"
        
        config_manager.save_config(mock_config, nested_config_file)
        
        # Verify nested directories were created
        assert nested_config_file.parent.exists()
        assert nested_config_file.exists()

    def test_save_config_error_handling(self, config_manager):
        """Test error handling in save_config."""
        mock_config = Mock()
        
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            with pytest.raises(ConfigurationError) as exc_info:
                config_manager.save_config(mock_config, "/restricted/config.yaml")
            
            assert "Failed to save configuration" in str(exc_info.value)

    def test_dict_to_config_conversion(self, config_manager, sample_config_dict):
        """Test conversion from dictionary to Config object."""
        config = config_manager._dict_to_config(sample_config_dict)
        
        # Verify Config object was created properly
        assert hasattr(config, 'monitored_folders')
        assert hasattr(config, 'confidence_threshold')
        assert hasattr(config, 'output_folder')
        assert hasattr(config, 'file_patterns')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'retry_settings')
        assert hasattr(config, 'output_config')
        assert hasattr(config, 'archive_config')
        
        # Verify values are correct
        assert config.confidence_threshold == sample_config_dict['confidence']['threshold']
        assert len(config.monitored_folders) == len(sample_config_dict['monitoring']['folders'])

    def test_end_to_end_config_loading_with_validation_and_caching(self, config_manager, temp_dir, sample_config_dict):
        """Test complete end-to-end configuration loading process."""
        # Create config file
        config_file = temp_dir / "complete_test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        # Mock directory creation for validation
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.exists', return_value=False):
            
            # First load - should load from file and validate
            config1 = config_manager.load_config(config_file)
            
            # Verify validation was called (directories were "created")
            assert mock_mkdir.call_count > 0
            
            # Second load - should use cache
            mock_mkdir.reset_mock()
            config2 = config_manager.load_config(config_file)
            
            # Should be same object (from cache)
            assert config1 is config2
            # Should not attempt directory creation again
            mock_mkdir.assert_not_called()
            
            # Verify cache contains the entry
            assert len(config_manager._config_cache) == 1