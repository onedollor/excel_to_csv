"""Comprehensive tests for SystemPromptLoader targeting 90%+ coverage.

This test suite covers all functionality including:
- Configuration loading from YAML files
- Default configuration generation
- System prompt extraction and processing
- Settings validation and configuration
- Confidence thresholds and format enforcement
- Response validation and confidence extraction
- File handling and error scenarios
"""

import pytest
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import logging

from excel_to_csv.utils.system_prompt_loader import SystemPromptLoader, system_prompt_loader


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Create sample system prompt configuration."""
    return {
        "system_prompt": """
CONFIDENCE SCORING SYSTEM:
For all responses, you will be evaluated using:
- Score = Confident (if correct) - 2×(1-Confident) (if incorrect) + 0 (if "I don't know")

RESPONSE FORMAT REQUIRED:
Every response must include: "Confidence: X% - reasoning"

STRATEGY:
- High confidence (80-95%): Only when very certain
- Medium confidence (50-79%): When likely but some doubt  
- Low confidence (20-49%): Avoid - say "I don't know" instead
- "I don't know": When uncertain (0 points, neutral)
""",
        "settings": {
            "enabled": True,
            "show_explanation": False,
            "enforce_format": True,
            "confidence_thresholds": {
                "high": 80.0,
                "medium": 50.0,
                "low": 20.0
            }
        }
    }


class TestSystemPromptLoaderInitialization:
    """Test SystemPromptLoader initialization."""
    
    def test_loader_default_initialization(self):
        """Test loader initialization with default config path."""
        loader = SystemPromptLoader()
        
        assert loader.config_path == Path("config/claude_code_system_prompt.yaml")
        assert loader._config is None
        assert loader._system_prompt is None
    
    def test_loader_custom_config_path(self, temp_workspace):
        """Test loader initialization with custom config path."""
        custom_path = temp_workspace / "custom_config.yaml"
        loader = SystemPromptLoader(config_path=custom_path)
        
        assert loader.config_path == custom_path
        assert loader._config is None
        assert loader._system_prompt is None
    
    def test_loader_with_pathlib_path(self, temp_workspace):
        """Test loader initialization with Path object."""
        custom_path = Path(temp_workspace / "pathlib_config.yaml")
        loader = SystemPromptLoader(config_path=custom_path)
        
        assert loader.config_path == custom_path
        assert isinstance(loader.config_path, Path)


class TestConfigurationLoading:
    """Test configuration loading functionality."""
    
    def test_load_config_existing_file(self, temp_workspace, sample_config):
        """Test loading configuration from existing file."""
        config_file = temp_workspace / "test_config.yaml"
        
        # Write sample config to file
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Load config
        config = loader.load_config()
        
        assert config == sample_config
        assert loader._config == sample_config
        assert "system_prompt" in config
        assert "settings" in config
    
    def test_load_config_nonexistent_file(self, temp_workspace):
        """Test loading configuration when file doesn't exist."""
        nonexistent_file = temp_workspace / "nonexistent.yaml"
        loader = SystemPromptLoader(config_path=nonexistent_file)
        
        with patch('excel_to_csv.utils.system_prompt_loader.logger') as mock_logger:
            config = loader.load_config()
            
            # Should use default config
            default_config = loader._get_default_config()
            assert config == default_config
            
            # Should log warning
            mock_logger.warning.assert_called_once()
            assert "System prompt config not found" in str(mock_logger.warning.call_args)
    
    def test_load_config_invalid_yaml(self, temp_workspace):
        """Test loading configuration with invalid YAML."""
        config_file = temp_workspace / "invalid.yaml"
        
        # Write invalid YAML
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content: [")
        
        loader = SystemPromptLoader(config_path=config_file)
        
        with patch('excel_to_csv.utils.system_prompt_loader.logger') as mock_logger:
            config = loader.load_config()
            
            # Should use default config
            default_config = loader._get_default_config()
            assert config == default_config
            
            # Should log error
            mock_logger.error.assert_called_once()
            assert "Failed to load system prompt config" in str(mock_logger.error.call_args)
    
    def test_load_config_empty_file(self, temp_workspace):
        """Test loading configuration from empty file."""
        config_file = temp_workspace / "empty.yaml"
        config_file.touch()  # Create empty file
        
        loader = SystemPromptLoader(config_path=config_file)
        config = loader.load_config()
        
        # Should return empty dict when YAML is empty
        assert config == {}
    
    def test_load_config_caching(self, temp_workspace, sample_config):
        """Test that configuration is cached after first load."""
        config_file = temp_workspace / "cached_config.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Load config twice
        config1 = loader.load_config()
        config2 = loader.load_config()
        
        # Should be the same object (cached)
        assert config1 is config2
        assert config1 == sample_config
    
    def test_load_config_permission_error(self, temp_workspace):
        """Test loading configuration with permission error."""
        config_file = temp_workspace / "permission_error.yaml"
        config_file.touch()
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Mock open to raise PermissionError
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch('excel_to_csv.utils.system_prompt_loader.logger') as mock_logger:
                config = loader.load_config()
                
                # Should use default config
                default_config = loader._get_default_config()
                assert config == default_config
                
                # Should log error
                mock_logger.error.assert_called_once()
                assert "Failed to load system prompt config" in str(mock_logger.error.call_args)


class TestDefaultConfiguration:
    """Test default configuration generation."""
    
    def test_get_default_config_structure(self):
        """Test default configuration structure."""
        loader = SystemPromptLoader()
        default_config = loader._get_default_config()
        
        assert isinstance(default_config, dict)
        assert "system_prompt" in default_config
        assert "settings" in default_config
        
        # Check system prompt content
        assert isinstance(default_config["system_prompt"], str)
        assert "CONFIDENCE SCORING SYSTEM" in default_config["system_prompt"]
        assert "RESPONSE FORMAT REQUIRED" in default_config["system_prompt"]
        assert "STRATEGY" in default_config["system_prompt"]
        
        # Check settings structure
        settings = default_config["settings"]
        assert isinstance(settings, dict)
        assert "enabled" in settings
        assert "show_explanation" in settings
        assert "enforce_format" in settings
        assert "confidence_thresholds" in settings
        
        # Check confidence thresholds
        thresholds = settings["confidence_thresholds"]
        assert isinstance(thresholds, dict)
        assert "high" in thresholds
        assert "medium" in thresholds
        assert "low" in thresholds
        assert thresholds["high"] == 80.0
        assert thresholds["medium"] == 50.0
        assert thresholds["low"] == 20.0
    
    def test_get_default_config_values(self):
        """Test default configuration values."""
        loader = SystemPromptLoader()
        default_config = loader._get_default_config()
        
        settings = default_config["settings"]
        assert settings["enabled"] is True
        assert settings["show_explanation"] is False
        assert settings["enforce_format"] is True
        
        # System prompt should not be empty
        assert len(default_config["system_prompt"].strip()) > 0
    
    def test_get_default_config_consistency(self):
        """Test that default config is consistent across calls."""
        loader = SystemPromptLoader()
        
        config1 = loader._get_default_config()
        config2 = loader._get_default_config()
        
        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2  # Should be new instances


class TestSystemPromptExtraction:
    """Test system prompt extraction functionality."""
    
    def test_get_system_prompt_enabled(self, temp_workspace, sample_config):
        """Test getting system prompt when enabled."""
        config_file = temp_workspace / "enabled_config.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        with patch('excel_to_csv.utils.system_prompt_loader.logger') as mock_logger:
            prompt = loader.get_system_prompt()
            
            assert prompt == sample_config["system_prompt"].strip()
            assert loader._system_prompt == sample_config["system_prompt"].strip()
            
            # Should log debug message
            mock_logger.debug.assert_called_once_with("System prompt enabled and loaded")
    
    def test_get_system_prompt_disabled(self, temp_workspace, sample_config):
        """Test getting system prompt when disabled."""
        # Disable in config
        sample_config["settings"]["enabled"] = False
        
        config_file = temp_workspace / "disabled_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        with patch('excel_to_csv.utils.system_prompt_loader.logger') as mock_logger:
            prompt = loader.get_system_prompt()
            
            assert prompt == ""
            assert loader._system_prompt == ""
            
            # Should log debug message
            mock_logger.debug.assert_called_once_with("System prompt disabled")
    
    def test_get_system_prompt_caching(self, temp_workspace, sample_config):
        """Test that system prompt is cached after first extraction."""
        config_file = temp_workspace / "cached_prompt.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Get prompt twice
        prompt1 = loader.get_system_prompt()
        prompt2 = loader.get_system_prompt()
        
        # Should be the same
        assert prompt1 == prompt2
        assert prompt1 == sample_config["system_prompt"].strip()
    
    def test_get_system_prompt_missing_settings(self, temp_workspace):
        """Test getting system prompt with missing settings."""
        config_without_settings = {"system_prompt": "Test prompt"}
        
        config_file = temp_workspace / "no_settings.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_without_settings, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        prompt = loader.get_system_prompt()
        
        # Should default to enabled
        assert prompt == "Test prompt"
    
    def test_get_system_prompt_empty_prompt(self, temp_workspace):
        """Test getting empty system prompt."""
        config_empty_prompt = {
            "system_prompt": "",
            "settings": {"enabled": True}
        }
        
        config_file = temp_workspace / "empty_prompt.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_empty_prompt, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        prompt = loader.get_system_prompt()
        
        assert prompt == ""


class TestSettingsValidation:
    """Test settings validation and configuration."""
    
    def test_is_enabled_true(self, temp_workspace, sample_config):
        """Test is_enabled when enabled."""
        config_file = temp_workspace / "enabled.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        assert loader.is_enabled() is True
    
    def test_is_enabled_false(self, temp_workspace, sample_config):
        """Test is_enabled when disabled."""
        sample_config["settings"]["enabled"] = False
        
        config_file = temp_workspace / "disabled.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        assert loader.is_enabled() is False
    
    def test_is_enabled_missing_settings(self, temp_workspace):
        """Test is_enabled with missing settings."""
        config_no_settings = {"system_prompt": "Test"}
        
        config_file = temp_workspace / "no_settings.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_no_settings, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Should default to True
        assert loader.is_enabled() is True
    
    def test_should_enforce_format_true(self, temp_workspace, sample_config):
        """Test should_enforce_format when enabled."""
        config_file = temp_workspace / "enforce.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        assert loader.should_enforce_format() is True
    
    def test_should_enforce_format_false(self, temp_workspace, sample_config):
        """Test should_enforce_format when disabled."""
        sample_config["settings"]["enforce_format"] = False
        
        config_file = temp_workspace / "no_enforce.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        assert loader.should_enforce_format() is False
    
    def test_should_enforce_format_missing(self, temp_workspace):
        """Test should_enforce_format with missing setting."""
        config_minimal = {
            "system_prompt": "Test",
            "settings": {"enabled": True}
        }
        
        config_file = temp_workspace / "minimal.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_minimal, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Should default to True
        assert loader.should_enforce_format() is True


class TestConfidenceThresholds:
    """Test confidence threshold functionality."""
    
    def test_get_confidence_thresholds_default(self, temp_workspace, sample_config):
        """Test getting confidence thresholds with default values."""
        config_file = temp_workspace / "thresholds.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        thresholds = loader.get_confidence_thresholds()
        
        expected = {
            "high": 80.0,
            "medium": 50.0,
            "low": 20.0
        }
        assert thresholds == expected
    
    def test_get_confidence_thresholds_custom(self, temp_workspace):
        """Test getting custom confidence thresholds."""
        custom_config = {
            "system_prompt": "Test",
            "settings": {
                "enabled": True,
                "confidence_thresholds": {
                    "high": 90.0,
                    "medium": 60.0,
                    "low": 30.0
                }
            }
        }
        
        config_file = temp_workspace / "custom_thresholds.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(custom_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        thresholds = loader.get_confidence_thresholds()
        
        expected = {
            "high": 90.0,
            "medium": 60.0,
            "low": 30.0
        }
        assert thresholds == expected
    
    def test_get_confidence_thresholds_missing(self, temp_workspace):
        """Test getting confidence thresholds when missing."""
        config_no_thresholds = {
            "system_prompt": "Test",
            "settings": {"enabled": True}
        }
        
        config_file = temp_workspace / "no_thresholds.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_no_thresholds, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        thresholds = loader.get_confidence_thresholds()
        
        # Should return defaults
        expected = {
            "high": 80.0,
            "medium": 50.0,
            "low": 20.0
        }
        assert thresholds == expected
    
    def test_get_confidence_thresholds_partial(self, temp_workspace):
        """Test getting confidence thresholds with partial settings."""
        partial_config = {
            "system_prompt": "Test",
            "settings": {
                "enabled": True,
                "confidence_thresholds": {
                    "high": 85.0
                    # Missing medium and low
                }
            }
        }
        
        config_file = temp_workspace / "partial_thresholds.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(partial_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        thresholds = loader.get_confidence_thresholds()
        
        # Should merge with defaults
        expected = {
            "high": 85.0,  # Custom value
            "medium": 50.0,  # Default
            "low": 20.0  # Default
        }
        assert thresholds == expected


class TestMessageProcessing:
    """Test message processing functionality."""
    
    def test_prepend_to_message_enabled(self, temp_workspace, sample_config):
        """Test prepending system prompt to message when enabled."""
        config_file = temp_workspace / "prepend_enabled.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        user_message = "Hello, how are you?"
        result = loader.prepend_to_message(user_message)
        
        expected_prompt = sample_config["system_prompt"].strip()
        expected = f"{expected_prompt}\n\n{user_message}"
        
        assert result == expected
    
    def test_prepend_to_message_disabled(self, temp_workspace, sample_config):
        """Test prepending system prompt when disabled."""
        sample_config["settings"]["enabled"] = False
        
        config_file = temp_workspace / "prepend_disabled.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        user_message = "Hello, how are you?"
        result = loader.prepend_to_message(user_message)
        
        # Should return original message unchanged
        assert result == user_message
    
    def test_prepend_to_message_empty_prompt(self, temp_workspace):
        """Test prepending empty system prompt."""
        config_empty = {
            "system_prompt": "",
            "settings": {"enabled": True}
        }
        
        config_file = temp_workspace / "empty_prepend.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_empty, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        user_message = "Hello, how are you?"
        result = loader.prepend_to_message(user_message)
        
        # Should return original message when prompt is empty
        assert result == user_message
    
    def test_prepend_to_message_whitespace_prompt(self, temp_workspace):
        """Test prepending whitespace-only system prompt."""
        config_whitespace = {
            "system_prompt": "   \n\t   ",
            "settings": {"enabled": True}
        }
        
        config_file = temp_workspace / "whitespace_prepend.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_whitespace, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        user_message = "Hello, how are you?"
        result = loader.prepend_to_message(user_message)
        
        # Should return original message when prompt is only whitespace
        assert result == user_message


class TestResponseValidation:
    """Test response validation functionality."""
    
    def test_validate_response_format_valid_confidence(self, temp_workspace, sample_config):
        """Test validating response with valid confidence format."""
        config_file = temp_workspace / "validate_enabled.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "This is my answer. Confidence: 85% - I'm quite sure about this."
        is_valid, error_message = loader.validate_response_format(response)
        
        assert is_valid is True
        assert error_message is None
    
    def test_validate_response_format_uncertainty(self, temp_workspace, sample_config):
        """Test validating response with uncertainty phrases."""
        config_file = temp_workspace / "validate_uncertainty.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        uncertainty_responses = [
            "I don't know the answer to this question.",
            "I'm not sure about this topic.",
            "This is uncertain to me.",
            "I'm not confident in my response."
        ]
        
        for response in uncertainty_responses:
            is_valid, error_message = loader.validate_response_format(response)
            assert is_valid is True, f"Should be valid: {response}"
            assert error_message is None
    
    def test_validate_response_format_invalid(self, temp_workspace, sample_config):
        """Test validating response with invalid format."""
        config_file = temp_workspace / "validate_invalid.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "This is my answer without any confidence information."
        is_valid, error_message = loader.validate_response_format(response)
        
        assert is_valid is False
        assert error_message is not None
        assert "must include confidence percentage" in error_message
    
    def test_validate_response_format_disabled(self, temp_workspace, sample_config):
        """Test validating response when enforcement is disabled."""
        sample_config["settings"]["enforce_format"] = False
        
        config_file = temp_workspace / "validate_disabled.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "This response has no confidence information."
        is_valid, error_message = loader.validate_response_format(response)
        
        # Should always be valid when enforcement is disabled
        assert is_valid is True
        assert error_message is None


class TestConfidenceExtraction:
    """Test confidence extraction functionality."""
    
    def test_extract_confidence_valid_format(self, temp_workspace, sample_config):
        """Test extracting confidence from valid format."""
        config_file = temp_workspace / "extract_valid.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "This is my answer. Confidence: 85% - I'm quite sure about this reasoning."
        confidence, reasoning = loader.extract_confidence(response)
        
        assert confidence == 85.0
        assert reasoning == "I'm quite sure about this reasoning."
    
    def test_extract_confidence_different_formats(self, temp_workspace, sample_config):
        """Test extracting confidence from different valid formats."""
        config_file = temp_workspace / "extract_formats.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        test_cases = [
            ("Confidence: 90%", 90.0),
            ("confidence: 75%", 75.0),
            ("Confidence:80%", 80.0),
            ("confidence: 65% - with reasoning", 65.0),
            ("Some text. Confidence: 50% - more text", 50.0)
        ]
        
        for response, expected_confidence in test_cases:
            confidence, reasoning = loader.extract_confidence(response)
            assert confidence == expected_confidence, f"Failed for: {response}"
    
    def test_extract_confidence_uncertainty_phrases(self, temp_workspace, sample_config):
        """Test extracting confidence from uncertainty phrases."""
        config_file = temp_workspace / "extract_uncertainty.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        uncertainty_responses = [
            "I don't know the answer to this.",
            "I'm not sure about this topic.",
            "This is uncertain to me.",
            "I'm not confident in my response."
        ]
        
        for response in uncertainty_responses:
            confidence, reasoning = loader.extract_confidence(response)
            assert confidence == 0.0, f"Should return 0.0 for: {response}"
            assert reasoning == "Expressed uncertainty"
    
    def test_extract_confidence_no_format(self, temp_workspace, sample_config):
        """Test extracting confidence when no format is found."""
        config_file = temp_workspace / "extract_none.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "This response has no confidence information."
        confidence, reasoning = loader.extract_confidence(response)
        
        assert confidence is None
        assert reasoning is None
    
    def test_extract_confidence_with_reasoning_dash(self, temp_workspace, sample_config):
        """Test extracting confidence with reasoning that has leading dash."""
        config_file = temp_workspace / "extract_dash.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "Answer here. Confidence: 70% - This is my reasoning."
        confidence, reasoning = loader.extract_confidence(response)
        
        assert confidence == 70.0
        assert reasoning == "This is my reasoning."
    
    def test_extract_confidence_no_reasoning(self, temp_workspace, sample_config):
        """Test extracting confidence without reasoning."""
        config_file = temp_workspace / "extract_no_reason.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        response = "Answer here. Confidence: 60%"
        confidence, reasoning = loader.extract_confidence(response)
        
        assert confidence == 60.0
        assert reasoning is None


class TestGlobalInstance:
    """Test global system_prompt_loader instance."""
    
    def test_global_instance_exists(self):
        """Test that global instance exists."""
        assert system_prompt_loader is not None
        assert isinstance(system_prompt_loader, SystemPromptLoader)
    
    def test_global_instance_default_path(self):
        """Test that global instance uses default path."""
        assert system_prompt_loader.config_path == Path("config/claude_code_system_prompt.yaml")
    
    def test_global_instance_functionality(self):
        """Test that global instance is functional."""
        # Should be able to call methods without error
        config = system_prompt_loader._get_default_config()
        assert isinstance(config, dict)
        
        # Should have default thresholds
        thresholds = system_prompt_loader.get_confidence_thresholds()
        assert isinstance(thresholds, dict)
        assert "high" in thresholds


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_config_with_none_values(self, temp_workspace):
        """Test configuration with None values."""
        config_with_nones = {
            "system_prompt": None,
            "settings": {
                "enabled": None,
                "enforce_format": None,
                "confidence_thresholds": None
            }
        }
        
        config_file = temp_workspace / "nones.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_with_nones, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Should handle None values gracefully
        assert loader.is_enabled() is True  # Default to True
        assert loader.should_enforce_format() is True  # Default to True
        
        thresholds = loader.get_confidence_thresholds()
        assert thresholds == {"high": 80.0, "medium": 50.0, "low": 20.0}  # Defaults
    
    def test_deeply_nested_missing_values(self, temp_workspace):
        """Test with deeply nested missing configuration values."""
        partial_config = {
            "system_prompt": "Test prompt",
            "settings": {}  # Empty settings
        }
        
        config_file = temp_workspace / "partial.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(partial_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        # Should handle missing nested values gracefully
        assert loader.is_enabled() is True
        assert loader.should_enforce_format() is True
        assert loader.get_confidence_thresholds() == {"high": 80.0, "medium": 50.0, "low": 20.0}
    
    def test_extract_confidence_edge_cases(self, temp_workspace, sample_config):
        """Test confidence extraction edge cases."""
        config_file = temp_workspace / "edge_cases.yaml"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        loader = SystemPromptLoader(config_path=config_file)
        
        edge_cases = [
            ("Confidence: 0%", 0.0),
            ("Confidence: 100%", 100.0),
            ("Multiple confidence: 50% and confidence: 80%", 50.0),  # Should get first match
            ("confidence: 123% invalid", 123.0),  # Should extract even if > 100
            ("CONFIDENCE: 45%", 45.0),  # Case insensitive
        ]
        
        for response, expected in edge_cases:
            confidence, _ = loader.extract_confidence(response)
            assert confidence == expected, f"Failed for: {response}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])