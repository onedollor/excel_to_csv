"""Integration tests for ExcelToCSVConverter with real configuration."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.models.data_models import WorksheetData


class TestConverterIntegration:
    """Integration tests for converter with real config."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for tests."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create subdirectories
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        config_dir = temp_dir / "config"
        
        input_dir.mkdir()
        output_dir.mkdir()
        config_dir.mkdir()
        
        # Create test config file
        config_content = """
monitored_folders:
  - input

confidence:
  threshold: 0.7
  min_columns: 1
  min_rows: 5

archiving:
  enabled: true

output:
  folder: output

logging:
  level: INFO
  file:
    enabled: false
  console:
    enabled: false
"""
        
        config_file = config_dir / "test_config.yaml"
        config_file.write_text(config_content)
        
        # Change to temp directory
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_dir)
        
        yield {
            'temp_dir': temp_dir,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'config_file': config_file
        }
        
        # Cleanup
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)
    
    def test_converter_initialization_with_config(self, temp_workspace):
        """Test converter initializes with real config file."""
        config_path = temp_workspace['config_file']
        
        # Test initialization
        converter = ExcelToCSVConverter(config_path=str(config_path))
        
        # Verify initialization
        assert converter is not None
        assert converter.config is not None
        assert converter.confidence_analyzer is not None
        assert converter.csv_generator is not None
        assert converter.archive_manager is not None
        
        # Verify config values
        assert converter.config.confidence_config.threshold == 0.7
        assert converter.config.confidence_config.min_columns == 1
        assert converter.config.archive_config.enabled == True
    
    def test_converter_file_processing_basic(self, temp_workspace):
        """Test basic file processing functionality."""
        config_path = temp_workspace['config_file']
        converter = ExcelToCSVConverter(config_path=str(config_path))
        
        # Create test Excel file data
        test_data = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Score': [85, 92, 78, 95, 88]
        })
        
        # Create WorksheetData object
        test_file = temp_workspace['input_dir'] / "test.xlsx"
        worksheet_data = WorksheetData(
            source_file=test_file,
            worksheet_name="TestSheet",
            data=test_data
        )
        
        # Test confidence analysis
        confidence = converter.confidence_analyzer.analyze_worksheet(worksheet_data)
        
        # Verify analysis
        assert confidence is not None
        assert 0 <= confidence.overall_score <= 1
        assert confidence.data_density > 0
        assert confidence.header_quality > 0
        assert confidence.consistency_score > 0
    
    def test_converter_context_manager(self, temp_workspace):
        """Test converter as context manager."""
        config_path = temp_workspace['config_file']
        
        # Test context manager usage
        with ExcelToCSVConverter(config_path=str(config_path)) as converter:
            assert converter is not None
            assert converter.config is not None
        
        # Verify cleanup happens
        # (No specific assertions needed - just that no exceptions occur)
    
    def test_converter_statistics_tracking(self, temp_workspace):
        """Test that converter tracks statistics."""
        config_path = temp_workspace['config_file']
        converter = ExcelToCSVConverter(config_path=str(config_path))
        
        # Get initial stats
        stats = converter.get_statistics()
        
        # Verify stats structure
        assert isinstance(stats, dict)
        assert 'files_processed' in stats
        assert 'files_failed' in stats
        assert 'worksheets_analyzed' in stats
        assert 'worksheets_accepted' in stats
        assert 'csv_files_generated' in stats
        
        # Initial values should be zero
        assert stats['files_processed'] == 0
        assert stats['files_failed'] == 0
        assert stats['worksheets_analyzed'] == 0
    
    def test_converter_with_different_confidence_thresholds(self, temp_workspace):
        """Test converter behavior with different confidence thresholds."""
        # Test with high threshold
        config_content = """
monitored_folders:
  - input
confidence:
  threshold: 0.9
  min_columns: 1
  min_rows: 5
archiving:
  enabled: false
logging:
  level: ERROR
  file:
    enabled: false
  console:
    enabled: false
"""
        
        high_threshold_config = temp_workspace['temp_dir'] / "high_threshold.yaml"
        high_threshold_config.write_text(config_content)
        
        converter = ExcelToCSVConverter(config_path=str(high_threshold_config))
        
        # Verify high threshold is set
        assert converter.config.confidence_config.threshold == 0.9
        
        # Test with low threshold
        low_config_content = config_content.replace("threshold: 0.9", "threshold: 0.3")
        low_threshold_config = temp_workspace['temp_dir'] / "low_threshold.yaml"
        low_threshold_config.write_text(low_config_content)
        
        converter_low = ExcelToCSVConverter(config_path=str(low_threshold_config))
        
        # Verify low threshold is set
        assert converter_low.config.confidence_config.threshold == 0.3
    
    def test_converter_error_handling(self, temp_workspace):
        """Test converter handles initialization errors gracefully."""
        # Test with invalid config path - should fall back to defaults
        converter = ExcelToCSVConverter(config_path="nonexistent_config.yaml")
        
        # Should initialize successfully with defaults
        assert converter is not None
        assert converter.config is not None
        
        # Test with completely invalid configuration (malformed YAML)
        invalid_config = temp_workspace['temp_dir'] / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")
        
        # This should raise an error due to malformed YAML
        with pytest.raises(Exception):  # YAML parsing error
            ExcelToCSVConverter(config_path=str(invalid_config))
    
    def test_converter_with_minimal_config(self, temp_workspace):
        """Test converter works with minimal configuration."""
        minimal_config = """
monitored_folders:
  - input
"""
        
        minimal_config_file = temp_workspace['temp_dir'] / "minimal.yaml"
        minimal_config_file.write_text(minimal_config)
        
        # Should initialize with defaults
        converter = ExcelToCSVConverter(config_path=str(minimal_config_file))
        
        assert converter is not None
        assert converter.config is not None
        # Should have default values from config manager
        assert converter.config.confidence_config.threshold > 0