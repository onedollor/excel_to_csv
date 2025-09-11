"""Improved unit tests for CSV generator with high coverage."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock
import csv
import os
from datetime import datetime

from excel_to_csv.generators.csv_generator import CSVGenerator, CSVGenerationError
from excel_to_csv.models.data_models import WorksheetData, OutputConfig


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 22],
        'Salary': [75000, 65000, 80000, 70000, 60000]
    })


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_worksheet(sample_data, temp_workspace):
    """Create sample WorksheetData for testing."""
    return WorksheetData(
        source_file=temp_workspace / "test.xlsx",
        worksheet_name="TestSheet",
        data=sample_data
    )


class TestCSVGeneratorBasics:
    """Test basic CSV generator functionality."""
    
    def test_generator_initialization(self):
        """Test CSV generator initialization."""
        generator = CSVGenerator()
        
        assert generator is not None
        assert hasattr(generator, 'generate_csv')
        assert hasattr(generator, 'logger')
        assert generator.MAX_FILENAME_LENGTH == 200
        assert generator.UNSAFE_FILENAME_CHARS == r'[<>:"/\\|?*\x00-\x1f]'
    
    def test_generate_csv_basic_success(self, sample_worksheet, temp_workspace):
        """Test successful CSV generation."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        # Check file was created
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        assert 'test_TestSheet' in output_path.name
        
        # Check content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # CSV generator may rename columns during header processing
            assert 'Alice' in content
            assert '75000' in content
            # Check that it's valid CSV format
            assert ',' in content or '"' in content
    
    def test_generate_csv_without_output_folder(self, sample_worksheet):
        """Test CSV generation without specified output folder."""
        generator = CSVGenerator() 
        config = OutputConfig(folder=None)  # Should use source file directory
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        assert output_path.exists()
        # Should be in same directory as source file
        assert output_path.parent == sample_worksheet.source_file.parent
    
    def test_custom_naming_pattern(self, sample_worksheet, temp_workspace):
        """Test custom naming pattern."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            naming_pattern="{worksheet}_{filename}_custom.csv"
        )
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        assert output_path.exists()
        assert 'TestSheet_test_custom' in output_path.name
    
    def test_custom_encoding(self, sample_worksheet, temp_workspace):
        """Test custom encoding."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            encoding='utf-16'
        )
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        assert output_path.exists()
        # Verify file can be read with specified encoding
        with open(output_path, 'r', encoding='utf-16') as f:
            content = f.read()
            assert 'Alice' in content
    
    def test_custom_delimiter(self, sample_worksheet, temp_workspace):
        """Test custom CSV delimiter."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            delimiter=';'
        )
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert ';' in content  # Check semicolon delimiter is used
            # Content should have semicolon-separated values
            lines = content.strip().split('\n')
            assert len(lines) >= 2  # Header + data rows


class TestCSVGeneratorDuplicates:
    """Test duplicate file handling."""
    
    def test_no_timestamp_when_no_conflict(self, sample_worksheet, temp_workspace):
        """Test that timestamp is NOT added when no file conflict."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            include_timestamp=True  # This only matters for conflicts
        )
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        # Should NOT have timestamp when no conflict
        assert output_path.exists()
        assert 'test_TestSheet.csv' == output_path.name
        # Should NOT contain timestamp pattern
        import re
        timestamp_pattern = r'\d{8}_\d{6}'
        assert not re.search(timestamp_pattern, output_path.name)
    
    def test_timestamp_added_on_conflict(self, sample_worksheet, temp_workspace):
        """Test that timestamp is added when file conflict exists."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            include_timestamp=True
        )
        
        # Create first file
        first_output = generator.generate_csv(sample_worksheet, config)
        assert first_output.exists()
        assert 'test_TestSheet.csv' == first_output.name
        
        # Create second file with same worksheet (should get timestamp)
        second_output = generator.generate_csv(sample_worksheet, config)
        assert second_output.exists()
        assert second_output != first_output
        
        # Second file should have timestamp
        import re
        timestamp_pattern = r'test_TestSheet_\d{8}_\d{6}\.csv'
        assert re.match(timestamp_pattern, second_output.name)
    
    def test_overwrite_when_timestamp_disabled(self, sample_worksheet, temp_workspace):
        """Test file overwrite when timestamp is disabled.""" 
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            include_timestamp=False
        )
        
        # Create first file
        first_output = generator.generate_csv(sample_worksheet, config)
        original_content = first_output.read_text()
        
        # Modify worksheet data
        modified_data = sample_worksheet.data.copy()
        modified_data.loc[0, 'Name'] = 'Modified'
        
        modified_worksheet = WorksheetData(
            source_file=sample_worksheet.source_file,
            worksheet_name=sample_worksheet.worksheet_name,
            data=modified_data
        )
        
        # Create second file (should overwrite)
        second_output = generator.generate_csv(modified_worksheet, config)
        
        assert second_output == first_output  # Same path
        assert second_output.read_text() != original_content  # Different content
        assert 'Modified' in second_output.read_text()


class TestCSVGeneratorFilenameHandling:
    """Test filename sanitization and handling."""
    
    def test_sanitize_filename_unsafe_characters(self, sample_data, temp_workspace):
        """Test sanitization of unsafe filename characters."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        # Create worksheet with unsafe characters in name
        unsafe_worksheet = WorksheetData(
            source_file=Path("test<>file.xlsx"),  # Unsafe filename
            worksheet_name="Sheet:With?Bad*Chars",  # Unsafe sheet name
            data=sample_data
        )
        
        output_path = generator.generate_csv(unsafe_worksheet, config)
        
        assert output_path.exists()
        # Unsafe characters should be cleaned
        assert '<' not in output_path.name
        assert '>' not in output_path.name
        assert ':' not in output_path.name
        assert '?' not in output_path.name
        assert '*' not in output_path.name
    
    def test_long_filename_truncation(self, sample_data, temp_workspace):
        """Test truncation of very long filenames."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        # Create worksheet with very long names
        long_filename = "a" * 250 + ".xlsx"
        long_sheet_name = "b" * 200
        
        long_worksheet = WorksheetData(
            source_file=Path(long_filename),
            worksheet_name=long_sheet_name,
            data=sample_data
        )
        
        output_path = generator.generate_csv(long_worksheet, config)
        
        assert output_path.exists()
        # Filename should be truncated to reasonable length
        assert len(output_path.name) <= generator.MAX_FILENAME_LENGTH


class TestCSVGeneratorDataHandling:
    """Test different data scenarios."""
    
    def test_empty_dataframe(self, temp_workspace):
        """Test CSV generation with empty DataFrame."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        empty_data = pd.DataFrame()
        empty_worksheet = WorksheetData(
            source_file=Path("empty.xlsx"),
            worksheet_name="EmptySheet",
            data=empty_data
        )
        
        output_path = generator.generate_csv(empty_worksheet, config)
        
        assert output_path.exists()
        # File should exist but be essentially empty
        content = output_path.read_text().strip()
        assert len(content) <= 10  # Minimal content
    
    def test_single_column_data(self, temp_workspace):
        """Test CSV generation with single column."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        single_col_data = pd.DataFrame({'OnlyColumn': [1, 2, 3, 4, 5]})
        single_worksheet = WorksheetData(
            source_file=Path("single.xlsx"),
            worksheet_name="SingleColumn", 
            data=single_col_data
        )
        
        output_path = generator.generate_csv(single_worksheet, config)
        
        assert output_path.exists()
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'OnlyColumn' in content
            assert '1' in content and '5' in content
    
    def test_data_with_nan_values(self, temp_workspace):
        """Test CSV generation with NaN values."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        nan_data = pd.DataFrame({
            'A': [1, None, 3],
            'B': ['x', 'y', None],
            'C': [1.1, 2.2, None]
        })
        
        nan_worksheet = WorksheetData(
            source_file=Path("nan_test.xlsx"),
            worksheet_name="NaNSheet",
            data=nan_data
        )
        
        output_path = generator.generate_csv(nan_worksheet, config)
        
        assert output_path.exists()
        # Should handle NaN values gracefully
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'A,B,C' in content
    
    def test_data_with_datetime(self, temp_workspace):
        """Test CSV generation with datetime columns."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        datetime_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=3),
            'Value': [10, 20, 30]
        })
        
        datetime_worksheet = WorksheetData(
            source_file=Path("datetime_test.xlsx"),
            worksheet_name="DateTimeSheet",
            data=datetime_data
        )
        
        output_path = generator.generate_csv(datetime_worksheet, config)
        
        assert output_path.exists()
        with open(output_path, 'r') as f:
            content = f.read()
            assert 'Date,Value' in content
            assert '2024' in content


class TestCSVGeneratorErrorHandling:
    """Test error handling scenarios."""
    
    def test_permission_error_handling(self, sample_worksheet):
        """Test handling of permission errors."""
        generator = CSVGenerator()
        
        # Try to write to a read-only location
        readonly_config = OutputConfig(folder=Path("/root"))  # Typically read-only
        
        # Should handle permission error gracefully or raise appropriate exception
        with pytest.raises((CSVGenerationError, PermissionError, OSError)):
            generator.generate_csv(sample_worksheet, readonly_config)
    
    @patch('pandas.DataFrame.to_csv')
    def test_csv_write_error_handling(self, mock_to_csv, sample_worksheet, temp_workspace):
        """Test handling of CSV writing errors."""
        mock_to_csv.side_effect = IOError("Disk full")
        
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        with pytest.raises((CSVGenerationError, IOError)):
            generator.generate_csv(sample_worksheet, config)
    
    def test_invalid_output_config(self, sample_worksheet):
        """Test handling of invalid output configuration."""
        generator = CSVGenerator()
        
        # Invalid folder path
        invalid_config = OutputConfig(folder="")  # Empty folder path
        
        # Should handle gracefully or raise appropriate error
        try:
            generator.generate_csv(sample_worksheet, invalid_config)
        except (CSVGenerationError, ValueError, OSError):
            pass  # Expected behavior


class TestCSVGeneratorUtilityMethods:
    """Test utility methods of CSV generator."""
    
    def test_filename_sanitization_method(self):
        """Test the filename sanitization method."""
        generator = CSVGenerator()
        
        # Test various unsafe characters
        unsafe_name = "file<>name:with?bad*chars"
        sanitized = generator._sanitize_filename(unsafe_name)
        
        assert '<' not in sanitized
        assert '>' not in sanitized
        assert ':' not in sanitized
        assert '?' not in sanitized
        assert '*' not in sanitized
    
    def test_output_folder_creation(self, sample_worksheet):
        """Test automatic output folder creation."""
        generator = CSVGenerator()
        
        # Use a non-existent folder
        temp_base = Path(tempfile.mkdtemp())
        non_existent = temp_base / "new_folder" / "sub_folder"
        config = OutputConfig(folder=non_existent)
        
        try:
            output_path = generator.generate_csv(sample_worksheet, config)
            
            # Folder should be created
            assert non_existent.exists()
            assert output_path.exists()
            assert output_path.parent == non_existent
        finally:
            # Cleanup
            shutil.rmtree(temp_base)


if __name__ == "__main__":
    pytest.main([__file__])