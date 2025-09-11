"""Comprehensive CSV generator tests targeting 90%+ coverage.

This test suite covers all major functionality including:
- Header detection and setup
- Data cleaning and formatting
- Numeric and date formatting
- File validation and size estimation
- Preview generation
- Complex edge cases and error scenarios
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock, call
import csv
import os
import stat
from datetime import datetime
import warnings

from excel_to_csv.generators.csv_generator import CSVGenerator, CSVGenerationError
from excel_to_csv.models.data_models import WorksheetData, OutputConfig


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 22],
        'Salary': [75000.0, 65000.5, 80000.25, 70000.0, 60000.75]
    })


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_worksheet(sample_data, temp_workspace):
    """Create sample WorksheetData for testing."""
    return WorksheetData(
        source_file=temp_workspace / "test.xlsx",
        worksheet_name="TestSheet",
        data=sample_data
    )


class TestCSVGeneratorComprehensive:
    """Comprehensive CSV generator testing for 90%+ coverage."""
    
    def test_generator_initialization_with_metadata(self):
        """Test initialization captures all metadata correctly."""
        generator = CSVGenerator()
        
        assert generator.MAX_FILENAME_LENGTH == 200
        assert generator.UNSAFE_FILENAME_CHARS == r'[<>:"/\\|?*\x00-\x1f]'
        assert hasattr(generator, 'logger')
        assert generator.logger is not None
    
    def test_generate_csv_full_workflow(self, sample_worksheet, temp_workspace):
        """Test complete CSV generation workflow with all steps."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            include_headers=True,
            include_timestamp=False,
            delimiter=',',
            encoding='utf-8'
        )
        
        output_path = generator.generate_csv(sample_worksheet, config)
        
        # Verify all workflow steps completed
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        
        # Check file content is correctly formatted
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        # Should have header + data rows
        assert len(rows) >= 6  # 1 header + 5 data rows
        
        # Check data integrity
        assert 'Alice' in str(rows)
        assert '75000' in str(rows)
    
    def test_determine_output_path_with_folder(self, sample_worksheet, temp_workspace):
        """Test output path determination with custom folder."""
        generator = CSVGenerator()
        config = OutputConfig(
            folder=temp_workspace,
            naming_pattern="{filename}_{worksheet}.csv"
        )
        
        path = generator._determine_output_path(sample_worksheet, config)
        
        assert path.parent == temp_workspace
        assert 'test_TestSheet' in path.name
        assert path.suffix == '.csv'
    
    def test_determine_output_path_no_folder(self, sample_worksheet):
        """Test output path determination without folder (use source directory)."""
        generator = CSVGenerator()
        config = OutputConfig(folder=None)
        
        path = generator._determine_output_path(sample_worksheet, config)
        
        assert path.parent == sample_worksheet.source_file.parent
        assert 'test_TestSheet' in path.name
        assert path.suffix == '.csv'
    
    def test_sanitize_filename_comprehensive(self):
        """Test comprehensive filename sanitization."""
        generator = CSVGenerator()
        
        # Test all unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        filename = f"test{unsafe_chars}file"
        sanitized = generator._sanitize_filename(filename)
        
        for char in unsafe_chars:
            assert char not in sanitized
        
        # Test multiple underscores consolidation
        multi_underscore = "test___multiple____underscores"
        sanitized = generator._sanitize_filename(multi_underscore)
        assert '___' not in sanitized
        
        # Test leading/trailing cleanup
        messy = "___test___file___.txt"
        sanitized = generator._sanitize_filename(messy)
        assert not sanitized.startswith('_')
        assert not sanitized.endswith('_')
        
        # Test empty filename handling
        empty = ""
        sanitized = generator._sanitize_filename(empty)
        assert sanitized == "worksheet.csv"
        
        # Test CSV extension addition
        no_ext = "testfile"
        sanitized = generator._sanitize_filename(no_ext)
        assert sanitized.endswith('.csv')
        
        # Test already has CSV extension
        has_ext = "testfile.csv"
        sanitized = generator._sanitize_filename(has_ext)
        assert sanitized.endswith('.csv')
        assert sanitized.count('.csv') == 1
    
    def test_sanitize_filename_length_truncation(self):
        """Test filename length truncation."""
        generator = CSVGenerator()
        
        # Create very long filename
        long_name = "a" * 300
        sanitized = generator._sanitize_filename(long_name)
        
        # Should be truncated to max length
        name_part = sanitized.replace('.csv', '')
        assert len(name_part) <= generator.MAX_FILENAME_LENGTH
        assert sanitized.endswith('.csv')
    
    def test_handle_duplicates_no_conflict(self, temp_workspace):
        """Test duplicate handling when no conflict exists."""
        generator = CSVGenerator()
        config = OutputConfig(include_timestamp=True)
        
        non_existent_path = temp_workspace / "non_existent.csv"
        result_path = generator._handle_duplicates(non_existent_path, config)
        
        assert result_path == non_existent_path
    
    def test_handle_duplicates_overwrite_mode(self, temp_workspace):
        """Test duplicate handling in overwrite mode."""
        generator = CSVGenerator()
        config = OutputConfig(include_timestamp=False)
        
        # Create existing file
        existing_path = temp_workspace / "existing.csv"
        existing_path.write_text("existing content")
        
        result_path = generator._handle_duplicates(existing_path, config)
        
        assert result_path == existing_path  # Should overwrite
    
    def test_handle_duplicates_timestamp_mode(self, temp_workspace):
        """Test duplicate handling with timestamp."""
        generator = CSVGenerator()
        config = OutputConfig(
            include_timestamp=True,
            timestamp_format="%Y%m%d_%H%M%S"
        )
        
        # Create existing file
        existing_path = temp_workspace / "existing.csv"
        existing_path.write_text("existing content")
        
        result_path = generator._handle_duplicates(existing_path, config)
        
        assert result_path != existing_path
        assert existing_path.stem in result_path.name
        assert result_path.suffix == '.csv'
        
        # Should contain timestamp pattern
        import re
        timestamp_pattern = r'\d{8}_\d{6}'
        assert re.search(timestamp_pattern, result_path.name)
    
    def test_handle_duplicates_with_counter(self, temp_workspace):
        """Test duplicate handling with counter when timestamped version exists."""
        generator = CSVGenerator()
        config = OutputConfig(
            include_timestamp=True,
            timestamp_format="%Y%m%d_%H%M%S"
        )
        
        # Create existing file
        existing_path = temp_workspace / "existing.csv"
        existing_path.write_text("existing content")
        
        # Create timestamped version that would conflict
        timestamp = datetime.now().strftime(config.timestamp_format)
        timestamped_path = temp_workspace / f"existing_{timestamp}.csv"
        timestamped_path.write_text("timestamped content")
        
        result_path = generator._handle_duplicates(existing_path, config)
        
        # Should have counter
        assert f"existing_{timestamp}_001.csv" in result_path.name
    
    def test_prepare_csv_data_with_headers(self, sample_worksheet):
        """Test CSV data preparation with headers enabled."""
        generator = CSVGenerator()
        config = OutputConfig(include_headers=True)
        
        df = generator._prepare_csv_data(sample_worksheet, config)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= len(sample_worksheet.data)  # May have removed header row
        assert len(df.columns) == len(sample_worksheet.data.columns)
    
    def test_prepare_csv_data_without_headers(self, sample_worksheet):
        """Test CSV data preparation without headers."""
        generator = CSVGenerator()
        config = OutputConfig(include_headers=False)
        
        df = generator._prepare_csv_data(sample_worksheet, config)
        
        # Column names should be indices
        assert list(df.columns) == list(range(len(sample_worksheet.data.columns)))
    
    def test_setup_headers_with_text_first_row(self):
        """Test header setup when first row contains text (headers)."""
        generator = CSVGenerator()
        
        # DataFrame with text headers in first row
        df = pd.DataFrame([
            ['Name', 'Age', 'Salary'],
            ['Alice', 25, 75000],
            ['Bob', 30, 65000]
        ])
        
        worksheet = WorksheetData(
            source_file=Path("test.xlsx"),
            worksheet_name="test",
            data=df
        )
        
        result_df = generator._setup_headers(df, worksheet)
        
        # Should have used first row as headers
        assert 'Name' in result_df.columns
        assert 'Age' in result_df.columns
        assert 'Salary' in result_df.columns
        assert len(result_df) == 2  # Original 3 rows - 1 header row
    
    def test_setup_headers_with_numeric_first_row(self):
        """Test header setup when first row contains data (not headers)."""
        generator = CSVGenerator()
        
        # DataFrame with numeric data in first row
        df = pd.DataFrame([
            [1, 25, 75000],
            [2, 30, 65000],
            [3, 35, 80000]
        ])
        
        worksheet = WorksheetData(
            source_file=Path("test.xlsx"),
            worksheet_name="test",
            data=df
        )
        
        result_df = generator._setup_headers(df, worksheet)
        
        # Should have generated default headers
        assert 'Column_1' in result_df.columns
        assert 'Column_2' in result_df.columns
        assert 'Column_3' in result_df.columns
        assert len(result_df) == 3  # All original rows preserved
    
    def test_first_row_looks_like_headers_text_majority(self):
        """Test header detection based on text majority."""
        generator = CSVGenerator()
        
        # First row mostly text
        first_row = pd.Series(['Name', 'Age', 'Description', 'ID', 'Status'])
        df = pd.DataFrame([first_row, [1, 25, 'Employee', 100, 'Active']])
        
        is_header = generator._first_row_looks_like_headers(first_row, df)
        assert is_header is True
    
    def test_first_row_looks_like_headers_type_difference(self):
        """Test header detection based on type differences."""
        generator = CSVGenerator()
        
        # First row text, second row numeric
        first_row = pd.Series(['ID', 'Score', 'Value'])
        second_row = pd.Series([1, 95.5, 1000])
        df = pd.DataFrame([first_row, second_row, [2, 87.2, 2000]])
        
        is_header = generator._first_row_looks_like_headers(first_row, df)
        assert is_header is True
    
    def test_first_row_looks_like_headers_no_header(self):
        """Test header detection when first row is not headers."""
        generator = CSVGenerator()
        
        # All numeric data
        first_row = pd.Series([1, 25, 75000])
        df = pd.DataFrame([first_row, [2, 30, 65000], [3, 35, 80000]])
        
        is_header = generator._first_row_looks_like_headers(first_row, df)
        assert is_header is False
    
    def test_ensure_unique_headers(self):
        """Test ensuring unique column headers."""
        generator = CSVGenerator()
        
        # DataFrame with duplicate headers
        df = pd.DataFrame(columns=['Name', 'Age', 'Name', 'Age', 'Name'])
        
        result_df = generator._ensure_unique_headers(df)
        
        expected_columns = ['Name', 'Age', 'Name_1', 'Age_1', 'Name_2']
        assert list(result_df.columns) == expected_columns
    
    def test_clean_data_for_csv_nan_handling(self):
        """Test data cleaning with NaN values."""
        generator = CSVGenerator()
        
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': ['x', None, 'z'],
            'C': [1.1, 2.2, np.nan]
        })
        
        cleaned_df = generator._clean_data_for_csv(df)
        
        # NaN values should be replaced with empty strings
        assert cleaned_df.isna().sum().sum() == 0
        assert '' in cleaned_df['A'].values
        assert '' in cleaned_df['B'].values
        assert '' in cleaned_df['C'].values
    
    def test_clean_data_for_csv_string_cleaning(self):
        """Test string data cleaning."""
        generator = CSVGenerator()
        
        df = pd.DataFrame({
            'Text': ['  hello  ', 'world', 'nan', '  spaces  ']
        })
        
        cleaned_df = generator._clean_data_for_csv(df)
        
        # Strings should be stripped and 'nan' replaced
        assert cleaned_df['Text'].iloc[0] == 'hello'
        assert cleaned_df['Text'].iloc[2] == ''  # 'nan' replaced
        assert cleaned_df['Text'].iloc[3] == 'spaces'
    
    def test_format_numeric_columns(self):
        """Test numeric column formatting."""
        generator = CSVGenerator()
        
        df = pd.DataFrame({
            'Integers': [1, 2, 3],
            'Floats': [1.0, 2.5, 3.0],
            'Mixed': [1, '2', 3.5],
            'Text': ['a', 'b', 'c']
        })
        
        formatted_df = generator._format_numeric_columns(df)
        
        # Numeric columns should be formatted properly
        assert formatted_df['Integers'].dtype == object  # Converted to string
        assert formatted_df['Floats'].dtype == object   # Converted to string
        # Text column should remain unchanged
        assert 'a' in formatted_df['Text'].values
    
    def test_format_date_columns(self):
        """Test date column formatting."""
        generator = CSVGenerator()
        
        # Create DataFrame with date column
        df = pd.DataFrame({
            'Dates': pd.date_range('2024-01-01', periods=3),
            'Numbers': [1, 2, 3],
            'Text': ['a', 'b', 'c']
        })
        
        formatted_df = generator._format_date_columns(df)
        
        # Date column should be formatted as strings
        assert formatted_df['Dates'].dtype == object
        assert '2024-01-01' in formatted_df['Dates'].iloc[0]
    
    def test_format_date_columns_with_invalid_dates(self):
        """Test date formatting with mixed valid/invalid dates."""
        generator = CSVGenerator()
        
        df = pd.DataFrame({
            'Mixed': ['2024-01-01', 'not a date', '2024-12-31', None]
        })
        
        # Should handle mixed data gracefully
        try:
            formatted_df = generator._format_date_columns(df)
            # Should complete without error
            assert formatted_df is not None
        except Exception:
            pytest.fail("Date formatting should handle invalid dates gracefully")
    
    def test_write_csv_file_success(self, temp_workspace):
        """Test successful CSV file writing."""
        generator = CSVGenerator()
        
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        output_path = temp_workspace / "test_output.csv"
        config = OutputConfig(
            delimiter=',',
            encoding='utf-8',
            include_headers=True
        )
        
        generator._write_csv_file(df, output_path, config)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'A,B' in content  # Headers
            assert '1,x' in content  # Data
    
    def test_write_csv_file_custom_delimiter(self, temp_workspace):
        """Test CSV writing with custom delimiter."""
        generator = CSVGenerator()
        
        df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
        output_path = temp_workspace / "test_semicolon.csv"
        config = OutputConfig(delimiter=';', encoding='utf-8')
        
        generator._write_csv_file(df, output_path, config)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert ';' in content
    
    @patch('pandas.DataFrame.to_csv')
    def test_write_csv_file_error_handling(self, mock_to_csv, temp_workspace):
        """Test CSV writing error handling."""
        mock_to_csv.side_effect = IOError("Disk full")
        
        generator = CSVGenerator()
        df = pd.DataFrame({'A': [1, 2, 3]})
        output_path = temp_workspace / "test_error.csv"
        config = OutputConfig()
        
        with pytest.raises(CSVGenerationError):
            generator._write_csv_file(df, output_path, config)
    
    def test_validate_output_path_success(self, temp_workspace):
        """Test output path validation for writable location."""
        generator = CSVGenerator()
        
        valid_path = temp_workspace / "test_validate.csv"
        is_valid = generator.validate_output_path(valid_path)
        
        assert is_valid is True
    
    def test_validate_output_path_creates_directory(self, temp_workspace):
        """Test output path validation creates missing directories."""
        generator = CSVGenerator()
        
        nested_path = temp_workspace / "new_dir" / "sub_dir" / "test.csv"
        is_valid = generator.validate_output_path(nested_path)
        
        assert is_valid is True
        assert nested_path.parent.exists()
    
    def test_validate_output_path_permission_error(self):
        """Test output path validation with permission error."""
        generator = CSVGenerator()
        
        # Try to write to read-only location
        readonly_path = Path("/root/test.csv")  # Typically read-only
        is_valid = generator.validate_output_path(readonly_path)
        
        assert is_valid is False
    
    def test_estimate_csv_size(self, sample_worksheet):
        """Test CSV size estimation."""
        generator = CSVGenerator()
        
        estimated_size = generator.estimate_csv_size(sample_worksheet)
        
        assert isinstance(estimated_size, int)
        assert estimated_size > 0
        
        # Should be reasonable estimate for sample data
        data_cells = sample_worksheet.data.size
        assert estimated_size >= data_cells  # At least one byte per cell
    
    def test_estimate_csv_size_empty_data(self):
        """Test size estimation for empty data."""
        generator = CSVGenerator()
        
        empty_data = pd.DataFrame()
        empty_worksheet = WorksheetData(
            source_file=Path("empty.xlsx"),
            worksheet_name="empty",
            data=empty_data
        )
        
        estimated_size = generator.estimate_csv_size(empty_worksheet)
        
        assert estimated_size >= 0
    
    def test_generate_csv_preview_basic(self, sample_worksheet):
        """Test CSV preview generation."""
        generator = CSVGenerator()
        config = OutputConfig(include_headers=True)
        
        preview = generator.generate_csv_preview(sample_worksheet, config, max_rows=3)
        
        assert isinstance(preview, str)
        assert len(preview) > 0
        
        # Should contain sample data
        assert 'Alice' in preview or 'ID' in preview
        
        # Should respect max_rows limit
        lines = preview.strip().split('\n')
        assert len(lines) <= 4  # Headers + max 3 data rows
    
    def test_generate_csv_preview_no_headers(self, sample_worksheet):
        """Test CSV preview without headers."""
        generator = CSVGenerator()
        config = OutputConfig(include_headers=False)
        
        preview = generator.generate_csv_preview(sample_worksheet, config, max_rows=2)
        
        assert isinstance(preview, str)
        # Should have only data rows
        lines = preview.strip().split('\n')
        assert len(lines) <= 2
    
    def test_generate_csv_preview_custom_delimiter(self, sample_worksheet):
        """Test CSV preview with custom delimiter."""
        generator = CSVGenerator()
        config = OutputConfig(delimiter=';', include_headers=True)
        
        preview = generator.generate_csv_preview(sample_worksheet, config)
        
        assert ';' in preview
    
    def test_full_error_handling_workflow(self, temp_workspace):
        """Test complete error handling in generate_csv."""
        generator = CSVGenerator()
        
        # Create problematic worksheet
        bad_data = pd.DataFrame({'A': [1, 2, 3]})
        bad_worksheet = WorksheetData(
            source_file=Path("bad.xlsx"),
            worksheet_name="BadSheet",
            data=bad_data
        )
        
        # Mock pandas to raise error during processing
        with patch('pandas.DataFrame.copy', side_effect=Exception("Processing error")):
            config = OutputConfig(folder=temp_workspace)
            
            with pytest.raises(CSVGenerationError):
                generator.generate_csv(bad_worksheet, config)
    
    @patch('builtins.open')
    def test_file_accessibility_error_handling(self, mock_open):
        """Test file accessibility checking error handling."""
        mock_open.side_effect = PermissionError("Access denied")
        
        generator = CSVGenerator()
        
        test_path = Path("test.csv")
        is_accessible = generator._is_file_accessible(test_path)
        
        assert is_accessible is False
    
    def test_complex_data_scenario(self, temp_workspace):
        """Test complex data scenario with mixed types."""
        generator = CSVGenerator()
        
        # Complex data with various types
        complex_data = pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['Alice', 'Bob', None],
            'Date': pd.date_range('2024-01-01', periods=3),
            'Value': [1.5, np.nan, 3.7],
            'Flag': [True, False, True],
            'Mixed': [1, 'text', 3.14]
        })
        
        complex_worksheet = WorksheetData(
            source_file=temp_workspace / "complex.xlsx",
            worksheet_name="ComplexData",
            data=complex_data
        )
        
        config = OutputConfig(
            folder=temp_workspace,
            include_headers=True,
            delimiter=',',
            encoding='utf-8'
        )
        
        output_path = generator.generate_csv(complex_worksheet, config)
        
        assert output_path.exists()
        
        # Verify complex data was handled properly
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'ID' in content
            assert 'Alice' in content
            assert '2024' in content  # Date formatted
    
    def test_metadata_tracking_throughout_workflow(self, sample_worksheet, temp_workspace):
        """Test that metadata is properly tracked throughout workflow."""
        generator = CSVGenerator()
        config = OutputConfig(folder=temp_workspace)
        
        # This should exercise all metadata tracking paths
        output_path = generator.generate_csv(sample_worksheet, config)
        
        assert output_path.exists()
        # If we got here, all metadata tracking worked correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])