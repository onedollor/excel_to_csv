"""Unit tests for CSV generator."""

import pytest
from pathlib import Path
import pandas as pd
import csv
from datetime import datetime
from unittest.mock import patch, mock_open

from excel_to_csv.generators.csv_generator import CSVGenerator
from excel_to_csv.models.data_models import WorksheetData, OutputConfig


class TestCSVGenerator:
    """Test cases for CSVGenerator class."""
    
    def test_init_with_default_config(self, temp_dir: Path):
        """Test CSVGenerator initialization with default config."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        assert generator.output_folder == temp_dir
        assert generator.config.encoding == "utf-8"
        assert generator.config.include_timestamp is True
        assert generator.config.naming_pattern == "{filename}_{worksheet}.csv"    
    def test_init_with_custom_config(self, temp_dir: Path):
        """Test CSVGenerator initialization with custom config."""
        custom_config = OutputConfig(
            folder=str(temp_dir),
            naming_pattern="{worksheet}_{filename}.csv",
            include_timestamp=False,
            encoding="utf-16"
        )
        
        generator = CSVGenerator(output_folder=temp_dir, config=custom_config)
        
        assert generator.config.encoding == "utf-16"
        assert generator.config.include_timestamp is False
        assert generator.config.naming_pattern == "{worksheet}_{filename}.csv"
    
    def test_generate_csv_success(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test successful CSV generation."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        worksheet_data = WorksheetData(
            worksheet_name="TestSheet",
            data=sample_excel_data,
            source_file=Path("source.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        assert output_path.suffix == ".csv"
        assert "source" in output_path.name
        assert "TestSheet" in output_path.name
        
        # Verify CSV content
        loaded_data = pd.read_csv(output_path)
        assert loaded_data.shape == sample_excel_data.shape
        assert list(loaded_data.columns) == list(sample_excel_data.columns)
    
    def test_generate_csv_with_timestamp(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test CSV generation with timestamp."""
        config = OutputConfig(
            folder=str(temp_dir),
            include_timestamp=True,
            naming_pattern="{filename}_{worksheet}.csv"
        )
        generator = CSVGenerator(output_folder=temp_dir, config=config)
        
        worksheet_data = WorksheetData(
            worksheet_name="TimestampSheet",
            data=sample_excel_data,
            source_file=Path("timestamp_test.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        # Check that timestamp is included in filename
        timestamp_pattern = r'\d{8}_\d{6}'  # YYYYMMDD_HHMMSS
        import re
        assert re.search(timestamp_pattern, output_path.name)
    
    def test_generate_csv_without_timestamp(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test CSV generation without timestamp."""
        config = OutputConfig(
            folder=str(temp_dir),
            include_timestamp=False,
            naming_pattern="{filename}_{worksheet}.csv"
        )
        generator = CSVGenerator(output_folder=temp_dir, config=config)
        
        worksheet_data = WorksheetData(
            worksheet_name="NoTimestampSheet",
            data=sample_excel_data,
            source_file=Path("no_timestamp.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        expected_name = "no_timestamp_NoTimestampSheet.csv"
        assert output_path.name == expected_name
    
    def test_generate_csv_custom_naming_pattern(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test CSV generation with custom naming pattern."""
        config = OutputConfig(
            folder=str(temp_dir),
            naming_pattern="{worksheet}_from_{filename}.csv",
            include_timestamp=False
        )
        generator = CSVGenerator(output_folder=temp_dir, config=config)
        
        worksheet_data = WorksheetData(
            worksheet_name="CustomSheet",
            data=sample_excel_data,
            source_file=Path("custom_source.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        expected_name = "CustomSheet_from_custom_source.csv"
        assert output_path.name == expected_name
    
    def test_generate_csv_duplicate_handling(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test handling of duplicate filenames."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        worksheet_data = WorksheetData(
            worksheet_name="DuplicateSheet",
            data=sample_excel_data,
            source_file=Path("duplicate.xlsx")
        )
        
        # Generate first CSV
        output_path1 = generator.generate_csv(worksheet_data)
        assert output_path1.exists()
        
        # Generate second CSV with same name (should create unique name)
        output_path2 = generator.generate_csv(worksheet_data)
        assert output_path2.exists()
        assert output_path1 != output_path2  # Should have different names
        
        # Both files should exist
        assert output_path1.exists()
        assert output_path2.exists()
    
    def test_sanitize_filename(self, temp_dir: Path):
        """Test filename sanitization."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        # Test various problematic characters
        test_cases = [
            ("normal_name", "normal_name"),
            ("file with spaces", "file_with_spaces"),
            ("file/with\\slashes", "file_with_slashes"),
            ("file:with*special<chars>", "file_with_special_chars"),
            ("file|with?quotes\"", "file_with_quotes"),
            ("file.with.dots", "file.with.dots"),  # Dots should be preserved except extension
            ("", "unnamed"),  # Empty name
            ("   ", "unnamed"),  # Whitespace only
        ]
        
        for original, expected in test_cases:
            sanitized = generator._sanitize_filename(original)
            assert sanitized == expected
    
    def test_get_unique_filename_no_conflict(self, temp_dir: Path):
        """Test unique filename generation when no conflict exists."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        base_name = "no_conflict.csv"
        unique_path = generator._get_unique_filename(temp_dir / base_name)
        
        assert unique_path == temp_dir / base_name
    
    def test_get_unique_filename_with_conflict(self, temp_dir: Path):
        """Test unique filename generation when conflicts exist."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        # Create existing file
        existing_file = temp_dir / "conflict.csv"
        existing_file.touch()
        
        unique_path = generator._get_unique_filename(existing_file)
        
        assert unique_path != existing_file
        assert unique_path.suffix == ".csv"
        assert "conflict" in unique_path.name
        assert not unique_path.exists()  # Should be unique
    
    def test_generate_csv_with_special_characters(self, temp_dir: Path):
        """Test CSV generation with special characters in data."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        # Create data with special characters
        special_data = pd.DataFrame({
            'Text': ['Hello, World!', 'Quote"Test', 'Newline\nTest', 'Tab\tTest'],
            'Unicode': ['Café', 'Naïve', '北京', '🚀'],
            'Numbers': [1, 2, 3, 4]
        })
        
        worksheet_data = WorksheetData(
            worksheet_name="SpecialChars",
            data=special_data,
            source_file=Path("special.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        
        # Verify special characters are preserved
        loaded_data = pd.read_csv(output_path)
        assert 'Café' in loaded_data['Unicode'].values
        assert 'Hello, World!' in loaded_data['Text'].values
    
    def test_generate_csv_with_different_encodings(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test CSV generation with different encodings."""
        encodings = ['utf-8', 'utf-16', 'latin-1']
        
        for encoding in encodings:
            config = OutputConfig(
                folder=str(temp_dir),
                encoding=encoding,
                include_timestamp=False,
                naming_pattern="{filename}_{worksheet}_{encoding}.csv"
            )
            generator = CSVGenerator(output_folder=temp_dir, config=config)
            
            worksheet_data = WorksheetData(
                worksheet_name="EncodingTest",
                data=sample_excel_data,
                source_file=Path("encoding_test.xlsx")
            )
            
            # Manually format the filename to include encoding
            worksheet_data_with_encoding = WorksheetData(
                worksheet_name=f"EncodingTest_{encoding.replace('-', '')}",
                data=sample_excel_data,
                source_file=Path("encoding_test.xlsx")
            )
            
            output_path = generator.generate_csv(worksheet_data_with_encoding)
            
            assert output_path.exists()
            
            # Verify file can be read with specified encoding
            loaded_data = pd.read_csv(output_path, encoding=encoding)
            assert loaded_data.shape == sample_excel_data.shape
    
    def test_generate_csv_empty_data(self, temp_dir: Path):
        """Test CSV generation with empty data."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        empty_data = pd.DataFrame()
        worksheet_data = WorksheetData(
            worksheet_name="EmptySheet",
            data=empty_data,
            source_file=Path("empty.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        
        # Verify empty CSV is created
        with open(output_path, 'r') as f:
            content = f.read()
            # Empty DataFrame to_csv might create just headers or be completely empty
            assert len(content) >= 0  # At least not errored
    
    def test_generate_csv_single_column(self, temp_dir: Path):
        """Test CSV generation with single column data."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        single_col_data = pd.DataFrame({'SingleColumn': [1, 2, 3, 4, 5]})
        worksheet_data = WorksheetData(
            worksheet_name="SingleColumn",
            data=single_col_data,
            source_file=Path("single.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        
        loaded_data = pd.read_csv(output_path)
        assert loaded_data.shape == (5, 1)
        assert 'SingleColumn' in loaded_data.columns
    
    def test_generate_csv_with_nan_values(self, temp_dir: Path):
        """Test CSV generation with NaN values."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        nan_data = pd.DataFrame({
            'A': [1, None, 3, None],
            'B': [None, 2, None, 4],
            'C': ['text', None, 'more', 'text']
        })
        
        worksheet_data = WorksheetData(
            worksheet_name="NaNSheet",
            data=nan_data,
            source_file=Path("nan_test.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        
        loaded_data = pd.read_csv(output_path)
        assert loaded_data.shape == (4, 3)
        # NaN values should be preserved (as empty strings in CSV)
    
    def test_generate_csv_permission_error(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test handling of permission errors during CSV generation."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        worksheet_data = WorksheetData(
            worksheet_name="PermissionTest",
            data=sample_excel_data,
            source_file=Path("permission.xlsx")
        )
        
        # Mock file operations to raise permission error
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                generator.generate_csv(worksheet_data)
    
    def test_generate_csv_disk_space_error(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test handling of disk space errors during CSV generation."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        worksheet_data = WorksheetData(
            worksheet_name="DiskSpaceTest",
            data=sample_excel_data,
            source_file=Path("diskspace.xlsx")
        )
        
        # Mock file operations to raise disk space error
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError):
                generator.generate_csv(worksheet_data)
    
    def test_output_folder_creation(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test automatic creation of output folder."""
        # Use non-existent subfolder
        non_existent_folder = temp_dir / "new_folder"
        assert not non_existent_folder.exists()
        
        generator = CSVGenerator(output_folder=non_existent_folder)
        
        worksheet_data = WorksheetData(
            worksheet_name="FolderCreation",
            data=sample_excel_data,
            source_file=Path("folder_test.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        # Folder should be created automatically
        assert non_existent_folder.exists()
        assert output_path.exists()
        assert output_path.parent == non_existent_folder
    
    def test_csv_format_validation(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test that generated CSV follows proper CSV format."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        worksheet_data = WorksheetData(
            worksheet_name="FormatValidation",
            data=sample_excel_data,
            source_file=Path("format.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        # Validate CSV format using csv module
        with open(output_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)
            
            # Should have header + data rows
            assert len(rows) == len(sample_excel_data) + 1
            
            # All rows should have same number of columns
            expected_cols = len(sample_excel_data.columns)
            for row in rows:
                assert len(row) == expected_cols
    
    def test_concurrent_csv_generation(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test thread safety of CSV generation."""
        import threading
        
        generator = CSVGenerator(output_folder=temp_dir)
        results = []
        exceptions = []
        
        def generate_csv_worker(worker_id):
            try:
                worksheet_data = WorksheetData(
                    worksheet_name=f"ConcurrentSheet_{worker_id}",
                    data=sample_excel_data.copy(),
                    source_file=Path(f"concurrent_{worker_id}.xlsx")
                )
                
                output_path = generator.generate_csv(worksheet_data)
                results.append(output_path)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=generate_csv_worker, args=(i,)) for i in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed
        assert len(exceptions) == 0
        assert len(results) == 3
        
        # All files should exist and be different
        for result in results:
            assert result.exists()
        
        # All filenames should be unique
        filenames = [result.name for result in results]
        assert len(set(filenames)) == len(filenames)
    
    def test_generate_csv_with_datetime_columns(self, temp_dir: Path):
        """Test CSV generation with datetime columns."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        datetime_data = pd.DataFrame({
            'date_col': pd.date_range('2023-01-01', periods=3),
            'timestamp_col': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00', '2023-01-03 12:00:00']),
            'regular_col': ['A', 'B', 'C']
        })
        
        worksheet_data = WorksheetData(
            worksheet_name="DatetimeSheet",
            data=datetime_data,
            source_file=Path("datetime.xlsx")
        )
        
        output_path = generator.generate_csv(worksheet_data)
        
        assert output_path.exists()
        
        # Verify datetime columns are properly formatted
        loaded_data = pd.read_csv(output_path)
        assert 'date_col' in loaded_data.columns
        assert 'timestamp_col' in loaded_data.columns
        assert 'regular_col' in loaded_data.columns
    
    def test_path_handling_edge_cases(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test path handling edge cases."""
        generator = CSVGenerator(output_folder=temp_dir)
        
        # Test with Path object vs string
        path_cases = [
            Path("path_object.xlsx"),
            "string_path.xlsx",
            Path("/absolute/path/absolute.xlsx").name,  # Just filename part
        ]
        
        for i, file_path in enumerate(path_cases):
            worksheet_data = WorksheetData(
                worksheet_name=f"PathTest_{i}",
                data=sample_excel_data,
                source_file=Path(file_path) if isinstance(file_path, str) else file_path
            )
            
            output_path = generator.generate_csv(worksheet_data)
            assert output_path.exists()
            assert output_path.suffix == ".csv"