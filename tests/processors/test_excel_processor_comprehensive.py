"""Comprehensive tests for Excel Processor with high coverage."""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import openpyxl
from openpyxl.utils.exceptions import InvalidFileException
import logging
import os

from excel_to_csv.processors.excel_processor import ExcelProcessor, ExcelProcessingError
from excel_to_csv.models.data_models import WorksheetData


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_excel_data():
    """Create sample Excel data."""
    return {
        'Sheet1': pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Age': [25, 30, 35, 28, 22],
            'Salary': [75000, 65000, 80000, 70000, 60000]
        }),
        'Sheet2': pd.DataFrame({
            'Product': ['Widget A', 'Widget B', 'Widget C'],
            'Price': [10.99, 15.50, 8.75],
            'Stock': [100, 50, 200]
        })
    }


@pytest.fixture
def sample_excel_file(temp_workspace, sample_excel_data):
    """Create a sample Excel file for testing."""
    file_path = temp_workspace / "sample.xlsx"
    
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, data in sample_excel_data.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return file_path


@pytest.fixture
def large_excel_file(temp_workspace):
    """Create a large Excel file for size testing."""
    file_path = temp_workspace / "large.xlsx"
    
    # Create data that results in file > 100MB when written (typical default limit)
    large_data = pd.DataFrame({
        'Column1': ['X' * 1000] * 1000,  # 1000 rows of 1000-char strings
        'Column2': list(range(1000)),
        'Column3': [3.14159] * 1000
    })
    
    # Note: This might not actually exceed 100MB, but we'll mock the size check
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        large_data.to_excel(writer, sheet_name='LargeSheet', index=False)
    
    return file_path


class TestExcelProcessorInitialization:
    """Test Excel Processor initialization."""
    
    def test_initialization_with_defaults(self):
        """Test ExcelProcessor initialization with default parameters."""
        processor = ExcelProcessor()
        
        assert processor.max_file_size_mb == 100
        assert processor.chunk_size == 10000
        assert hasattr(processor, 'logger')
        assert processor.SUPPORTED_EXTENSIONS == {'.xlsx', '.xls'}
        assert processor.METADATA_PREVIEW_ROWS == 100
    
    def test_initialization_with_custom_parameters(self):
        """Test ExcelProcessor initialization with custom parameters."""
        processor = ExcelProcessor(max_file_size_mb=200, chunk_size=5000)
        
        assert processor.max_file_size_mb == 200
        assert processor.chunk_size == 5000
        assert hasattr(processor, 'logger')


class TestFileValidation:
    """Test file validation functionality."""
    
    def test_validate_file_success(self, sample_excel_file):
        """Test successful file validation."""
        processor = ExcelProcessor()
        
        # This should not raise any exceptions
        processor._validate_file(sample_excel_file)
    
    def test_validate_file_not_found(self, temp_workspace):
        """Test validation fails for non-existent file."""
        processor = ExcelProcessor()
        nonexistent_file = temp_workspace / "does_not_exist.xlsx"
        
        with pytest.raises(ExcelProcessingError, match="File not found"):
            processor._validate_file(nonexistent_file)
    
    def test_validate_file_not_a_file(self, temp_workspace):
        """Test validation fails when path is a directory."""
        processor = ExcelProcessor()
        directory_path = temp_workspace / "directory"
        directory_path.mkdir()
        
        with pytest.raises(ExcelProcessingError, match="Path is not a file"):
            processor._validate_file(directory_path)
    
    def test_validate_file_unsupported_extension(self, temp_workspace):
        """Test validation fails for unsupported file extensions."""
        processor = ExcelProcessor()
        unsupported_file = temp_workspace / "document.txt"
        unsupported_file.write_text("This is not an Excel file")
        
        with pytest.raises(ExcelProcessingError, match="Unsupported file extension"):
            processor._validate_file(unsupported_file)
    
    def test_validate_file_too_large(self, temp_workspace):
        """Test validation fails for files that are too large."""
        processor = ExcelProcessor(max_file_size_mb=1)  # Very small limit
        test_file = temp_workspace / "large.xlsx"
        
        # Create a file larger than 1MB
        large_content = "X" * (2 * 1024 * 1024)  # 2MB of content
        test_file.write_text(large_content)
        
        with pytest.raises(ExcelProcessingError, match="File too large"):
            processor._validate_file(test_file)
    
    def test_validate_file_not_readable(self, temp_workspace):
        """Test validation fails when file is not readable."""
        processor = ExcelProcessor()
        test_file = temp_workspace / "readable.xlsx"
        test_file.write_text("Some content")
        
        # Mock open to raise permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(ExcelProcessingError, match="Cannot read file"):
                processor._validate_file(test_file)
    
    def test_validate_file_io_error(self, temp_workspace):
        """Test validation handles IO errors."""
        processor = ExcelProcessor()
        test_file = temp_workspace / "io_error.xlsx"
        test_file.write_text("Some content")
        
        # Mock open to raise IO error
        with patch('builtins.open', side_effect=IOError("I/O error")):
            with pytest.raises(ExcelProcessingError, match="Cannot read file"):
                processor._validate_file(test_file)


class TestExcelFileProcessing:
    """Test main Excel file processing functionality."""
    
    def test_process_file_success(self, sample_excel_file, sample_excel_data):
        """Test successful Excel file processing."""
        processor = ExcelProcessor()
        
        worksheets = processor.process_file(sample_excel_file)
        
        assert len(worksheets) == 2  # Sheet1 and Sheet2
        
        # Check first worksheet
        sheet1 = next(ws for ws in worksheets if ws.worksheet_name == 'Sheet1')
        assert sheet1.row_count == 5
        assert sheet1.column_count == 4
        assert len(sheet1.data) == 5
        
        # Check second worksheet
        sheet2 = next(ws for ws in worksheets if ws.worksheet_name == 'Sheet2')
        assert sheet2.row_count == 3
        assert sheet2.column_count == 3
        assert len(sheet2.data) == 3
    
    def test_process_file_with_string_path(self, sample_excel_file):
        """Test processing file with string path instead of Path object."""
        processor = ExcelProcessor()
        
        worksheets = processor.process_file(str(sample_excel_file))
        
        assert len(worksheets) == 2
        assert all(isinstance(ws, WorksheetData) for ws in worksheets)
    
    def test_process_file_validation_failure(self, temp_workspace):
        """Test that processing fails when validation fails."""
        processor = ExcelProcessor()
        nonexistent_file = temp_workspace / "does_not_exist.xlsx"
        
        with pytest.raises(ExcelProcessingError, match="Excel processing failed"):
            processor.process_file(nonexistent_file)
    
    def test_process_file_extraction_failure(self, sample_excel_file):
        """Test processing handles worksheet extraction failures."""
        processor = ExcelProcessor()
        
        # Mock _extract_worksheets to raise an exception
        with patch.object(processor, '_extract_worksheets', side_effect=RuntimeError("Extraction failed")):
            with pytest.raises(ExcelProcessingError, match="Excel processing failed"):
                processor.process_file(sample_excel_file)


class TestWorksheetExtraction:
    """Test worksheet extraction functionality."""
    
    def test_extract_worksheets_success(self, sample_excel_file):
        """Test successful worksheet extraction."""
        processor = ExcelProcessor()
        
        worksheets = processor._extract_worksheets(sample_excel_file)
        
        assert len(worksheets) == 2
        worksheet_names = [ws.worksheet_name for ws in worksheets]
        assert 'Sheet1' in worksheet_names
        assert 'Sheet2' in worksheet_names
    
    def test_extract_worksheets_invalid_file(self, temp_workspace):
        """Test worksheet extraction with invalid Excel file."""
        processor = ExcelProcessor()
        
        # Create file with .xlsx extension but invalid content
        invalid_file = temp_workspace / "invalid.xlsx"
        invalid_file.write_text("This is not a valid Excel file")
        
        # Should handle the error gracefully during extraction
        with pytest.raises(Exception):  # Will be wrapped in ExcelProcessingError by process_file
            processor._extract_worksheets(invalid_file)
    
    def test_extract_worksheets_empty_file(self, temp_workspace):
        """Test extraction from empty Excel file."""
        processor = ExcelProcessor()
        
        # Create valid but empty Excel file
        empty_file = temp_workspace / "empty.xlsx"
        empty_data = pd.DataFrame()
        
        with pd.ExcelWriter(empty_file, engine='openpyxl') as writer:
            empty_data.to_excel(writer, sheet_name='EmptySheet', index=False)
        
        worksheets = processor._extract_worksheets(empty_file)
        
        assert len(worksheets) >= 1
        empty_sheet = worksheets[0]
        assert empty_sheet.worksheet_name == 'EmptySheet'
        assert empty_sheet.row_count == 0


class TestWorkbookInfo:
    """Test workbook information extraction."""
    
    def test_get_workbook_info_success(self, sample_excel_file):
        """Test successful workbook info extraction."""
        processor = ExcelProcessor()
        
        workbook_info = processor._get_workbook_info(sample_excel_file)
        
        assert isinstance(workbook_info, dict)
        assert 'Sheet1' in workbook_info
        assert 'Sheet2' in workbook_info
        
        # Check structure of sheet info
        sheet1_info = workbook_info['Sheet1']
        assert 'max_row' in sheet1_info
        assert 'max_column' in sheet1_info
        assert 'sheet_state' in sheet1_info
    
    def test_get_workbook_info_invalid_file(self, temp_workspace):
        """Test workbook info extraction with invalid file."""
        processor = ExcelProcessor()
        
        invalid_file = temp_workspace / "invalid.xlsx"
        invalid_file.write_text("Not an Excel file")
        
        # Should handle invalid file gracefully
        with pytest.raises(Exception):  # openpyxl will raise InvalidFileException
            processor._get_workbook_info(invalid_file)


class TestWorksheetProcessing:
    """Test individual worksheet processing."""
    
    def test_process_worksheet_success(self, sample_excel_file):
        """Test successful worksheet processing."""
        processor = ExcelProcessor()
        
        # First get workbook info
        workbook_info = processor._get_workbook_info(sample_excel_file)
        sheet_info = workbook_info['Sheet1']
        
        worksheet_data = processor._process_worksheet(
            sample_excel_file, 'Sheet1', sheet_info
        )
        
        assert isinstance(worksheet_data, WorksheetData)
        assert worksheet_data.worksheet_name == 'Sheet1'
        assert worksheet_data.source_file == sample_excel_file
        assert worksheet_data.row_count > 0
        assert worksheet_data.column_count > 0
        assert worksheet_data.data is not None
    
    def test_process_worksheet_with_metadata(self, sample_excel_file):
        """Test worksheet processing includes proper metadata."""
        processor = ExcelProcessor()
        
        workbook_info = processor._get_workbook_info(sample_excel_file)
        sheet_info = workbook_info['Sheet1']
        
        worksheet_data = processor._process_worksheet(
            sample_excel_file, 'Sheet1', sheet_info
        )
        
        # Check metadata fields
        assert hasattr(worksheet_data, 'has_headers')
        assert hasattr(worksheet_data, 'column_types')
        assert hasattr(worksheet_data, 'metadata')
        assert isinstance(worksheet_data.metadata, dict)


class TestDataReading:
    """Test worksheet data reading functionality."""
    
    def test_read_worksheet_data_success(self, sample_excel_file):
        """Test successful worksheet data reading."""
        processor = ExcelProcessor()
        
        # Create ExcelFile object
        excel_file = pd.ExcelFile(sample_excel_file)
        
        data = processor._read_worksheet_data(excel_file, 'Sheet1')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert len(data.columns) > 0
    
    def test_read_worksheet_data_nonexistent_sheet(self, sample_excel_file):
        """Test reading data from non-existent worksheet."""
        processor = ExcelProcessor()
        
        excel_file = pd.ExcelFile(sample_excel_file)
        
        with pytest.raises(Exception):  # Should raise ValueError or similar
            processor._read_worksheet_data(excel_file, 'NonexistentSheet')
    
    def test_read_worksheet_data_empty_sheet(self, temp_workspace):
        """Test reading data from empty worksheet."""
        processor = ExcelProcessor()
        
        # Create file with empty sheet
        empty_file = temp_workspace / "empty_sheet.xlsx"
        empty_data = pd.DataFrame()
        
        with pd.ExcelWriter(empty_file, engine='openpyxl') as writer:
            empty_data.to_excel(writer, sheet_name='EmptySheet', index=False)
        
        excel_file = pd.ExcelFile(empty_file)
        data = processor._read_worksheet_data(excel_file, 'EmptySheet')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 0


class TestMetadataEnhancement:
    """Test metadata enhancement functionality."""
    
    def test_enhance_metadata_basic(self):
        """Test basic metadata enhancement."""
        processor = ExcelProcessor()
        
        # Create sample dataframe
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'string_col': ['a', 'b', 'c']
        })
        
        base_metadata = {'test_key': 'test_value'}
        
        enhanced = processor._enhance_metadata(df, base_metadata)
        
        assert isinstance(enhanced, dict)
        assert 'test_key' in enhanced
        assert 'data_types' in enhanced
        assert 'column_stats' in enhanced
        assert 'memory_usage' in enhanced
    
    def test_enhance_metadata_with_nulls(self):
        """Test metadata enhancement with null values."""
        processor = ExcelProcessor()
        
        df = pd.DataFrame({
            'col_with_nulls': [1, None, 3, None, 5],
            'col_no_nulls': [1, 2, 3, 4, 5]
        })
        
        enhanced = processor._enhance_metadata(df, {})
        
        assert 'null_counts' in enhanced
        assert enhanced['null_counts']['col_with_nulls'] > 0
        assert enhanced['null_counts']['col_no_nulls'] == 0
    
    def test_enhance_metadata_empty_dataframe(self):
        """Test metadata enhancement with empty dataframe."""
        processor = ExcelProcessor()
        
        df = pd.DataFrame()
        enhanced = processor._enhance_metadata(df, {})
        
        assert isinstance(enhanced, dict)
        # Should handle empty dataframe gracefully


class TestWorksheetPreview:
    """Test worksheet preview functionality."""
    
    def test_get_worksheet_preview_success(self, sample_excel_file):
        """Test successful worksheet preview."""
        processor = ExcelProcessor()
        
        preview = processor.get_worksheet_preview(sample_excel_file, 'Sheet1')
        
        assert isinstance(preview, dict)
        assert 'worksheet_name' in preview
        assert 'preview_data' in preview
        assert 'metadata' in preview
        assert preview['worksheet_name'] == 'Sheet1'
    
    def test_get_worksheet_preview_with_rows_limit(self, sample_excel_file):
        """Test worksheet preview with custom row limit."""
        processor = ExcelProcessor()
        
        preview = processor.get_worksheet_preview(sample_excel_file, 'Sheet1', preview_rows=2)
        
        assert len(preview['preview_data']) <= 2
    
    def test_get_worksheet_preview_nonexistent_sheet(self, sample_excel_file):
        """Test preview of non-existent worksheet."""
        processor = ExcelProcessor()
        
        with pytest.raises(ExcelProcessingError, match="Worksheet not found"):
            processor.get_worksheet_preview(sample_excel_file, 'NonexistentSheet')
    
    def test_get_worksheet_preview_invalid_file(self, temp_workspace):
        """Test preview with invalid file."""
        processor = ExcelProcessor()
        
        invalid_file = temp_workspace / "invalid.xlsx"
        invalid_file.write_text("Not Excel content")
        
        with pytest.raises(ExcelProcessingError):
            processor.get_worksheet_preview(invalid_file, 'Sheet1')


class TestWorksheetListing:
    """Test worksheet listing functionality."""
    
    def test_list_worksheets_success(self, sample_excel_file):
        """Test successful worksheet listing."""
        processor = ExcelProcessor()
        
        sheet_names = processor.list_worksheets(sample_excel_file)
        
        assert isinstance(sheet_names, list)
        assert 'Sheet1' in sheet_names
        assert 'Sheet2' in sheet_names
        assert len(sheet_names) == 2
    
    def test_list_worksheets_string_path(self, sample_excel_file):
        """Test listing worksheets with string path."""
        processor = ExcelProcessor()
        
        sheet_names = processor.list_worksheets(str(sample_excel_file))
        
        assert isinstance(sheet_names, list)
        assert len(sheet_names) == 2
    
    def test_list_worksheets_invalid_file(self, temp_workspace):
        """Test listing worksheets from invalid file."""
        processor = ExcelProcessor()
        
        invalid_file = temp_workspace / "invalid.xlsx"
        invalid_file.write_text("Not Excel content")
        
        with pytest.raises(ExcelProcessingError):
            processor.list_worksheets(invalid_file)
    
    def test_list_worksheets_single_sheet(self, temp_workspace):
        """Test listing worksheets from single-sheet file."""
        processor = ExcelProcessor()
        
        single_sheet_file = temp_workspace / "single.xlsx"
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pd.ExcelWriter(single_sheet_file, engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name='OnlySheet', index=False)
        
        sheet_names = processor.list_worksheets(single_sheet_file)
        
        assert len(sheet_names) == 1
        assert sheet_names[0] == 'OnlySheet'


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_corrupted_excel_file(self, temp_workspace):
        """Test handling of corrupted Excel files."""
        processor = ExcelProcessor()
        
        # Create file with .xlsx extension but corrupted content
        corrupted_file = temp_workspace / "corrupted.xlsx"
        corrupted_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')  # Invalid binary content
        
        with pytest.raises(ExcelProcessingError):
            processor.process_file(corrupted_file)
    
    def test_locked_excel_file(self, sample_excel_file):
        """Test handling of locked Excel files."""
        processor = ExcelProcessor()
        
        # Mock openpyxl to raise permission error
        with patch('openpyxl.load_workbook', side_effect=PermissionError("File is locked")):
            with pytest.raises(ExcelProcessingError):
                processor.process_file(sample_excel_file)
    
    def test_memory_error_handling(self, sample_excel_file):
        """Test handling of memory errors during processing."""
        processor = ExcelProcessor()
        
        # Mock pandas to raise memory error
        with patch('pandas.read_excel', side_effect=MemoryError("Out of memory")):
            with pytest.raises(ExcelProcessingError):
                processor.process_file(sample_excel_file)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_wide_spreadsheet(self, temp_workspace):
        """Test processing spreadsheet with many columns."""
        processor = ExcelProcessor()
        
        # Create spreadsheet with 100 columns
        wide_data = pd.DataFrame({f'Col{i}': [i] * 10 for i in range(100)})
        wide_file = temp_workspace / "wide.xlsx"
        
        with pd.ExcelWriter(wide_file, engine='openpyxl') as writer:
            wide_data.to_excel(writer, sheet_name='WideSheet', index=False)
        
        worksheets = processor.process_file(wide_file)
        
        assert len(worksheets) == 1
        assert worksheets[0].column_count == 100
    
    def test_spreadsheet_with_special_characters(self, temp_workspace):
        """Test processing spreadsheet with special characters."""
        processor = ExcelProcessor()
        
        special_data = pd.DataFrame({
            'Unicode': ['α', 'β', 'γ', '中文', '🚀'],
            'Symbols': ['@#$%', '&*()=', '[]{}', '<>', '|\\'],
            'Numbers': [1.5, -2.7, 3.14159, 0, 999999]
        })
        
        special_file = temp_workspace / "special.xlsx"
        
        with pd.ExcelWriter(special_file, engine='openpyxl') as writer:
            special_data.to_excel(writer, sheet_name='SpecialChars', index=False)
        
        worksheets = processor.process_file(special_file)
        
        assert len(worksheets) == 1
        assert worksheets[0].row_count == 5
        assert worksheets[0].column_count == 3
    
    def test_spreadsheet_with_mixed_data_types(self, temp_workspace):
        """Test processing spreadsheet with mixed data types."""
        processor = ExcelProcessor()
        
        mixed_data = pd.DataFrame({
            'Integers': [1, 2, 3],
            'Floats': [1.1, 2.2, 3.3],
            'Strings': ['a', 'b', 'c'],
            'Booleans': [True, False, True],
            'Dates': pd.date_range('2024-01-01', periods=3),
            'Mixed': [1, 'text', 3.14]
        })
        
        mixed_file = temp_workspace / "mixed.xlsx"
        
        with pd.ExcelWriter(mixed_file, engine='openpyxl') as writer:
            mixed_data.to_excel(writer, sheet_name='MixedTypes', index=False)
        
        worksheets = processor.process_file(mixed_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert worksheet.row_count == 3
        assert worksheet.column_count == 6


class TestLogging:
    """Test logging functionality."""
    
    def test_successful_processing_logged(self, sample_excel_file, caplog):
        """Test that successful processing is logged."""
        processor = ExcelProcessor()
        
        with caplog.at_level(logging.INFO):
            processor.process_file(sample_excel_file)
        
        log_messages = [record.message for record in caplog.records]
        
        # Check for start and completion logs
        start_logged = any("Starting Excel file processing" in msg for msg in log_messages)
        complete_logged = any("Excel processing completed" in msg for msg in log_messages)
        
        assert start_logged
        assert complete_logged
    
    def test_validation_failure_logged(self, temp_workspace, caplog):
        """Test that validation failures are logged."""
        processor = ExcelProcessor()
        nonexistent_file = temp_workspace / "does_not_exist.xlsx"
        
        with caplog.at_level(logging.ERROR):
            try:
                processor._validate_file(nonexistent_file)
            except ExcelProcessingError:
                pass
        
        log_messages = [record.message for record in caplog.records]
        error_logged = any("File not found" in msg for msg in log_messages)
        assert error_logged


if __name__ == "__main__":
    pytest.main([__file__])