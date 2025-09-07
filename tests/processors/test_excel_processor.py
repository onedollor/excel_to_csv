"""Unit tests for Excel processor."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile

from excel_to_csv.processors.excel_processor import ExcelProcessor
from excel_to_csv.models.data_models import WorksheetData
from excel_to_csv.config.config_manager import ConfigurationError


class TestExcelProcessor:
    """Test cases for ExcelProcessor class."""
    
    def test_init(self):
        """Test ExcelProcessor initialization."""
        processor = ExcelProcessor()
        assert processor.max_file_size_mb == 100    
    def test_init_with_custom_size(self):
        """Test ExcelProcessor initialization with custom file size."""
        processor = ExcelProcessor(max_file_size_mb=50)
        assert processor.max_file_size_mb == 50
    
    def test_process_excel_file_success(self, sample_excel_file: Path, sample_excel_data: pd.DataFrame):
        """Test successful Excel file processing."""
        processor = ExcelProcessor()
        
        worksheets = processor.process_excel_file(sample_excel_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert isinstance(worksheet, WorksheetData)
        assert worksheet.name == "Sheet1"
        assert worksheet.data.shape == sample_excel_data.shape
        assert list(worksheet.data.columns) == list(sample_excel_data.columns)
    
    def test_process_excel_file_nonexistent(self, temp_dir: Path):
        """Test processing non-existent Excel file."""
        processor = ExcelProcessor()
        nonexistent_file = temp_dir / "nonexistent.xlsx"
        
        with pytest.raises(FileNotFoundError):
            processor.process_excel_file(nonexistent_file)
    
    def test_process_excel_file_invalid_format(self, invalid_excel_file: Path):
        """Test processing invalid Excel file format."""
        processor = ExcelProcessor()
        
        with pytest.raises(Exception):  # pandas/openpyxl will raise various exceptions
            processor.process_excel_file(invalid_excel_file)
    
    def test_process_excel_file_too_large(self, temp_dir: Path):
        """Test processing Excel file that exceeds size limit."""
        processor = ExcelProcessor(max_file_size_mb=0.001)  # Very small limit
        
        # Create a file that exceeds the limit
        large_file = temp_dir / "large.xlsx"
        large_file.write_bytes(b"x" * 2000)  # 2KB file, but limit is ~1KB
        
        with pytest.raises(ValueError, match="File size.*exceeds maximum"):
            processor.process_excel_file(large_file)
    
    def test_process_excel_file_empty_worksheets(self, temp_dir: Path):
        """Test processing Excel file with empty worksheets."""
        processor = ExcelProcessor()
        
        # Create Excel file with empty worksheet
        empty_df = pd.DataFrame()
        excel_file = temp_dir / "empty.xlsx"
        empty_df.to_excel(excel_file, index=False)
        
        worksheets = processor.process_excel_file(excel_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert worksheet.data.empty
        assert worksheet.name == "Sheet1"
    
    def test_process_excel_file_multiple_worksheets(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test processing Excel file with multiple worksheets."""
        processor = ExcelProcessor()
        
        # Create Excel file with multiple worksheets
        excel_file = temp_dir / "multi_sheet.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            sample_excel_data.to_excel(writer, sheet_name="Sheet1", index=False)
            sample_excel_data.to_excel(writer, sheet_name="Sheet2", index=False)
        
        worksheets = processor.process_excel_file(excel_file)
        
        assert len(worksheets) == 2
        sheet_names = [ws.name for ws in worksheets]
        assert "Sheet1" in sheet_names
        assert "Sheet2" in sheet_names
    
    def test_process_excel_file_with_metadata(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test processing Excel file and extracting metadata."""
        processor = ExcelProcessor()
        
        excel_file = temp_dir / "with_metadata.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)
        
        worksheets = processor.process_excel_file(excel_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert worksheet.file_path == excel_file
        assert isinstance(worksheet.row_count, int)
        assert isinstance(worksheet.column_count, int)
        assert worksheet.row_count > 0
        assert worksheet.column_count > 0
    
    def test_get_file_size_mb(self, sample_excel_file: Path):
        """Test file size calculation."""
        processor = ExcelProcessor()
        
        size_mb = processor._get_file_size_mb(sample_excel_file)
        
        assert isinstance(size_mb, (int, float))
        assert size_mb > 0
        assert size_mb < 1  # Sample file should be small
    
    def test_validate_file_size_success(self, sample_excel_file: Path):
        """Test successful file size validation."""
        processor = ExcelProcessor(max_file_size_mb=100)
        
        # Should not raise an exception
        processor._validate_file_size(sample_excel_file)
    
    def test_validate_file_size_failure(self, sample_excel_file: Path):
        """Test file size validation failure."""
        processor = ExcelProcessor(max_file_size_mb=0.001)  # Very small limit
        
        with pytest.raises(ValueError, match="File size.*exceeds maximum"):
            processor._validate_file_size(sample_excel_file)
    
    def test_read_excel_with_pandas(self, sample_excel_file: Path):
        """Test reading Excel file with pandas."""
        processor = ExcelProcessor()
        
        worksheets = processor._read_excel_with_pandas(sample_excel_file)
        
        assert len(worksheets) >= 1
        for name, df in worksheets.items():
            assert isinstance(name, str)
            assert isinstance(df, pd.DataFrame)
    
    def test_read_excel_with_openpyxl(self, sample_excel_file: Path):
        """Test reading Excel file with openpyxl for metadata."""
        processor = ExcelProcessor()
        
        metadata = processor._read_excel_metadata_with_openpyxl(sample_excel_file)
        
        assert isinstance(metadata, dict)
        assert len(metadata) >= 1
        for sheet_name, info in metadata.items():
            assert isinstance(sheet_name, str)
            assert 'max_row' in info
            assert 'max_column' in info
    
    def test_create_worksheet_data(self, sample_excel_file: Path, sample_excel_data: pd.DataFrame):
        """Test creating WorksheetData from pandas DataFrame."""
        processor = ExcelProcessor()
        
        metadata = {'max_row': 6, 'max_column': 5}  # Sample metadata
        worksheet_data = processor._create_worksheet_data(
            worksheet_name="TestSheet",
            data=sample_excel_data,
            metadata=metadata,
            source_file=sample_excel_file
        )
        
        assert isinstance(worksheet_data, WorksheetData)
        assert worksheet_data.name == "TestSheet"
        assert worksheet_data.file_path == sample_excel_file
        assert worksheet_data.data.equals(sample_excel_data)
        assert worksheet_data.row_count == 6
        assert worksheet_data.column_count == 5
    
    def test_process_xls_file(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test processing .xls file format."""
        processor = ExcelProcessor()
        
        # Create .xls file
        xls_file = temp_dir / "test.xls"
        sample_excel_data.to_excel(xls_file, index=False)
        
        worksheets = processor.process_excel_file(xls_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert isinstance(worksheet, WorksheetData)
        assert worksheet.file_path == xls_file
    
    @patch('excel_to_csv.processors.excel_processor.pd.read_excel')
    def test_pandas_read_error_handling(self, mock_read_excel, sample_excel_file: Path):
        """Test error handling when pandas fails to read Excel file."""
        processor = ExcelProcessor()
        
        # Mock pandas to raise an exception
        mock_read_excel.side_effect = Exception("Pandas read error")
        
        with pytest.raises(Exception, match="Pandas read error"):
            processor._read_excel_with_pandas(sample_excel_file)
    
    @patch('excel_to_csv.processors.excel_processor.openpyxl.load_workbook')
    def test_openpyxl_read_error_handling(self, mock_load_workbook, sample_excel_file: Path):
        """Test error handling when openpyxl fails to read Excel file."""
        processor = ExcelProcessor()
        
        # Mock openpyxl to raise an exception
        mock_load_workbook.side_effect = Exception("Openpyxl read error")
        
        with pytest.raises(Exception, match="Openpyxl read error"):
            processor._read_excel_metadata_with_openpyxl(sample_excel_file)
    
    def test_worksheet_data_with_special_characters(self, temp_dir: Path):
        """Test processing worksheets with special characters in names."""
        processor = ExcelProcessor()
        
        # Create Excel file with special character worksheet names
        data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        excel_file = temp_dir / "special_chars.xlsx"
        
        with pd.ExcelWriter(excel_file) as writer:
            data.to_excel(writer, sheet_name="Sheet with spaces", index=False)
            data.to_excel(writer, sheet_name="Sheet-with-dashes", index=False)
            data.to_excel(writer, sheet_name="Sheet_with_underscores", index=False)
        
        worksheets = processor.process_excel_file(excel_file)
        
        assert len(worksheets) == 3
        sheet_names = [ws.name for ws in worksheets]
        assert "Sheet with spaces" in sheet_names
        assert "Sheet-with-dashes" in sheet_names
        assert "Sheet_with_underscores" in sheet_names
    
    def test_process_sparse_data(self, temp_dir: Path, sample_sparse_data: pd.DataFrame):
        """Test processing Excel file with sparse data."""
        processor = ExcelProcessor()
        
        excel_file = temp_dir / "sparse.xlsx"
        sample_sparse_data.to_excel(excel_file, index=False)
        
        worksheets = processor.process_excel_file(excel_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert isinstance(worksheet, WorksheetData)
        assert worksheet.data.shape == sample_sparse_data.shape
    
    def test_process_file_with_formulas(self, temp_dir: Path):
        """Test processing Excel file with formulas."""
        processor = ExcelProcessor()
        
        # Create Excel file with formulas (pandas will read calculated values)
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [5, 7, 9]  # This would be =A+B in Excel
        })
        excel_file = temp_dir / "formulas.xlsx"
        data.to_excel(excel_file, index=False)
        
        worksheets = processor.process_excel_file(excel_file)
        
        assert len(worksheets) == 1
        worksheet = worksheets[0]
        assert 'A' in worksheet.data.columns
        assert 'B' in worksheet.data.columns
        assert 'C' in worksheet.data.columns
    
    def test_error_logging(self, temp_dir: Path, caplog):
        """Test that errors are properly logged."""
        processor = ExcelProcessor()
        
        # Try to process a non-existent file
        nonexistent_file = temp_dir / "missing.xlsx"
        
        with pytest.raises(FileNotFoundError):
            processor.process_excel_file(nonexistent_file)
        
        # Check that error was logged (caplog captures log messages)
        assert len(caplog.records) > 0
    
    def test_concurrent_processing_safety(self, sample_excel_file: Path):
        """Test that processor can handle concurrent access safely."""
        import threading
        
        processor = ExcelProcessor()
        results = []
        exceptions = []
        
        def process_file():
            try:
                worksheets = processor.process_excel_file(sample_excel_file)
                results.append(worksheets)
            except Exception as e:
                exceptions.append(e)
        
        # Create multiple threads to process the same file
        threads = [threading.Thread(target=process_file) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed
        assert len(exceptions) == 0
        assert len(results) == 3
        
        # All results should be identical
        for result in results:
            assert len(result) == len(results[0])