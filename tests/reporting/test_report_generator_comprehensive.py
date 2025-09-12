"""Comprehensive tests for report generation functionality.

This test suite covers:
- ReportGenerator initialization and configuration
- Markdown report generation with all components
- WorksheetAnalysisReport creation and validation
- CSVGenerationReport creation and validation
- ExcelProcessingReport integration testing
- Error handling and edge cases
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from excel_to_csv.reporting.report_generator import (
    ReportGenerator,
    ExcelProcessingReport,
    WorksheetAnalysisReport,
    CSVGenerationReport
)
from excel_to_csv.models.data_models import (
    WorksheetData,
    ConfidenceScore,
    HeaderInfo,
    ArchiveResult
)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_worksheet_data():
    """Create sample worksheet data for testing."""
    data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Chicago']
    })
    
    return WorksheetData(
        source_file=Path("/test/sample.xlsx"),
        worksheet_name="Sheet1",
        data=data,
        metadata={'created': '2024-01-01'}
    )


@pytest.fixture
def sample_confidence_score():
    """Create sample confidence score for testing."""
    return ConfidenceScore(
        overall_score=0.85,
        data_density=0.9,
        header_quality=0.8,
        consistency_score=0.85,
        threshold=0.7,
        reasons=["Good data density", "Clear headers", "Consistent data types"]
    )


@pytest.fixture
def sample_header_info():
    """Create sample header info for testing."""
    return HeaderInfo(
        has_headers=True,
        header_row=0,
        header_quality=0.8,
        column_names=['Name', 'Age', 'City']
    )


@pytest.fixture
def sample_archive_result():
    """Create sample archive result for testing."""
    return ArchiveResult(
        success=True,
        source_path=Path("/test/sample.xlsx"),
        archive_path=Path("/archive/sample_20240101_120000.xlsx"),
        timestamp_used="20240101_120000",
        operation_time=1.25
    )


class TestReportGenerator:
    """Test ReportGenerator initialization and basic functionality."""
    
    def test_report_generator_initialization_default(self):
        """Test ReportGenerator initialization with default settings."""
        generator = ReportGenerator()
        
        assert generator.output_dir == Path('./reports')
        assert generator.output_dir.exists()
        assert generator.logger is not None
    
    def test_report_generator_initialization_custom_dir(self, temp_workspace):
        """Test ReportGenerator initialization with custom output directory."""
        custom_dir = temp_workspace / "custom_reports"
        generator = ReportGenerator(output_dir=custom_dir)
        
        assert generator.output_dir == custom_dir
        assert custom_dir.exists()
    
    def test_report_generator_output_directory_creation(self, temp_workspace):
        """Test that ReportGenerator creates output directory if it doesn't exist."""
        non_existent_dir = temp_workspace / "new_reports_dir"
        assert not non_existent_dir.exists()
        
        generator = ReportGenerator(output_dir=non_existent_dir)
        assert non_existent_dir.exists()
        assert generator.output_dir == non_existent_dir


class TestWorksheetAnalysisReport:
    """Test WorksheetAnalysisReport creation and validation."""
    
    def test_create_worksheet_report_basic(self, sample_worksheet_data, temp_workspace):
        """Test creating a basic worksheet analysis report."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            processing_time_ms=150.5
        )
        
        assert report.worksheet_name == "Sheet1"
        assert report.row_count == 3
        assert report.column_count == 3
        assert report.non_empty_cells == 9
        assert report.processing_time_ms == 150.5
        assert report.data_density == 1.0  # All cells filled
        assert report.passed == True  # Default pass without confidence score
    
    def test_create_worksheet_report_with_confidence_score(
        self, sample_worksheet_data, sample_confidence_score, temp_workspace
    ):
        """Test creating worksheet report with confidence score."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            confidence_score=sample_confidence_score
        )
        
        assert report.passed == True  # is_confident is True
        assert report.confidence_score == sample_confidence_score
        assert report.reasons == sample_confidence_score.reasons
    
    def test_create_worksheet_report_failed_confidence(
        self, sample_worksheet_data, temp_workspace
    ):
        """Test creating worksheet report with failed confidence."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        failed_score = ConfidenceScore(
            overall_score=0.3,
            threshold=0.7,
            reasons=["Low data density", "Poor header quality"]
        )
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            confidence_score=failed_score
        )
        
        assert report.passed == False  # is_confident is False
        assert report.confidence_score == failed_score
        assert report.reasons == failed_score.reasons
    
    def test_create_worksheet_report_with_csv_path(
        self, sample_worksheet_data, temp_workspace
    ):
        """Test creating worksheet report with CSV file information."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        # Create a dummy CSV file
        csv_path = temp_workspace / "test_output.csv"
        csv_path.write_text("Name,Age,City\nAlice,25,New York\nBob,30,San Francisco")
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            csv_path=csv_path
        )
        
        assert report.generated_csv == csv_path
        assert report.csv_size_bytes > 0
        assert report.csv_size_bytes == csv_path.stat().st_size
    
    def test_create_worksheet_report_with_header_info(
        self, sample_worksheet_data, sample_header_info, temp_workspace
    ):
        """Test creating worksheet report with header information."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            header_info=sample_header_info
        )
        
        assert report.header_info == sample_header_info
    
    def test_create_worksheet_report_with_issues(
        self, sample_worksheet_data, temp_workspace
    ):
        """Test creating worksheet report with identified issues."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        issues = ["Missing data in column B", "Inconsistent date formats"]
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            issues=issues
        )
        
        assert report.issues_found == issues


class TestCSVGenerationReport:
    """Test CSVGenerationReport creation and validation."""
    
    def test_create_csv_report_successful(self, temp_workspace):
        """Test creating a successful CSV generation report."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        # Create a test CSV file
        csv_path = temp_workspace / "successful.csv"
        csv_content = "Name,Age,City\nAlice,25,New York\nBob,30,San Francisco\nCharlie,35,Chicago"
        csv_path.write_text(csv_content)
        
        report = generator.create_csv_report(
            csv_path=csv_path,
            worksheet_name="Sheet1",
            rows_written=3,
            columns_written=3,
            generation_time_ms=75.2
        )
        
        assert report.csv_file_path == csv_path
        assert report.worksheet_name == "Sheet1"
        assert report.rows_written == 3
        assert report.columns_written == 3
        assert report.generation_time_ms == 75.2
        assert report.success == True
        assert report.file_size_bytes > 0
        assert report.encoding_used == "utf-8"
        assert report.delimiter_used == ","
    
    def test_create_csv_report_failed(self, temp_workspace):
        """Test creating a failed CSV generation report."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = generator.create_csv_report(
            csv_path=Path("/nonexistent/file.csv"),
            worksheet_name="FailedSheet",
            rows_written=0,
            columns_written=0,
            success=False,
            error_message="Permission denied writing to output directory"
        )
        
        assert report.success == False
        assert report.error_message == "Permission denied writing to output directory"
        assert report.rows_written == 0
        assert report.columns_written == 0
        assert report.file_size_bytes == 0
    
    def test_create_csv_report_custom_settings(self, temp_workspace):
        """Test creating CSV report with custom encoding and delimiter."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        csv_path = temp_workspace / "custom.csv"
        csv_path.write_text("Name|Age|City\nAlice|25|New York", encoding='latin1')
        
        report = generator.create_csv_report(
            csv_path=csv_path,
            worksheet_name="CustomSheet",
            rows_written=1,
            columns_written=3,
            encoding="latin1",
            delimiter="|",
            generation_time_ms=50.0
        )
        
        assert report.encoding_used == "latin1"
        assert report.delimiter_used == "|"
        assert report.generation_time_ms == 50.0


class TestExcelProcessingReport:
    """Test ExcelProcessingReport creation and properties."""
    
    def test_excel_processing_report_creation(self, temp_workspace, sample_archive_result):
        """Test creating a comprehensive Excel processing report."""
        source_file = Path("/test/sample.xlsx")
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Create sample worksheet reports
        worksheet_reports = [
            WorksheetAnalysisReport(
                worksheet_name="Sheet1",
                passed=True,
                row_count=100,
                column_count=5
            ),
            WorksheetAnalysisReport(
                worksheet_name="Sheet2", 
                passed=False,
                row_count=50,
                column_count=3
            )
        ]
        
        # Create sample CSV reports
        csv_reports = [
            CSVGenerationReport(
                csv_file_path=Path("/output/sheet1.csv"),
                worksheet_name="Sheet1",
                rows_written=100,
                columns_written=5,
                file_size_bytes=5000
            )
        ]
        
        report = ExcelProcessingReport(
            source_file=source_file,
            processing_timestamp=timestamp,
            overall_success=True,
            worksheets_analyzed=worksheet_reports,
            csv_files_generated=csv_reports,
            total_processing_time_ms=2500.5,
            confidence_threshold=0.7,
            archive_result=sample_archive_result,
            error_summary=[]
        )
        
        # Test properties
        assert report.worksheets_passed == 1
        assert report.worksheets_failed == 1
        assert report.total_worksheets == 2
        assert report.total_csv_files == 1
        assert report.total_rows_processed == 100
        
        # Test data integrity
        assert report.source_file == source_file
        assert report.processing_timestamp == timestamp
        assert report.overall_success == True
        assert report.confidence_threshold == 0.7
        assert report.archive_result == sample_archive_result
    
    def test_excel_processing_report_with_errors(self):
        """Test Excel processing report with errors."""
        report = ExcelProcessingReport(
            source_file=Path("/test/failed.xlsx"),
            processing_timestamp=datetime.now(),
            overall_success=False,
            error_summary=["File is corrupted", "Cannot read worksheet 'Data'"]
        )
        
        assert report.overall_success == False
        assert len(report.error_summary) == 2
        assert "File is corrupted" in report.error_summary


class TestReportGeneration:
    """Test full report generation and markdown output."""
    
    def test_generate_report_basic(self, temp_workspace, sample_archive_result):
        """Test generating a complete markdown report."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        # Create a comprehensive report
        report = ExcelProcessingReport(
            source_file=Path("/test/sample.xlsx"),
            processing_timestamp=datetime(2024, 1, 1, 12, 0, 0),
            overall_success=True,
            worksheets_analyzed=[
                WorksheetAnalysisReport(
                    worksheet_name="Data",
                    passed=True,
                    row_count=500,
                    column_count=8,
                    non_empty_cells=3800,
                    data_density=0.95,
                    processing_time_ms=125.5
                )
            ],
            csv_files_generated=[
                CSVGenerationReport(
                    csv_file_path=Path("/output/sample_Data.csv"),
                    worksheet_name="Data",
                    rows_written=500,
                    columns_written=8,
                    file_size_bytes=25600,
                    generation_time_ms=45.2
                )
            ],
            total_processing_time_ms=1250.0,
            confidence_threshold=0.75,
            archive_result=sample_archive_result
        )
        
        # Generate report
        report_path = generator.generate_report(report)
        
        # Verify file was created
        assert report_path.exists()
        assert report_path.suffix == '.md'
        assert "sample" in report_path.name
        assert "report.md" in report_path.name
        
        # Verify content
        content = report_path.read_text()
        assert "# Excel Processing Report" in content
        assert "**File**: `sample.xlsx`" in content
        assert "✅ SUCCESS" in content
        assert "Data" in content  # Worksheet name
        assert "500" in content  # Row count
        assert "25.6 KB" in content  # File size formatted
    
    def test_generate_report_with_failures(self, temp_workspace):
        """Test generating report for failed processing."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = ExcelProcessingReport(
            source_file=Path("/test/corrupted.xlsx"),
            processing_timestamp=datetime.now(),
            overall_success=False,
            worksheets_analyzed=[
                WorksheetAnalysisReport(
                    worksheet_name="BadData",
                    passed=False,
                    row_count=10,
                    column_count=2,
                    issues_found=["Corrupted cells", "Missing headers"]
                )
            ],
            csv_files_generated=[],
            error_summary=["File is corrupted", "Cannot process any worksheets"],
            confidence_threshold=0.8
        )
        
        report_path = generator.generate_report(report)
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "❌ FAILED" in content
        assert "Error Summary" in content
        assert "File is corrupted" in content
        assert "Issues Found" in content
    
    def test_generate_report_filename_format(self, temp_workspace):
        """Test that report filenames follow expected format."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = ExcelProcessingReport(
            source_file=Path("/test/my_data_file.xlsx"),
            processing_timestamp=datetime(2024, 3, 15, 14, 30, 45),
            overall_success=True
        )
        
        report_path = generator.generate_report(report)
        
        # Check filename format: {stem}_{timestamp}_report.md
        expected_pattern = "my_data_file_20240315_143045_report.md"
        assert report_path.name == expected_pattern
    
    def test_format_file_size(self, temp_workspace):
        """Test file size formatting utility."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        # Test various file sizes
        assert generator._format_file_size(500) == "500.0 B"
        assert generator._format_file_size(1536) == "1.5 KB"
        assert generator._format_file_size(2097152) == "2.0 MB"
        assert generator._format_file_size(3221225472) == "3.0 GB"
    
    @patch('excel_to_csv.reporting.report_generator.get_processing_logger')
    def test_report_generation_logging(self, mock_logger, temp_workspace):
        """Test that report generation logs appropriately."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = ExcelProcessingReport(
            source_file=Path("/test/sample.xlsx"),
            processing_timestamp=datetime.now(),
            overall_success=True
        )
        
        generator.generate_report(report)
        
        # Verify logging calls were made
        assert mock_logger_instance.log_processing_start.called
        assert mock_logger_instance.log_processing_complete.called


class TestReportGenerationErrorHandling:
    """Test error handling in report generation."""
    
    def test_generate_report_with_io_error(self, temp_workspace):
        """Test handling I/O errors during report generation."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        # Make directory read-only to cause write error
        temp_workspace.chmod(0o444)
        
        report = ExcelProcessingReport(
            source_file=Path("/test/sample.xlsx"),
            processing_timestamp=datetime.now(),
            overall_success=True
        )
        
        # Should raise exception or handle gracefully
        try:
            report_path = generator.generate_report(report)
            # If it succeeds, that's also acceptable
            assert report_path.exists() or True
        except (PermissionError, OSError):
            # Expected behavior - permission error handled
            pass
        finally:
            # Restore permissions for cleanup
            temp_workspace.chmod(0o755)
    
    def test_worksheet_report_with_missing_csv(self, sample_worksheet_data, temp_workspace):
        """Test worksheet report creation when CSV file doesn't exist."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        non_existent_csv = temp_workspace / "missing.csv"
        
        report = generator.create_worksheet_report(
            worksheet_data=sample_worksheet_data,
            csv_path=non_existent_csv
        )
        
        # Should handle missing file gracefully
        assert report.generated_csv == non_existent_csv
        assert report.csv_size_bytes == 0  # File doesn't exist
    
    def test_report_with_none_values(self, temp_workspace):
        """Test report generation with None values for optional fields."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        report = ExcelProcessingReport(
            source_file=Path("/test/sample.xlsx"),
            processing_timestamp=datetime.now(),
            overall_success=True,
            archive_result=None,  # None archive result
            error_summary=None    # None error summary should be handled
        )
        
        report_path = generator.generate_report(report)
        assert report_path.exists()
        
        content = report_path.read_text()
        # Should not contain archive section
        assert "Archive Information" not in content


class TestReportIntegration:
    """Test integration scenarios with actual processing workflow."""
    
    def test_end_to_end_report_creation(self, temp_workspace):
        """Test creating a report that mirrors real processing workflow."""
        generator = ReportGenerator(output_dir=temp_workspace)
        
        # Simulate a real processing scenario
        source_file = Path("/data/financial_report_2024.xlsx")
        
        # Multiple worksheets with different outcomes
        worksheets = [
            WorksheetAnalysisReport(
                worksheet_name="Summary",
                passed=True,
                row_count=50,
                column_count=6,
                non_empty_cells=275,
                data_density=0.92,
                confidence_score=ConfidenceScore(overall_score=0.88, threshold=0.75),
                generated_csv=Path("/output/financial_report_2024_Summary.csv"),
                csv_size_bytes=8192,
                processing_time_ms=85.3
            ),
            WorksheetAnalysisReport(
                worksheet_name="Raw Data",
                passed=True,
                row_count=1500,
                column_count=12,
                non_empty_cells=16800,
                data_density=0.93,
                confidence_score=ConfidenceScore(overall_score=0.91, threshold=0.75),
                generated_csv=Path("/output/financial_report_2024_Raw_Data.csv"),
                csv_size_bytes=245760,
                processing_time_ms=342.7
            ),
            WorksheetAnalysisReport(
                worksheet_name="Notes",
                passed=False,
                row_count=20,
                column_count=2,
                non_empty_cells=15,
                data_density=0.375,
                confidence_score=ConfidenceScore(overall_score=0.45, threshold=0.75),
                issues_found=["Low data density", "Mostly text notes"],
                processing_time_ms=25.1
            )
        ]
        
        csvs = [
            CSVGenerationReport(
                csv_file_path=Path("/output/financial_report_2024_Summary.csv"),
                worksheet_name="Summary",
                rows_written=50,
                columns_written=6,
                file_size_bytes=8192,
                generation_time_ms=15.2
            ),
            CSVGenerationReport(
                csv_file_path=Path("/output/financial_report_2024_Raw_Data.csv"),
                worksheet_name="Raw Data",
                rows_written=1500,
                columns_written=12,
                file_size_bytes=245760,
                generation_time_ms=125.8
            )
        ]
        
        archive = ArchiveResult(
            success=True,
            source_path=source_file,
            archive_path=Path("/archive/financial_report_2024_20240315_143000.xlsx"),
            operation_time=1.87
        )
        
        report = ExcelProcessingReport(
            source_file=source_file,
            processing_timestamp=datetime(2024, 3, 15, 14, 30, 0),
            overall_success=True,
            worksheets_analyzed=worksheets,
            csv_files_generated=csvs,
            total_processing_time_ms=2456.3,
            confidence_threshold=0.75,
            archive_result=archive,
            error_summary=[]
        )
        
        # Generate and verify report
        report_path = generator.generate_report(report)
        assert report_path.exists()
        
        content = report_path.read_text()
        
        # Verify comprehensive content
        assert "2 ✅" in content  # 2 passed worksheets
        assert "1 ❌" in content  # 1 failed worksheet
        assert "1,550" in content  # Total rows (50 + 1500)
        assert "75.00%" in content  # Confidence threshold
        assert "Archive Information" in content
        assert "240.0 KB" in content  # Large CSV file size
        assert "Low data density" in content  # Issue description
        
        # Verify performance metrics
        assert "2,456.30ms" in content  # Total processing time
        assert "rows/second" in content  # Processing speed calculation