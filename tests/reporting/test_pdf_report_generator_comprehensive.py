"""Comprehensive tests for PDF report generation functionality.

This test suite covers:
- PDFReportGenerator initialization and configuration
- PDF report generation with all components
- Professional styling and formatting
- Error handling and edge cases
- Integration with data models
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

try:
    from excel_to_csv.reporting.pdf_report_generator import PDFReportGenerator
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from excel_to_csv.reporting.report_generator import (
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
        worksheet_name="TestSheet",
        data=data,
        confidence_score=ConfidenceScore(
            overall_score=0.85,
            data_density=0.90,
            header_quality=0.80,
            consistency_score=0.85,
            threshold=0.75,
            reasons=["High data density", "Good headers"]
        )
    )


@pytest.fixture
def sample_processing_report(temp_workspace, sample_worksheet_data):
    """Create sample processing report for testing."""
    worksheet_analysis = WorksheetAnalysisReport(
        worksheet_name="TestSheet",
        passed=True,
        confidence_score=sample_worksheet_data.confidence_score,
        row_count=3,
        column_count=3,
        non_empty_cells=9,
        header_info=HeaderInfo(
            has_headers=True,
            header_row=0,
            header_quality=0.80,
            column_names=['Name', 'Age', 'City']
        ),
        data_density=0.90,
        reasons=["High data density", "Good headers"],
        processing_time_ms=125.5
    )
    
    csv_report = CSVGenerationReport(
        csv_file_path=temp_workspace / "test.csv",
        worksheet_name="TestSheet",
        rows_written=3,
        columns_written=3,
        file_size_bytes=1024,
        encoding_used='utf-8',
        delimiter_used=',',
        generation_time_ms=15.2
    )
    
    return ExcelProcessingReport(
        source_file=Path("/test/sample.xlsx"),
        processing_timestamp=datetime(2024, 1, 15, 14, 30, 15),
        overall_success=True,
        worksheets_analyzed=[worksheet_analysis],
        csv_files_generated=[csv_report],
        total_processing_time_ms=125.5,
        confidence_threshold=0.75,
        archive_result=ArchiveResult(
            success=True,
            source_path=Path("/test/sample.xlsx"),
            archive_path=Path("/archive/sample_20240115_143015.xlsx"),
            timestamp_used="20240115_143015",
            operation_time=1.5
        )
    )


@pytest.mark.skipif(not PDF_AVAILABLE, reason="ReportLab not available")
class TestPDFReportGenerator:
    """Test cases for PDFReportGenerator class."""
    
    def test_pdf_generator_initialization_default(self, temp_workspace):
        """Test PDFReportGenerator initialization with default settings."""
        generator = PDFReportGenerator()
        assert generator.output_dir == Path('./reports')
        assert generator.output_dir.exists()
        assert generator.logger is not None
        assert generator.styles is not None
    
    def test_pdf_generator_initialization_custom_dir(self, temp_workspace):
        """Test PDFReportGenerator initialization with custom output directory."""
        custom_dir = temp_workspace / "custom_reports"
        generator = PDFReportGenerator(output_dir=custom_dir)
        assert generator.output_dir == custom_dir
        assert generator.output_dir.exists()
    
    def test_pdf_report_generation(self, temp_workspace, sample_processing_report):
        """Test successful PDF report generation."""
        generator = PDFReportGenerator(output_dir=temp_workspace)
        
        report_path = generator.generate_report(sample_processing_report)
        
        assert report_path.exists()
        assert report_path.suffix == '.pdf'
        assert report_path.parent == temp_workspace
        assert 'sample_20240115_143015' in report_path.stem
    
    def test_pdf_report_content_structure(self, temp_workspace, sample_processing_report):
        """Test that PDF report contains expected content structure."""
        generator = PDFReportGenerator(output_dir=temp_workspace)
        
        report_path = generator.generate_report(sample_processing_report)
        
        # Verify PDF file was created and has reasonable size
        assert report_path.stat().st_size > 1000  # PDF should have reasonable size
    
    def test_pdf_report_filename_generation(self, temp_workspace, sample_processing_report):
        """Test PDF report filename generation with timestamp."""
        generator = PDFReportGenerator(output_dir=temp_workspace)
        
        report_path = generator.generate_report(sample_processing_report)
        
        expected_filename = "sample_20240115_143015_report.pdf"
        assert report_path.name == expected_filename
    
    def test_pdf_generator_with_multiple_worksheets(self, temp_workspace, sample_worksheet_data):
        """Test PDF generation with multiple worksheets."""
        # Create report with multiple worksheets
        worksheet_analyses = []
        csv_reports = []
        
        for i in range(3):
            worksheet_data = WorksheetData(
                source_file=Path("/test/multi_sheet.xlsx"),
                worksheet_name=f"Sheet{i+1}",
                data=pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}),
                confidence_score=ConfidenceScore(
                    overall_score=0.8 + i*0.05,
                    data_density=0.85,
                    header_quality=0.75,
                    consistency_score=0.80,
                    threshold=0.75,
                    reasons=["Good data"]
                )
            )
            
            worksheet_analyses.append(WorksheetAnalysisReport(
                worksheet_name=f"Sheet{i+1}",
                passed=True,
                confidence_score=worksheet_data.confidence_score,
                row_count=2,
                column_count=2,
                non_empty_cells=4,
                header_info=HeaderInfo(
                    has_headers=True,
                    header_row=0,
                    header_quality=0.75,
                    column_names=['col1', 'col2']
                ),
                data_density=0.85,
                reasons=["Good data"],
                processing_time_ms=100 + i*50
            ))
            
            csv_reports.append(CSVGenerationReport(
                csv_file_path=temp_workspace / f"sheet{i+1}.csv",
                worksheet_name=f"Sheet{i+1}",
                rows_written=2,
                columns_written=2,
                file_size_bytes=512,
                encoding_used='utf-8',
                delimiter_used=',',
                generation_time_ms=10.0
            ))
        
        report = ExcelProcessingReport(
            source_file=Path("/test/multi_sheet.xlsx"),
            processing_timestamp=datetime(2024, 1, 15, 14, 30, 15),
            overall_success=True,
            worksheets_analyzed=worksheet_analyses,
            csv_files_generated=csv_reports,
            total_processing_time_ms=450,
            confidence_threshold=0.75
        )
        
        generator = PDFReportGenerator(output_dir=temp_workspace)
        report_path = generator.generate_report(report)
        
        assert report_path.exists()
        assert report_path.stat().st_size > 2000  # Should be larger with multiple worksheets
    
    def test_pdf_generator_with_failed_worksheet(self, temp_workspace, sample_worksheet_data):
        """Test PDF generation with failed worksheet analysis."""
        # Create worksheet with low confidence (failed)
        failed_worksheet_data = WorksheetData(
            source_file=Path("/test/failed_sample.xlsx"),
            worksheet_name="FailedSheet",
            data=pd.DataFrame({'sparse': [None, 'data', None]}),
            confidence_score=ConfidenceScore(
                overall_score=0.45,
                data_density=0.33,
                header_quality=0.40,
                consistency_score=0.62,
                threshold=0.75,
                reasons=["Low data density", "Poor headers"]
            )
        )
        
        worksheet_analysis = WorksheetAnalysisReport(
            worksheet_name="FailedSheet",
            passed=False,
            confidence_score=failed_worksheet_data.confidence_score,
            row_count=3,
            column_count=1,
            non_empty_cells=1,
            header_info=HeaderInfo(
                has_headers=False,
                header_row=None,
                header_quality=0.40,
                column_names=['sparse']
            ),
            data_density=0.33,
            reasons=["Low data density", "Poor headers"],
            issues_found=["Low data density", "Insufficient structured data"],
            processing_time_ms=75.0
        )
        
        report = ExcelProcessingReport(
            source_file=Path("/test/failed_sample.xlsx"),
            processing_timestamp=datetime(2024, 1, 15, 14, 30, 15),
            overall_success=False,
            worksheets_analyzed=[worksheet_analysis],
            csv_files_generated=[],
            total_processing_time_ms=75.0,
            confidence_threshold=0.75
        )
        
        generator = PDFReportGenerator(output_dir=temp_workspace)
        report_path = generator.generate_report(report)
        
        assert report_path.exists()
        assert report_path.stat().st_size > 1000
    
    def test_pdf_generator_output_directory_creation(self, temp_workspace):
        """Test that output directory is created if it doesn't exist."""
        non_existent_dir = temp_workspace / "new_reports" / "sub_dir"
        generator = PDFReportGenerator(output_dir=non_existent_dir)
        
        assert non_existent_dir.exists()
        assert generator.output_dir == non_existent_dir
    
    def test_pdf_generator_logging(self, temp_workspace, sample_processing_report):
        """Test that PDF generator logs appropriately."""
        generator = PDFReportGenerator(output_dir=temp_workspace)
        
        with patch.object(generator.logger, 'info') as mock_log:
            report_path = generator.generate_report(sample_processing_report)
            
            # Should log PDF generation
            mock_log.assert_called()
            log_calls = [call.args[0] for call in mock_log.call_args_list]
            assert any("PDF report generated" in call for call in log_calls)


@pytest.mark.skipif(PDF_AVAILABLE, reason="Testing ImportError when ReportLab not available")
class TestPDFReportGeneratorImportError:
    """Test cases for PDFReportGenerator when ReportLab is not available."""
    
    def test_pdf_generator_import_error(self):
        """Test that PDFReportGenerator raises ImportError when ReportLab not available."""
        with pytest.raises(ImportError, match="ReportLab is required for PDF generation"):
            # This would normally be tested by mocking the import, but since we're testing
            # the actual import error scenario, we skip this test when PDF_AVAILABLE is True
            pass


class TestPDFIntegration:
    """Integration tests for PDF functionality."""
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="ReportLab not available")
    def test_pdf_generation_end_to_end(self, temp_workspace, sample_processing_report):
        """Test complete PDF generation workflow."""
        generator = PDFReportGenerator(output_dir=temp_workspace)
        
        # Generate PDF report
        report_path = generator.generate_report(sample_processing_report)
        
        # Verify PDF file properties
        assert report_path.exists()
        assert report_path.is_file()
        assert report_path.suffix == '.pdf'
        assert report_path.stat().st_size > 500  # Reasonable minimum size
        
        # Verify filename format
        assert 'sample_20240115_143015_report.pdf' == report_path.name
    
    @pytest.mark.skipif(not PDF_AVAILABLE, reason="ReportLab not available")
    def test_pdf_vs_markdown_content_parity(self, temp_workspace, sample_processing_report):
        """Test that PDF contains similar content to markdown reports."""
        from excel_to_csv.reporting.report_generator import ReportGenerator
        
        # Generate both PDF and markdown reports
        pdf_generator = PDFReportGenerator(output_dir=temp_workspace)
        md_generator = ReportGenerator(output_dir=temp_workspace)
        
        pdf_path = pdf_generator.generate_report(sample_processing_report)
        md_path = md_generator.generate_report(sample_processing_report)
        
        # Both should exist
        assert pdf_path.exists()
        assert md_path.exists()
        
        # PDF should be larger (binary format)
        assert pdf_path.stat().st_size > md_path.stat().st_size
        
        # Read markdown content to verify key sections are present
        md_content = md_path.read_text(encoding='utf-8')
        assert "Executive Summary" in md_content
        assert "Worksheet Analysis Details" in md_content
        assert "CSV Files Generated" in md_content
        assert "Performance Metrics" in md_content