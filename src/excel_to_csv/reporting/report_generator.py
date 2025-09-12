"""Report generation for Excel processing results.

This module generates detailed markdown reports after processing each Excel file,
including worksheet analysis, pass/fail status, and CSV generation details.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

import pandas as pd

from excel_to_csv.models.data_models import (
    WorksheetData, 
    ConfidenceScore, 
    HeaderInfo,
    ArchiveResult
)
from excel_to_csv.utils.logger import get_processing_logger


@dataclass
class WorksheetAnalysisReport:
    """Detailed analysis report for a single worksheet."""
    worksheet_name: str
    passed: bool
    confidence_score: Optional[ConfidenceScore] = None
    row_count: int = 0
    column_count: int = 0
    non_empty_cells: int = 0
    header_info: Optional[HeaderInfo] = None
    data_density: float = 0.0
    reasons: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    generated_csv: Optional[Path] = None
    csv_size_bytes: int = 0
    processing_time_ms: float = 0.0


@dataclass 
class CSVGenerationReport:
    """Report for CSV file generation details."""
    csv_file_path: Path
    worksheet_name: str
    rows_written: int
    columns_written: int
    file_size_bytes: int
    encoding_used: str = "utf-8"
    delimiter_used: str = ","
    generation_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ExcelProcessingReport:
    """Comprehensive report for Excel file processing."""
    source_file: Path
    processing_timestamp: datetime
    overall_success: bool
    worksheets_analyzed: List[WorksheetAnalysisReport] = field(default_factory=list)
    csv_files_generated: List[CSVGenerationReport] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    confidence_threshold: float = 0.0
    archive_result: Optional[ArchiveResult] = None
    error_summary: List[str] = field(default_factory=list)
    
    @property
    def worksheets_passed(self) -> int:
        """Count of worksheets that passed analysis."""
        return sum(1 for ws in self.worksheets_analyzed if ws.passed)
    
    @property
    def worksheets_failed(self) -> int:
        """Count of worksheets that failed analysis."""
        return sum(1 for ws in self.worksheets_analyzed if not ws.passed)
    
    @property
    def total_worksheets(self) -> int:
        """Total number of worksheets analyzed."""
        return len(self.worksheets_analyzed)
    
    @property
    def total_csv_files(self) -> int:
        """Total number of CSV files generated."""
        return len(self.csv_files_generated)
    
    @property
    def total_rows_processed(self) -> int:
        """Total rows across all generated CSV files."""
        return sum(csv.rows_written for csv in self.csv_files_generated)


class ReportGenerator:
    """Generates detailed markdown reports for Excel processing results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory to write report files. Defaults to './reports'
        """
        self.output_dir = Path(output_dir or './reports')
        self.logger = get_processing_logger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, report: ExcelProcessingReport) -> Path:
        """Generate a comprehensive markdown report for Excel processing.
        
        Args:
            report: Excel processing report data
            
        Returns:
            Path to the generated report file
        """
        self.logger.log_processing_start(
            f"Generating report for {report.source_file.name}",
            len(str(report.source_file))
        )
        
        # Generate report filename
        timestamp = report.processing_timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{report.source_file.stem}_{timestamp}_report.md"
        report_path = self.output_dir / filename
        
        # Generate markdown content
        content = self._generate_markdown_content(report)
        
        # Write report file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.logger.log_processing_complete(
            f"Report generated: {report_path}",
            1,
            0.1,
            1
        )
        
        return report_path
    
    def _generate_markdown_content(self, report: ExcelProcessingReport) -> str:
        """Generate the markdown content for the report."""
        content = []
        
        # Title and header
        content.append(f"# Excel Processing Report")
        content.append(f"**File**: `{report.source_file.name}`")
        content.append(f"**Processed**: {report.processing_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Overall Status**: {'✅ SUCCESS' if report.overall_success else '❌ FAILED'}")
        content.append("")
        
        # Executive summary
        content.extend(self._generate_executive_summary(report))
        
        # Worksheet analysis details
        content.extend(self._generate_worksheet_analysis(report))
        
        # CSV generation details
        content.extend(self._generate_csv_details(report))
        
        # Performance metrics
        content.extend(self._generate_performance_metrics(report))
        
        # Archive information
        if report.archive_result:
            content.extend(self._generate_archive_info(report.archive_result))
        
        # Error summary
        if report.error_summary:
            content.extend(self._generate_error_summary(report.error_summary))
        
        # Technical details
        content.extend(self._generate_technical_details(report))
        
        return '\n'.join(content)
    
    def _generate_executive_summary(self, report: ExcelProcessingReport) -> List[str]:
        """Generate executive summary section."""
        content = [
            "## 📊 Executive Summary",
            "",
            f"- **Total Worksheets**: {report.total_worksheets}",
            f"- **Worksheets Passed**: {report.worksheets_passed} ✅",
            f"- **Worksheets Failed**: {report.worksheets_failed} ❌",
            f"- **CSV Files Generated**: {report.total_csv_files}",
            f"- **Total Rows Processed**: {report.total_rows_processed:,}",
            f"- **Processing Time**: {report.total_processing_time_ms:.2f}ms",
            f"- **Confidence Threshold**: {report.confidence_threshold:.2%}",
            ""
        ]
        return content
    
    def _generate_worksheet_analysis(self, report: ExcelProcessingReport) -> List[str]:
        """Generate detailed worksheet analysis section."""
        content = [
            "## 📋 Worksheet Analysis Details",
            ""
        ]
        
        for i, worksheet in enumerate(report.worksheets_analyzed, 1):
            status_emoji = "✅" if worksheet.passed else "❌"
            status_text = "PASSED" if worksheet.passed else "FAILED"
            
            content.extend([
                f"### {i}. Worksheet: `{worksheet.worksheet_name}` {status_emoji}",
                f"**Status**: {status_text}",
                "",
                "**📈 Data Metrics**:",
                f"- Rows: {worksheet.row_count:,}",
                f"- Columns: {worksheet.column_count:,}",
                f"- Non-empty cells: {worksheet.non_empty_cells:,}",
                f"- Data density: {worksheet.data_density:.2%}",
                f"- Processing time: {worksheet.processing_time_ms:.2f}ms",
                ""
            ])
            
            # Confidence score details
            if worksheet.confidence_score:
                score = worksheet.confidence_score
                content.extend([
                    "**🎯 Confidence Analysis**:",
                    f"- Overall score: {score.overall_score:.3f}",
                    f"- Data density score: {score.data_density:.3f}",
                    f"- Header quality score: {score.header_quality:.3f}",
                    f"- Consistency score: {score.consistency_score:.3f}",
                    f"- Threshold: {score.threshold:.3f}",
                    f"- Is confident: {'Yes' if score.is_confident else 'No'}",
                    ""
                ])
                
                if score.reasons:
                    content.extend([
                        "**📝 Analysis Reasons**:",
                        *[f"- {reason}" for reason in score.reasons],
                        ""
                    ])
            
            # Header information
            if worksheet.header_info:
                header = worksheet.header_info
                content.extend([
                    "**📑 Header Information**:",
                    f"- Has headers: {'Yes' if header.has_headers else 'No'}",
                    f"- Header row: {header.header_row if header.header_row is not None else 'N/A'}",
                    f"- Header quality: {header.header_quality:.3f}",
                    f"- Column names: {', '.join(header.column_names[:5])}{'...' if len(header.column_names) > 5 else ''}",
                    ""
                ])
            
            # Issues found
            if worksheet.issues_found:
                content.extend([
                    "**⚠️ Issues Found**:",
                    *[f"- {issue}" for issue in worksheet.issues_found],
                    ""
                ])
            
            # CSV generation info
            if worksheet.generated_csv:
                content.extend([
                    "**📄 Generated CSV**:",
                    f"- File: `{worksheet.generated_csv.name}`",
                    f"- Size: {self._format_file_size(worksheet.csv_size_bytes)}",
                    ""
                ])
            
            content.append("---")
            content.append("")
        
        return content
    
    def _generate_csv_details(self, report: ExcelProcessingReport) -> List[str]:
        """Generate CSV generation details section."""
        if not report.csv_files_generated:
            return ["## 📄 CSV Files Generated", "", "*No CSV files were generated.*", ""]
        
        content = [
            "## 📄 CSV Files Generated",
            "",
            f"Total CSV files: **{len(report.csv_files_generated)}**",
            ""
        ]
        
        for i, csv in enumerate(report.csv_files_generated, 1):
            status_emoji = "✅" if csv.success else "❌"
            
            content.extend([
                f"### {i}. `{csv.csv_file_path.name}` {status_emoji}",
                f"- **Source worksheet**: {csv.worksheet_name}",
                f"- **Rows written**: {csv.rows_written:,}",
                f"- **Columns written**: {csv.columns_written:,}",
                f"- **File size**: {self._format_file_size(csv.file_size_bytes)}",
                f"- **Encoding**: {csv.encoding_used}",
                f"- **Delimiter**: `'{csv.delimiter_used}'`",
                f"- **Generation time**: {csv.generation_time_ms:.2f}ms",
                ""
            ])
            
            if not csv.success and csv.error_message:
                content.extend([
                    f"- **Error**: {csv.error_message}",
                    ""
                ])
        
        return content
    
    def _generate_performance_metrics(self, report: ExcelProcessingReport) -> List[str]:
        """Generate performance metrics section."""
        content = [
            "## ⚡ Performance Metrics",
            "",
            f"- **Total processing time**: {report.total_processing_time_ms:.2f}ms",
            f"- **Average time per worksheet**: {report.total_processing_time_ms / max(report.total_worksheets, 1):.2f}ms",
        ]
        
        if report.total_rows_processed > 0:
            rows_per_second = report.total_rows_processed / (report.total_processing_time_ms / 1000)
            content.append(f"- **Processing speed**: {rows_per_second:,.0f} rows/second")
        
        content.extend([
            f"- **Memory efficiency**: {report.total_rows_processed} rows processed",
            ""
        ])
        
        return content
    
    def _generate_archive_info(self, archive_result: ArchiveResult) -> List[str]:
        """Generate archive information section."""
        content = [
            "## 📦 Archive Information",
            "",
            f"- **Archive status**: {'✅ SUCCESS' if archive_result.success else '❌ FAILED'}",
            f"- **Source path**: `{archive_result.source_path}`",
        ]
        
        if archive_result.archive_path:
            content.append(f"- **Archive path**: `{archive_result.archive_path}`")
        
        if archive_result.timestamp_used:
            content.append(f"- **Timestamp used**: {archive_result.timestamp_used}")
        
        content.append(f"- **Operation time**: {archive_result.operation_time:.3f}s")
        
        if archive_result.error_message:
            content.extend([
                f"- **Error**: {archive_result.error_message}",
                ""
            ])
        else:
            content.append("")
        
        return content
    
    def _generate_error_summary(self, errors: List[str]) -> List[str]:
        """Generate error summary section."""
        content = [
            "## ⚠️ Error Summary",
            "",
            f"Total errors encountered: **{len(errors)}**",
            ""
        ]
        
        for i, error in enumerate(errors, 1):
            content.append(f"{i}. {error}")
        
        content.append("")
        return content
    
    def _generate_technical_details(self, report: ExcelProcessingReport) -> List[str]:
        """Generate technical details section."""
        content = [
            "## 🔧 Technical Details",
            "",
            f"- **Source file**: `{report.source_file}`",
            f"- **File size**: {self._format_file_size(report.source_file.stat().st_size) if report.source_file.exists() else 'Unknown'}",
            f"- **Processing timestamp**: {report.processing_timestamp.isoformat()}",
            f"- **Report generated**: {datetime.now().isoformat()}",
            ""
        ]
        
        # Add worksheet summary table
        if report.worksheets_analyzed:
            content.extend([
                "### Worksheet Summary Table",
                "",
                "| Worksheet | Status | Rows | Columns | Confidence | CSV Generated |",
                "|-----------|--------|------|---------|------------|---------------|"
            ])
            
            for ws in report.worksheets_analyzed:
                status = "✅ Pass" if ws.passed else "❌ Fail"
                confidence = f"{ws.confidence_score.overall_score:.3f}" if ws.confidence_score else "N/A"
                csv_gen = "Yes" if ws.generated_csv else "No"
                
                content.append(
                    f"| {ws.worksheet_name} | {status} | {ws.row_count:,} | "
                    f"{ws.column_count} | {confidence} | {csv_gen} |"
                )
            
            content.append("")
        
        return content
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def create_worksheet_report(
        self,
        worksheet_data: WorksheetData,
        confidence_score: Optional[ConfidenceScore] = None,
        header_info: Optional[HeaderInfo] = None,
        csv_path: Optional[Path] = None,
        processing_time_ms: float = 0.0,
        issues: Optional[List[str]] = None
    ) -> WorksheetAnalysisReport:
        """Create a worksheet analysis report from processing results.
        
        Args:
            worksheet_data: The processed worksheet data
            confidence_score: Confidence analysis results
            header_info: Header analysis results
            csv_path: Path to generated CSV file
            processing_time_ms: Processing time in milliseconds
            issues: List of issues found during processing
            
        Returns:
            Worksheet analysis report
        """
        # Determine pass/fail status
        passed = True
        reasons = []
        
        if confidence_score:
            passed = confidence_score.is_confident
            reasons.extend(confidence_score.reasons)
        
        # Calculate data density
        total_cells = worksheet_data.row_count * worksheet_data.column_count
        data_density = worksheet_data.non_empty_cell_count / max(total_cells, 1)
        
        # Get CSV file size if it exists
        csv_size = 0
        if csv_path and csv_path.exists():
            csv_size = csv_path.stat().st_size
        
        return WorksheetAnalysisReport(
            worksheet_name=worksheet_data.worksheet_name,
            passed=passed,
            confidence_score=confidence_score,
            row_count=worksheet_data.row_count,
            column_count=worksheet_data.column_count,
            non_empty_cells=worksheet_data.non_empty_cell_count,
            header_info=header_info,
            data_density=data_density,
            reasons=reasons,
            issues_found=issues or [],
            generated_csv=csv_path,
            csv_size_bytes=csv_size,
            processing_time_ms=processing_time_ms
        )
    
    def create_csv_report(
        self,
        csv_path: Path,
        worksheet_name: str,
        rows_written: int,
        columns_written: int,
        encoding: str = "utf-8",
        delimiter: str = ",",
        generation_time_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> CSVGenerationReport:
        """Create a CSV generation report.
        
        Args:
            csv_path: Path to the CSV file
            worksheet_name: Name of source worksheet
            rows_written: Number of rows written
            columns_written: Number of columns written
            encoding: Character encoding used
            delimiter: CSV delimiter used
            generation_time_ms: Generation time in milliseconds
            success: Whether generation was successful
            error_message: Error message if generation failed
            
        Returns:
            CSV generation report
        """
        file_size = 0
        if csv_path.exists():
            file_size = csv_path.stat().st_size
        
        return CSVGenerationReport(
            csv_file_path=csv_path,
            worksheet_name=worksheet_name,
            rows_written=rows_written,
            columns_written=columns_written,
            file_size_bytes=file_size,
            encoding_used=encoding,
            delimiter_used=delimiter,
            generation_time_ms=generation_time_ms,
            success=success,
            error_message=error_message
        )