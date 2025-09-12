"""PDF report generation for Excel processing results.

This module generates professional PDF reports after processing each Excel file,
including worksheet analysis, pass/fail status, and CSV generation details.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from io import BytesIO
import traceback

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
        PageBreak, KeepTogether, Image
    )
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    print(traceback.format_exc())
    REPORTLAB_AVAILABLE = False

from excel_to_csv.models.data_models import (
    WorksheetData, 
    ConfidenceScore, 
    HeaderInfo,
    ArchiveResult
)
from excel_to_csv.utils.logger import get_processing_logger
from .report_generator import (
    ExcelProcessingReport,
    WorksheetAnalysisReport,
    CSVGenerationReport
)


class PDFReportGenerator:
    """Generates professional PDF reports for Excel processing results."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the PDF report generator.
        
        Args:
            output_dir: Directory to write report files. Defaults to './reports'
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for PDF generation. "
                "Install with: pip install reportlab"
            )
        
        self.output_dir = Path(output_dir or './reports')
        self.logger = get_processing_logger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=8
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.darkgreen
        ))
        
        # Status styles
        self.styles.add(ParagraphStyle(
            name='StatusSuccess',
            parent=self.styles['Normal'],
            textColor=colors.darkgreen,
            fontSize=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='StatusFailed',
            parent=self.styles['Normal'],
            textColor=colors.red,
            fontSize=12,
            fontName='Helvetica-Bold'
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=20
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            textColor=colors.orange,
            fontSize=11,
            fontName='Helvetica-Oblique'
        ))
        
        # Error style
        self.styles.add(ParagraphStyle(
            name='Error',
            parent=self.styles['Normal'],
            textColor=colors.red,
            fontSize=11,
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(self, report: ExcelProcessingReport) -> Path:
        """Generate a comprehensive PDF report for Excel processing.
        
        Args:
            report: Excel processing report data
            
        Returns:
            Path to the generated PDF report file
        """
        self.logger.log_processing_start(
            f"Generating PDF report for {report.source_file.name}",
            len(str(report.source_file))
        )
        
        # Generate report filename with microseconds to avoid same-second collisions
        timestamp = report.processing_timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{report.source_file.stem}_{timestamp}_report.pdf"
        report_path = self.output_dir / filename
        
        # Fallback: if by any chance file exists, append a numeric suffix
        if report_path.exists():
            base = f"{report.source_file.stem}_{timestamp}_report"
            counter = 1
            while report_path.exists():
                report_path = self.output_dir / f"{base}_{counter}.pdf"
                counter += 1

        # Create PDF document
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Add content sections
        story.extend(self._create_header(report))
        story.extend(self._create_executive_summary(report))
        story.extend(self._create_worksheet_analysis(report))
        story.extend(self._create_csv_details(report))
        story.extend(self._create_performance_metrics(report))
        
        if report.archive_result:
            story.extend(self._create_archive_info(report.archive_result))
        
        if report.error_summary:
            story.extend(self._create_error_summary(report.error_summary))
        
        story.extend(self._create_technical_details(report))
        
        # Build PDF
        self.logger.info(f"[DEBUG] About to build PDF at: {report_path.resolve()}")
        doc.build(story)
        self.logger.info(f"[DEBUG] Finished building PDF at: {report_path.resolve()}")

        self.logger.log_processing_complete(
            f"PDF report generated: {report_path}",
            1,
            0.1,
            1
        )
        
        return report_path
    
    def _create_header(self, report: ExcelProcessingReport) -> List:
        """Create the report header section."""
        story = []
        
        # Title
        story.append(Paragraph("Excel Processing Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # File information table
        file_info_data = [
            ['File:', report.source_file.name],
            ['Processed:', report.processing_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Overall Status:', 'SUCCESS' if report.overall_success else 'FAILED'],
        ]
        
        file_info_table = Table(file_info_data, colWidths=[1.5*inch, 4*inch])
        file_info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        # Color code the status
        if report.overall_success:
            file_info_table.setStyle(TableStyle([
                ('TEXTCOLOR', (1, 2), (1, 2), colors.darkgreen),
                ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
            ]))
        else:
            file_info_table.setStyle(TableStyle([
                ('TEXTCOLOR', (1, 2), (1, 2), colors.red),
                ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
            ]))
        
        story.append(file_info_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_executive_summary(self, report: ExcelProcessingReport) -> List:
        """Create the executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Summary metrics table
        summary_data = [
            ['Total Worksheets', str(report.total_worksheets)],
            ['Worksheets Passed', f"{report.worksheets_passed} ✓"],
            ['Worksheets Failed', f"{report.worksheets_failed} ✗"],
            ['CSV Files Generated', str(report.total_csv_files)],
            ['Total Rows Processed', f"{report.total_rows_processed:,}"],
            ['Processing Time', f"{report.total_processing_time_ms:.2f}ms"],
            ['Confidence Threshold', f"{report.confidence_threshold:.2%}"],
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        # Color code pass/fail rows
        if report.worksheets_passed > 0:
            summary_table.setStyle(TableStyle([('TEXTCOLOR', (1, 1), (1, 1), colors.darkgreen)]))
        if report.worksheets_failed > 0:
            summary_table.setStyle(TableStyle([('TEXTCOLOR', (1, 2), (1, 2), colors.red)]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_worksheet_analysis(self, report: ExcelProcessingReport) -> List:
        """Create the worksheet analysis section."""
        story = []
        
        story.append(Paragraph("Worksheet Analysis Details", self.styles['SectionHeader']))
        
        for i, worksheet in enumerate(report.worksheets_analyzed, 1):
            # Keep worksheet analysis together
            worksheet_content = []
            
            # Worksheet header
            status_text = "PASSED ✓" if worksheet.passed else "FAILED ✗"
            status_style = self.styles['StatusSuccess'] if worksheet.passed else self.styles['StatusFailed']
            
            worksheet_content.append(
                Paragraph(f"{i}. Worksheet: '{worksheet.worksheet_name}'", self.styles['SubsectionHeader'])
            )
            worksheet_content.append(Paragraph(f"Status: {status_text}", status_style))
            worksheet_content.append(Spacer(1, 8))
            
            # Data metrics table
            metrics_data = [
                ['Rows', f"{worksheet.row_count:,}"],
                ['Columns', str(worksheet.column_count)],
                ['Non-empty cells', f"{worksheet.non_empty_cells:,}"],
                ['Data density', f"{worksheet.data_density:.2%}"],
                ['Processing time', f"{worksheet.processing_time_ms:.2f}ms"],
            ]
            
            metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.5*inch])
            metrics_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            
            worksheet_content.append(Paragraph("Data Metrics:", self.styles['Normal']))
            worksheet_content.append(metrics_table)
            worksheet_content.append(Spacer(1, 8))
            
            # Confidence analysis
            if worksheet.confidence_score:
                confidence_content = []
                score = worksheet.confidence_score
                
                confidence_data = [
                    ['Overall score', f"{score.overall_score:.3f}"],
                    ['Data density score', f"{score.data_density:.3f}"],
                    ['Header quality score', f"{score.header_quality:.3f}"],
                    ['Consistency score', f"{score.consistency_score:.3f}"],
                    ['Threshold', f"{score.threshold:.3f}"],
                    ['Is confident', 'Yes' if score.is_confident else 'No'],
                ]
                
                confidence_table = Table(confidence_data, colWidths=[1.8*inch, 1.5*inch])
                confidence_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ]))
                
                # Color code the confidence result
                confidence_color = colors.darkgreen if score.is_confident else colors.red
                confidence_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (1, 5), (1, 5), confidence_color),
                    ('FONTNAME', (1, 5), (1, 5), 'Helvetica-Bold'),
                ]))
                
                worksheet_content.append(Paragraph("Confidence Analysis:", self.styles['Normal']))
                worksheet_content.append(confidence_table)
                
                # Analysis reasons
                if score.reasons:
                    worksheet_content.append(Spacer(1, 6))
                    worksheet_content.append(Paragraph("Analysis Reasons:", self.styles['Normal']))
                    for reason in score.reasons:
                        worksheet_content.append(Paragraph(f"• {reason}", self.styles['Metric']))
                
                worksheet_content.append(Spacer(1, 8))
            
            # Header information
            if worksheet.header_info:
                header = worksheet.header_info
                header_data = [
                    ['Has headers', 'Yes' if header.has_headers else 'No'],
                    ['Header row', str(header.header_row) if header.header_row is not None else 'N/A'],
                    ['Header quality', f"{header.header_quality:.3f}"],
                    ['Column count', str(len(header.column_names))],
                ]
                
                header_table = Table(header_data, colWidths=[1.8*inch, 1.5*inch])
                header_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ]))
                
                worksheet_content.append(Paragraph("Header Information:", self.styles['Normal']))
                worksheet_content.append(header_table)
                worksheet_content.append(Spacer(1, 8))
            
            # Issues found
            if worksheet.issues_found:
                worksheet_content.append(Paragraph("Issues Found:", self.styles['Normal']))
                for issue in worksheet.issues_found:
                    worksheet_content.append(Paragraph(f"• {issue}", self.styles['Warning']))
                worksheet_content.append(Spacer(1, 8))
            
            # CSV generation info
            if worksheet.generated_csv:
                worksheet_content.append(Paragraph("Generated CSV:", self.styles['Normal']))
                worksheet_content.append(Paragraph(
                    f"• File: {worksheet.generated_csv.name}",
                    self.styles['Metric']
                ))
                worksheet_content.append(Paragraph(
                    f"• Size: {self._format_file_size(worksheet.csv_size_bytes)}",
                    self.styles['Metric']
                ))
            
            # Add all worksheet content as a kept-together block
            story.append(KeepTogether(worksheet_content))
            story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.lightgrey))
            story.append(Spacer(1, 12))
        
        return story
    
    def _create_csv_details(self, report: ExcelProcessingReport) -> List:
        """Create the CSV generation details section."""
        story = []
        
        story.append(Paragraph("CSV Files Generated", self.styles['SectionHeader']))
        
        if not report.csv_files_generated:
            story.append(Paragraph("No CSV files were generated.", self.styles['Normal']))
            story.append(Spacer(1, 12))
            return story
        
        story.append(Paragraph(f"Total CSV files: {len(report.csv_files_generated)}", self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # CSV details table
        csv_headers = ['File Name', 'Worksheet', 'Rows', 'Columns', 'Size', 'Status']
        csv_data = [csv_headers]
        
        for csv in report.csv_files_generated:
            status = "✓ Success" if csv.success else "✗ Failed"
            csv_data.append([
                csv.csv_file_path.name,
                csv.worksheet_name,
                f"{csv.rows_written:,}",
                str(csv.columns_written),
                self._format_file_size(csv.file_size_bytes),
                status
            ])
        
        csv_table = Table(csv_data, colWidths=[2*inch, 1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
        csv_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (4, -1), 'RIGHT'),  # Right align numeric columns
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        # Color code status column
        for i, csv in enumerate(report.csv_files_generated, 1):
            status_color = colors.darkgreen if csv.success else colors.red
            csv_table.setStyle(TableStyle([
                ('TEXTCOLOR', (5, i), (5, i), status_color),
                ('FONTNAME', (5, i), (5, i), 'Helvetica-Bold'),
            ]))
        
        story.append(csv_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_performance_metrics(self, report: ExcelProcessingReport) -> List:
        """Create the performance metrics section."""
        story = []
        
        story.append(Paragraph("Performance Metrics", self.styles['SectionHeader']))
        
        # Calculate additional metrics
        avg_time_per_worksheet = (report.total_processing_time_ms / 
                                 max(report.total_worksheets, 1))
        
        processing_speed = 0
        if report.total_rows_processed > 0 and report.total_processing_time_ms > 0:
            processing_speed = report.total_rows_processed / (report.total_processing_time_ms / 1000)
        
        performance_data = [
            ['Total processing time', f"{report.total_processing_time_ms:.2f}ms"],
            ['Average time per worksheet', f"{avg_time_per_worksheet:.2f}ms"],
            ['Processing speed', f"{processing_speed:,.0f} rows/second" if processing_speed > 0 else "N/A"],
            ['Memory efficiency', f"{report.total_rows_processed:,} rows processed"],
        ]
        
        performance_table = Table(performance_data, colWidths=[2.5*inch, 2*inch])
        performance_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(performance_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_archive_info(self, archive_result: ArchiveResult) -> List:
        """Create the archive information section."""
        story = []
        
        story.append(Paragraph("Archive Information", self.styles['SectionHeader']))
        
        archive_data = [
            ['Archive status', 'SUCCESS ✓' if archive_result.success else 'FAILED ✗'],
            ['Source path', str(archive_result.source_path)],
        ]
        
        if archive_result.archive_path:
            archive_data.append(['Archive path', str(archive_result.archive_path)])
        
        if archive_result.timestamp_used:
            archive_data.append(['Timestamp used', archive_result.timestamp_used])
        
        archive_data.append(['Operation time', f"{archive_result.operation_time:.3f}s"])
        
        if archive_result.error_message:
            archive_data.append(['Error message', archive_result.error_message])
        
        archive_table = Table(archive_data, colWidths=[1.8*inch, 3.5*inch])
        archive_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        # Color code the status
        status_color = colors.darkgreen if archive_result.success else colors.red
        archive_table.setStyle(TableStyle([
            ('TEXTCOLOR', (1, 0), (1, 0), status_color),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ]))
        
        # Color error message if present
        if archive_result.error_message:
            archive_table.setStyle(TableStyle([
                ('TEXTCOLOR', (1, -1), (1, -1), colors.red),
            ]))
        
        story.append(archive_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_error_summary(self, errors: List[str]) -> List:
        """Create the error summary section."""
        story = []
        
        story.append(Paragraph("Error Summary", self.styles['SectionHeader']))
        story.append(Paragraph(f"Total errors encountered: {len(errors)}", self.styles['Normal']))
        story.append(Spacer(1, 8))
        
        for i, error in enumerate(errors, 1):
            story.append(Paragraph(f"{i}. {error}", self.styles['Error']))
        
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_technical_details(self, report: ExcelProcessingReport) -> List:
        """Create the technical details section."""
        story = []
        
        story.append(Paragraph("Technical Details", self.styles['SectionHeader']))
        
        # Get file size if available
        file_size = "Unknown"
        try:
            if report.source_file.exists():
                file_size = self._format_file_size(report.source_file.stat().st_size)
        except Exception:
            pass
        
        technical_data = [
            ['Source file', str(report.source_file)],
            ['File size', file_size],
            ['Processing timestamp', report.processing_timestamp.isoformat()],
            ['Report generated', datetime.now().isoformat()],
        ]
        
        technical_table = Table(technical_data, colWidths=[2*inch, 3.5*inch])
        technical_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        story.append(technical_table)
        
        # Add worksheet summary table if there are worksheets
        if report.worksheets_analyzed:
            story.append(Spacer(1, 16))
            story.append(Paragraph("Worksheet Summary Table", self.styles['SubsectionHeader']))
            
            summary_headers = ['Worksheet', 'Status', 'Rows', 'Columns', 'Confidence', 'CSV Generated']
            summary_data = [summary_headers]
            
            for ws in report.worksheets_analyzed:
                status = "✓ Pass" if ws.passed else "✗ Fail"
                confidence = f"{ws.confidence_score.overall_score:.3f}" if ws.confidence_score else "N/A"
                csv_gen = "Yes" if ws.generated_csv else "No"
                
                summary_data.append([
                    ws.worksheet_name,
                    status,
                    f"{ws.row_count:,}",
                    str(ws.column_count),
                    confidence,
                    csv_gen
                ])
            
            summary_table = Table(summary_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            summary_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (2, 1), (3, -1), 'RIGHT'),  # Right align numeric columns
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            # Color code status column
            for i, ws in enumerate(report.worksheets_analyzed, 1):
                status_color = colors.darkgreen if ws.passed else colors.red
                summary_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (1, i), (1, i), status_color),
                    ('FONTNAME', (1, i), (1, i), 'Helvetica-Bold'),
                ]))
            
            story.append(summary_table)
        
        return story
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    # Helper methods from original ReportGenerator for compatibility
    def create_worksheet_report(self, *args, **kwargs) -> WorksheetAnalysisReport:
        """Create a worksheet analysis report (delegates to original generator)."""
        from .report_generator import ReportGenerator
        temp_generator = ReportGenerator()
        return temp_generator.create_worksheet_report(*args, **kwargs)
    
    def create_csv_report(self, *args, **kwargs) -> CSVGenerationReport:
        """Create a CSV generation report (delegates to original generator)."""
        from .report_generator import ReportGenerator
        temp_generator = ReportGenerator()
        return temp_generator.create_csv_report(*args, **kwargs)

