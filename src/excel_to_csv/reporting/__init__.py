"""Reporting module for Excel processing results.

This module provides comprehensive reporting capabilities for Excel-to-CSV
conversion operations, including detailed PDF and markdown reports with worksheet
analysis, pass/fail status, and CSV generation details.
"""

from .report_generator import (
    ReportGenerator,
    ExcelProcessingReport,
    WorksheetAnalysisReport,
    CSVGenerationReport
)

try:
    from .pdf_report_generator import PDFReportGenerator
    PDF_AVAILABLE = True
except ImportError:
    PDFReportGenerator = None
    PDF_AVAILABLE = False

__all__ = [
    'ReportGenerator',
    'PDFReportGenerator',
    'ExcelProcessingReport', 
    'WorksheetAnalysisReport',
    'CSVGenerationReport',
    'PDF_AVAILABLE'
]