"""Excel-to-CSV Converter.

An intelligent automation tool for converting Excel worksheets to CSV files
with confidence-based data table detection. Supports both CLI processing
and continuous service monitoring.
"""

__version__ = "1.0.0"
__author__ = "Excel-to-CSV Converter Team"
__email__ = "info@example.com"

from excel_to_csv.models.data_models import (
    WorksheetData,
    ConfidenceScore,
    Config,
    HeaderInfo,
    OutputConfig,
)
from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.processors.excel_processor import ExcelProcessor
from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
from excel_to_csv.generators.csv_generator import CSVGenerator
from excel_to_csv.monitoring.file_monitor import FileMonitor

__all__ = [
    "WorksheetData",
    "ConfidenceScore", 
    "Config",
    "HeaderInfo",
    "OutputConfig",
    "ExcelToCSVConverter",
    "ExcelProcessor",
    "ConfidenceAnalyzer", 
    "CSVGenerator",
    "FileMonitor",
]