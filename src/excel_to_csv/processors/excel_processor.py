"""Excel file processing for Excel-to-CSV converter.

This module provides robust Excel file processing capabilities including:
- Reading various Excel formats (.xlsx, .xls)
- Extracting worksheet metadata and data
- Error handling for corrupt or locked files
- Memory-efficient processing of large files
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import openpyxl
from openpyxl.utils.exceptions import InvalidFileException

from excel_to_csv.models.data_models import WorksheetData
from excel_to_csv.utils.logger import get_processing_logger


class ExcelProcessingError(Exception):
    """Raised when Excel file processing fails."""
    pass


class ExcelProcessor:
    """Processes Excel files and extracts worksheet data.
    
    The ExcelProcessor handles:
    - Reading Excel files in various formats
    - Extracting worksheet metadata and data
    - Error handling for various file conditions
    - Memory-efficient processing
    
    Example:
        >>> processor = ExcelProcessor()
        >>> worksheets = processor.process_file("data.xlsx")
        >>> for ws in worksheets:
        ...     print(f"Worksheet: {ws.worksheet_name}, Rows: {ws.row_count}")
    """
    
    # Supported Excel file extensions
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xls'}
    
    # Maximum number of rows to preview for metadata
    METADATA_PREVIEW_ROWS = 100
    
    def __init__(self, max_file_size_mb: int = 100, chunk_size: int = 10000):
        """Initialize Excel processor.
        
        Args:
            max_file_size_mb: Maximum file size to process in MB
            chunk_size: Chunk size for reading large files
        """
        self.max_file_size_mb = max_file_size_mb
        self.chunk_size = chunk_size
        self.logger = get_processing_logger(__name__)
    
    def process_file(self, file_path: Union[str, Path]) -> List[WorksheetData]:
        """Process Excel file and return worksheet data.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of WorksheetData objects for each worksheet
            
        Raises:
            ExcelProcessingError: If file cannot be processed
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        # Validate file
        self._validate_file(file_path)
        
        # Log processing start
        file_size = file_path.stat().st_size
        self.logger.log_processing_start(file_path, file_size)
        
        try:
            # Extract worksheets
            worksheets = self._extract_worksheets(file_path)
            
            # Log completion
            processing_time = time.time() - start_time
            self.logger.log_processing_complete(
                file_path, 
                len(worksheets), 
                processing_time,
                len(worksheets)  # All worksheets become potential CSV files
            )
            
            return worksheets
            
        except Exception as e:
            self.logger.log_error(
                error_type="excel_processing_failed",
                message=f"Failed to process Excel file {file_path}: {e}",
                file_path=file_path,
                exc_info=True
            )
            raise ExcelProcessingError(f"Excel processing failed: {e}") from e
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate Excel file before processing.
        
        Args:
            file_path: Path to Excel file
            
        Raises:
            ExcelProcessingError: If validation fails
        """
        # Check if file exists
        if not file_path.exists():
            raise ExcelProcessingError(f"File not found: {file_path}")
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            raise ExcelProcessingError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ExcelProcessingError(
                f"Unsupported file extension: {file_path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ExcelProcessingError(
                f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"
            )
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except (IOError, OSError, PermissionError) as e:
            raise ExcelProcessingError(f"Cannot read file {file_path}: {e}")
    
    def _extract_worksheets(self, file_path: Path) -> List[WorksheetData]:
        """Extract all worksheets from Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of WorksheetData objects
        """
        worksheets = []
        
        try:
            # First, get worksheet names and metadata using openpyxl
            workbook_info = self._get_workbook_info(file_path)
            
            # Process each worksheet
            with pd.ExcelFile(file_path, engine=None) as excel_file:
                for sheet_name in excel_file.sheet_names:
                    try:
                        worksheet_data = self._process_worksheet(
                            excel_file, 
                            sheet_name, 
                            file_path,
                            workbook_info.get(sheet_name, {})
                        )
                        if worksheet_data:
                            worksheets.append(worksheet_data)
                    except Exception as e:
                        self.logger.log_error(
                            error_type="worksheet_processing_failed",
                            message=f"Failed to process worksheet '{sheet_name}': {e}",
                            file_path=file_path,
                            worksheet_name=sheet_name
                        )
                        # Continue with other worksheets
                        continue
            
            return worksheets
            
        except Exception as e:
            raise ExcelProcessingError(f"Failed to extract worksheets: {e}") from e
    
    def _get_workbook_info(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Get workbook metadata using openpyxl.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary mapping worksheet names to metadata
        """
        workbook_info = {}
        
        try:
            # Only process .xlsx files with openpyxl (it doesn't support .xls)
            if file_path.suffix.lower() == '.xlsx':
                workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                
                for sheet_name in workbook.sheetnames:
                    worksheet = workbook[sheet_name]
                    
                    # Get basic worksheet info
                    workbook_info[sheet_name] = {
                        'max_row': worksheet.max_row,
                        'max_column': worksheet.max_column,
                        'sheet_state': getattr(worksheet, 'sheet_state', 'visible'),
                        'has_formulas': False,  # Will be determined during pandas processing
                    }
                
                workbook.close()
            
        except (InvalidFileException, Exception) as e:
            # Log warning but don't fail - we can still process with pandas
            self.logger.warning(f"Could not extract metadata with openpyxl: {e}")
        
        return workbook_info
    
    def _process_worksheet(
        self, 
        excel_file: pd.ExcelFile, 
        sheet_name: str,
        source_file: Path,
        metadata: Dict[str, Any]
    ) -> Optional[WorksheetData]:
        """Process individual worksheet and return WorksheetData.
        
        Args:
            excel_file: pandas ExcelFile object
            sheet_name: Name of worksheet to process
            source_file: Source Excel file path
            metadata: Worksheet metadata from openpyxl
            
        Returns:
            WorksheetData object or None if worksheet is empty/invalid
        """
        try:
            # Read worksheet data
            df = self._read_worksheet_data(excel_file, sheet_name)
            
            # Skip completely empty worksheets
            if df.empty:
                self.logger.debug(f"Skipping empty worksheet: {sheet_name}")
                return None
            
            # Skip worksheets with no actual data
            if df.dropna(how='all').empty:
                self.logger.debug(f"Skipping worksheet with no data: {sheet_name}")
                return None
            
            # Enhance metadata with pandas-derived info
            enhanced_metadata = self._enhance_metadata(df, metadata)
            
            # Create WorksheetData object
            worksheet_data = WorksheetData(
                source_file=source_file,
                worksheet_name=sheet_name,
                data=df,
                metadata=enhanced_metadata
            )
            
            self.logger.debug(
                f"Processed worksheet '{sheet_name}': "
                f"{worksheet_data.row_count} rows, "
                f"{worksheet_data.column_count} columns, "
                f"density: {worksheet_data.data_density:.3f}"
            )
            
            return worksheet_data
            
        except Exception as e:
            self.logger.log_error(
                error_type="worksheet_read_failed",
                message=f"Failed to read worksheet '{sheet_name}': {e}",
                file_path=source_file,
                worksheet_name=sheet_name
            )
            return None
    
    def _read_worksheet_data(self, excel_file: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
        """Read worksheet data using pandas.
        
        Args:
            excel_file: pandas ExcelFile object
            sheet_name: Name of worksheet to read
            
        Returns:
            DataFrame containing worksheet data
        """
        try:
            # Read the worksheet
            df = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                header=None,  # Don't assume first row is header
                na_values=[''],  # Treat empty strings as NaN
                keep_default_na=True,
                na_filter=True,
                dtype=str,  # Read everything as string initially
                engine=None  # Let pandas choose the best engine
            )
            
            return df
            
        except Exception as e:
            raise ExcelProcessingError(f"Failed to read worksheet data: {e}") from e
    
    def _enhance_metadata(self, df: pd.DataFrame, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance metadata with pandas-derived information.
        
        Args:
            df: DataFrame containing worksheet data
            base_metadata: Base metadata from openpyxl
            
        Returns:
            Enhanced metadata dictionary
        """
        metadata = base_metadata.copy()
        
        # Basic data info
        metadata.update({
            'pandas_shape': df.shape,
            'pandas_columns': list(df.columns),
            'pandas_dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            'total_cells': df.size,
            'non_empty_cells': df.count().sum(),
            'empty_cells': df.isnull().sum().sum(),
            'data_density': (df.count().sum() / df.size) if df.size > 0 else 0.0,
        })
        
        # Data type analysis
        numeric_columns = []
        date_columns = []
        text_columns = []
        
        for col in df.columns:
            # Sample non-null values for type detection
            sample_data = df[col].dropna().head(100)
            
            if len(sample_data) == 0:
                continue
            
            # Try to detect numeric columns
            try:
                pd.to_numeric(sample_data, errors='raise')
                numeric_columns.append(col)
                continue
            except (ValueError, TypeError):
                pass
            
            # Try to detect date columns
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(sample_data, errors='raise')
                date_columns.append(col)
                continue
            except (ValueError, TypeError):
                pass
            
            # Default to text
            text_columns.append(col)
        
        metadata.update({
            'numeric_columns': numeric_columns,
            'date_columns': date_columns, 
            'text_columns': text_columns,
            'column_types_detected': {
                'numeric': len(numeric_columns),
                'date': len(date_columns),
                'text': len(text_columns),
            }
        })
        
        # Row analysis
        empty_rows = df.isnull().all(axis=1).sum()
        full_rows = df.notnull().all(axis=1).sum()
        
        metadata.update({
            'empty_rows': empty_rows,
            'full_rows': full_rows,
            'partial_rows': len(df) - empty_rows - full_rows,
        })
        
        # Column analysis
        empty_columns = df.isnull().all(axis=0).sum()
        full_columns = df.notnull().all(axis=0).sum()
        
        metadata.update({
            'empty_columns': empty_columns,
            'full_columns': full_columns,
            'partial_columns': len(df.columns) - empty_columns - full_columns,
        })
        
        # Potential header detection (simple heuristic)
        if len(df) > 0:
            first_row_text_ratio = 0
            if len(df.columns) > 0:
                first_row = df.iloc[0]
                text_cells = sum(1 for val in first_row if isinstance(val, str) and val.strip())
                first_row_text_ratio = text_cells / len(first_row)
            
            metadata['potential_header_row'] = 0 if first_row_text_ratio > 0.5 else None
            metadata['first_row_text_ratio'] = first_row_text_ratio
        
        return metadata
    
    def get_worksheet_preview(
        self, 
        file_path: Union[str, Path], 
        sheet_name: str,
        max_rows: int = 10
    ) -> pd.DataFrame:
        """Get preview of worksheet data.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of worksheet
            max_rows: Maximum number of rows to return
            
        Returns:
            DataFrame with preview data
            
        Raises:
            ExcelProcessingError: If preview cannot be generated
        """
        file_path = Path(file_path)
        
        try:
            with pd.ExcelFile(file_path) as excel_file:
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    header=None,
                    nrows=max_rows,
                    dtype=str
                )
                return df
                
        except Exception as e:
            raise ExcelProcessingError(f"Failed to generate preview: {e}") from e
    
    def list_worksheets(self, file_path: Union[str, Path]) -> List[str]:
        """List all worksheet names in Excel file.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            List of worksheet names
            
        Raises:
            ExcelProcessingError: If worksheets cannot be listed
        """
        file_path = Path(file_path)
        
        try:
            with pd.ExcelFile(file_path) as excel_file:
                return excel_file.sheet_names
                
        except Exception as e:
            raise ExcelProcessingError(f"Failed to list worksheets: {e}") from e