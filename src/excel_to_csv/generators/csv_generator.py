"""CSV file generation for Excel-to-CSV converter.

This module provides robust CSV generation capabilities including:
- Proper encoding and formatting
- Intelligent filename generation
- Special character handling
- Timestamp-based duplicate handling
- Data type preservation and formatting
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from excel_to_csv.models.data_models import WorksheetData, OutputConfig
from excel_to_csv.utils.logger import get_processing_logger
from excel_to_csv.utils.logging_decorators import log_operation, log_method, operation_context
from excel_to_csv.utils.correlation import CorrelationContext


class CSVGenerationError(Exception):
    """Raised when CSV generation fails."""
    pass


class CSVGenerator:
    """Generates CSV files from qualified worksheet data.
    
    The CSVGenerator handles:
    - Converting worksheet data to properly formatted CSV
    - Generating meaningful filenames with collision handling
    - Preserving data integrity and formatting
    - Supporting various encoding and delimiter options
    
    Example:
        >>> generator = CSVGenerator()
        >>> output_path = generator.generate_csv(worksheet_data, config)
        >>> print(f"Generated: {output_path}")
    """
    
    # Characters to remove/replace in filenames for filesystem safety
    UNSAFE_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    
    # Maximum filename length (leaving room for extension and timestamp)
    MAX_FILENAME_LENGTH = 200
    
    @log_operation("initialize_csv_generator")
    def __init__(self):
        """Initialize CSV generator."""
        with operation_context(
            "csv_generator_initialization",
            logger=None,  # Will use default logger
            max_filename_length=self.MAX_FILENAME_LENGTH
        ) as metrics:
            
            self.logger = get_processing_logger(__name__)
            
            metrics.add_metadata("unsafe_chars_pattern", self.UNSAFE_FILENAME_CHARS)
            metrics.add_metadata("max_filename_length", self.MAX_FILENAME_LENGTH)
            
            self.logger.info(
                "CSV Generator initialized successfully",
                extra={
                    "structured": {
                        "operation": "csv_generator_init_success",
                        "max_filename_length": self.MAX_FILENAME_LENGTH,
                        "unsafe_chars_pattern": self.UNSAFE_FILENAME_CHARS
                    }
                }
            )
    
    @log_operation("generate_csv_file", log_args=False)
    def generate_csv(
        self, 
        worksheet_data: WorksheetData, 
        output_config: OutputConfig
    ) -> Path:
        """Generate CSV file from worksheet data.
        
        Args:
            worksheet_data: WorksheetData object to convert
            output_config: Output configuration settings
            
        Returns:
            Path to generated CSV file
            
        Raises:
            CSVGenerationError: If CSV generation fails
        """
        with operation_context(
            "csv_file_generation",
            self.logger,
            worksheet_name=worksheet_data.worksheet_name,
            source_file=str(worksheet_data.source_file),
            row_count=worksheet_data.row_count,
            column_count=worksheet_data.column_count,
            delimiter=output_config.delimiter,
            encoding=output_config.encoding
        ) as metrics:
            
            try:
                self.logger.info(
                    f"Starting CSV generation for worksheet '{worksheet_data.worksheet_name}' - {worksheet_data.row_count:,} rows x {worksheet_data.column_count} columns",
                    extra={
                        "structured": {
                            "operation": "csv_generation_start",
                            "worksheet_name": worksheet_data.worksheet_name,
                            "source_file": str(worksheet_data.source_file),
                            "row_count": worksheet_data.row_count,
                            "column_count": worksheet_data.column_count,
                            "data_density": worksheet_data.data_density,
                            "output_config": {
                                "delimiter": output_config.delimiter,
                                "encoding": output_config.encoding,
                                "include_headers": output_config.include_headers,
                                "include_timestamp": output_config.include_timestamp
                            }
                        }
                    }
                )
                
                # Determine output path
                output_path = self._determine_output_path(worksheet_data, output_config)
                metrics.add_metadata("initial_output_path", str(output_path))
                
                # Handle duplicate files
                output_path = self._handle_duplicates(output_path, output_config)
                metrics.add_metadata("final_output_path", str(output_path))
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Generate CSV content
                csv_data = self._prepare_csv_data(worksheet_data, output_config)
                
                # Write CSV file
                self._write_csv_file(csv_data, output_path, output_config)
                
                # Get file statistics
                file_size = output_path.stat().st_size
                data_rows = len(csv_data)
                data_columns = len(csv_data.columns) if not csv_data.empty else 0
                
                # Add comprehensive metrics
                metrics.add_metadata("output_file_size", file_size)
                metrics.add_metadata("csv_rows", data_rows)
                metrics.add_metadata("csv_columns", data_columns)
                metrics.add_metadata("generation_success", True)
                
                # Log successful generation
                self.logger.info(
                    f"CSV generation completed: {output_path} ({file_size:,} bytes, {data_rows:,} rows)",
                    extra={
                        "structured": {
                            "operation": "csv_generation_success",
                            "output_path": str(output_path),
                            "file_size_bytes": file_size,
                            "csv_rows": data_rows,
                            "csv_columns": data_columns,
                            "worksheet_name": worksheet_data.worksheet_name,
                            "source_file": str(worksheet_data.source_file)
                        }
                    }
                )
                
                return output_path
                
            except Exception as e:
                error_type = type(e).__name__
                metrics.add_metadata("generation_success", False)
                metrics.add_metadata("error_type", error_type)
                
                self.logger.error(
                    f"CSV generation failed for '{worksheet_data.worksheet_name}': {e}",
                    exc_info=True,
                    extra={
                        "structured": {
                            "operation": "csv_generation_failed",
                            "worksheet_name": worksheet_data.worksheet_name,
                            "source_file": str(worksheet_data.source_file),
                            "error_type": error_type,
                            "error_message": str(e)
                        }
                    }
                )
                raise CSVGenerationError(f"CSV generation failed: {e}") from e
    
    @log_operation("determine_output_path", log_args=False)
    def _determine_output_path(
        self, 
        worksheet_data: WorksheetData, 
        output_config: OutputConfig
    ) -> Path:
        """Determine the output path for the CSV file.
        
        Args:
            worksheet_data: Worksheet data
            output_config: Output configuration
            
        Returns:
            Path for output CSV file
        """
        with operation_context(
            "output_path_determination",
            self.logger,
            worksheet_name=worksheet_data.worksheet_name,
            source_file=str(worksheet_data.source_file)
        ) as metrics:
            
            # Get source filename without extension
            source_filename = worksheet_data.source_file.stem
            
            # Generate filename using pattern
            csv_filename = output_config.generate_filename(
                source_filename, 
                worksheet_data.worksheet_name
            )
            
            # Sanitize filename for filesystem safety
            csv_filename = self._sanitize_filename(csv_filename)
            
            # Determine output directory
            if output_config.folder:
                output_dir = output_config.folder
            else:
                # Use same directory as source file
                output_dir = worksheet_data.source_file.parent
            
            final_path = output_dir / csv_filename
        
            # Log path determination
            metrics.add_metadata("source_filename", source_filename)
            metrics.add_metadata("generated_filename", csv_filename)
            metrics.add_metadata("output_directory", str(output_dir))
            metrics.add_metadata("final_path", str(final_path))
            
            self.logger.debug(
                f"Determined output path: {final_path} for worksheet '{worksheet_data.worksheet_name}'",
                extra={
                    "structured": {
                        "operation": "output_path_determined",
                        "worksheet_name": worksheet_data.worksheet_name,
                        "source_filename": source_filename,
                        "generated_filename": csv_filename,
                        "output_directory": str(output_dir),
                        "final_path": str(final_path)
                    }
                }
            )
            
            return final_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem safety.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove/replace unsafe characters
        sanitized = re.sub(self.UNSAFE_FILENAME_CHARS, '_', filename)
        
        # Replace multiple consecutive underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores and dots
        sanitized = sanitized.strip('_.')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "worksheet"
        
        # Truncate if too long (leave room for extension)
        name_part, ext = os.path.splitext(sanitized)
        if len(name_part) > self.MAX_FILENAME_LENGTH:
            name_part = name_part[:self.MAX_FILENAME_LENGTH]
            sanitized = name_part + ext
        
        # Ensure it has .csv extension
        if not sanitized.lower().endswith('.csv'):
            sanitized += '.csv'
        
        return sanitized
    
    @log_operation("handle_duplicate_files", log_args=False)
    def _handle_duplicates(self, output_path: Path, output_config: OutputConfig) -> Path:
        """Handle duplicate filenames.
        
        Args:
            output_path: Proposed output path
            output_config: Output configuration
            
        Returns:
            Final output path (possibly with timestamp)
        """
        with operation_context(
            "duplicate_file_handling",
            self.logger,
            original_path=str(output_path),
            include_timestamp=output_config.include_timestamp
        ) as metrics:
            
            if not output_path.exists():
                metrics.add_metadata("file_exists", False)
                metrics.add_metadata("resolution_method", "no_conflict")
                self.logger.debug(f"No duplicate file conflict: {output_path}")
                return output_path
            
            if not output_config.include_timestamp:
                # Overwrite existing file
                metrics.add_metadata("file_exists", True)
                metrics.add_metadata("resolution_method", "overwrite")
                
                self.logger.warning(
                    f"Overwriting existing file: {output_path}",
                    extra={
                        "structured": {
                            "operation": "duplicate_resolved_by_overwrite",
                            "overwritten_path": str(output_path)
                        }
                    }
                )
                return output_path
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime(output_config.timestamp_format)
            stem = output_path.stem
            suffix = output_path.suffix
            
            timestamped_name = f"{stem}_{timestamp}{suffix}"
            timestamped_path = output_path.parent / timestamped_name
            
            # If timestamped version also exists, add a counter
            counter = 1
            while timestamped_path.exists():
                timestamped_name = f"{stem}_{timestamp}_{counter:03d}{suffix}"
                timestamped_path = output_path.parent / timestamped_name
                counter += 1
            
            self.logger.info(
            f"Added timestamp to avoid duplicate: {timestamped_path}",
            extra={
                "structured": {
                    "operation": "duplicate_resolved_with_timestamp",
                    "original_path": str(output_path),
                    "timestamped_path": str(timestamped_path),
                    "timestamp": timestamp,
                    "counter_used": counter - 1 if counter > 1 else 0
                }
            }
        )
        
        metrics.add_metadata("resolution_method", "timestamp_added")
        metrics.add_metadata("final_path", str(timestamped_path))
        metrics.add_metadata("counter_used", counter - 1 if counter > 1 else 0)
        
        return timestamped_path
    
    @log_operation("prepare_csv_data", log_args=False)
    def _prepare_csv_data(
        self, 
        worksheet_data: WorksheetData, 
        output_config: OutputConfig
    ) -> pd.DataFrame:
        """Prepare worksheet data for CSV output.
        
        Args:
            worksheet_data: Worksheet data to prepare
            output_config: Output configuration
            
        Returns:
            Prepared DataFrame for CSV output
        """
        with operation_context(
            "csv_data_preparation",
            self.logger,
            worksheet_name=worksheet_data.worksheet_name,
            include_headers=output_config.include_headers,
            original_shape=worksheet_data.data.shape
        ) as metrics:
            df = worksheet_data.data.copy()
            
            # Handle headers
            if output_config.include_headers:
                df = self._setup_headers(df, worksheet_data)
            else:
                # Reset column names to indices if not including headers
                df.columns = range(len(df.columns))
            
            # Clean and format data
            df = self._clean_data_for_csv(df)
            
            # Log data preparation results
            final_shape = df.shape
            metrics.add_metadata("final_shape", final_shape)
            metrics.add_metadata("headers_processed", output_config.include_headers)
            metrics.add_metadata("data_cleaned", True)
            
            self.logger.debug(
                f"CSV data prepared: {final_shape[0]:,} rows x {final_shape[1]} columns for '{worksheet_data.worksheet_name}'",
                extra={
                    "structured": {
                        "operation": "csv_data_preparation_complete",
                        "worksheet_name": worksheet_data.worksheet_name,
                        "original_shape": worksheet_data.data.shape,
                        "final_shape": final_shape,
                        "include_headers": output_config.include_headers,
                        "column_names": list(df.columns)[:5] if len(df.columns) <= 5 else list(df.columns)[:5] + ["..."]
                    }
                }
            )
            
            return df
    
    def _setup_headers(self, df: pd.DataFrame, worksheet_data: WorksheetData) -> pd.DataFrame:
        """Set up appropriate headers for the CSV.
        
        Args:
            df: DataFrame to process
            worksheet_data: Original worksheet data
            
        Returns:
            DataFrame with proper headers
        """
        # Try to detect if first row should be headers
        if len(df) > 0:
            first_row = df.iloc[0]
            
            # Check if first row looks like headers
            if self._first_row_looks_like_headers(first_row, df):
                # Use first row as headers and remove it from data
                df.columns = [str(val).strip() if pd.notna(val) else f"Column_{i}" 
                             for i, val in enumerate(first_row)]
                df = df.iloc[1:].reset_index(drop=True)
                
                self.logger.debug(f"Using first row as headers for {worksheet_data.worksheet_name}")
            else:
                # Generate default column names
                df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]
                
                self.logger.debug(f"Generated default headers for {worksheet_data.worksheet_name}")
        
        # Ensure headers are unique
        df = self._ensure_unique_headers(df)
        
        return df
    
    def _first_row_looks_like_headers(self, first_row: pd.Series, df: pd.DataFrame) -> bool:
        """Determine if first row looks like headers.
        
        Args:
            first_row: First row of data
            df: Full DataFrame
            
        Returns:
            True if first row appears to be headers
        """
        # Count text cells in first row
        text_cells = sum(1 for val in first_row 
                        if pd.notna(val) and isinstance(val, str) and val.strip())
        
        text_ratio = text_cells / len(first_row) if len(first_row) > 0 else 0
        
        # If most cells in first row are text, likely headers
        if text_ratio > 0.6:
            return True
        
        # Check if second row has different data types (suggests first row is headers)
        if len(df) > 1:
            second_row = df.iloc[1]
            
            # Compare data types between first and second row
            type_differences = 0
            for val1, val2 in zip(first_row, second_row):
                if pd.notna(val1) and pd.notna(val2):
                    # Check if one is string and other is numeric
                    val1_str = isinstance(val1, str) and not val1.replace('.', '').replace('-', '').isdigit()
                    val2_numeric = str(val2).replace('.', '').replace('-', '').isdigit()
                    
                    if val1_str and val2_numeric:
                        type_differences += 1
            
            type_diff_ratio = type_differences / len(first_row) if len(first_row) > 0 else 0
            if type_diff_ratio > 0.3:
                return True
        
        return False
    
    def _ensure_unique_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all column headers are unique.
        
        Args:
            df: DataFrame with potentially duplicate headers
            
        Returns:
            DataFrame with unique headers
        """
        columns = list(df.columns)
        seen = {}
        
        for i, col in enumerate(columns):
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                columns[i] = f"{col_str}_{seen[col_str]}"
            else:
                seen[col_str] = 0
        
        df.columns = columns
        return df
    
    def _clean_data_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format data for CSV output.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Replace NaN values with empty strings for cleaner CSV
        df_clean = df_clean.fillna('')
        
        # Clean string data
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Strip whitespace from string columns
                df_clean[col] = df_clean[col].astype(str).str.strip()
                
                # Replace 'nan' strings with empty strings
                df_clean[col] = df_clean[col].replace('nan', '')
        
        # Handle numeric formatting
        df_clean = self._format_numeric_columns(df_clean)
        
        # Handle date formatting
        df_clean = self._format_date_columns(df_clean)
        
        return df_clean
    
    def _format_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format numeric columns for CSV output.
        
        Args:
            df: DataFrame to format
            
        Returns:
            DataFrame with formatted numeric columns
        """
        for col in df.columns:
            # Try to identify numeric columns
            try:
                # Check if column can be converted to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # If most values are numeric, format the column
                valid_numeric = numeric_series.dropna()
                if len(valid_numeric) > len(df) * 0.7:  # 70% threshold
                    # Format numbers to remove unnecessary decimal places
                    df[col] = numeric_series.apply(
                        lambda x: f"{x:g}" if pd.notna(x) else ""
                    )
            except (ValueError, TypeError):
                # Skip columns that can't be processed
                continue
        
        return df
    
    def _format_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format date columns for CSV output.
        
        Args:
            df: DataFrame to format
            
        Returns:
            DataFrame with formatted date columns
        """
        for col in df.columns:
            # Try to identify date columns
            try:
                # Sample some values to test date conversion
                sample_data = df[col].dropna().head(10)
                if len(sample_data) == 0:
                    continue
                
                # Try to convert to datetime
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    date_series = pd.to_datetime(sample_data, errors='coerce')
                valid_dates = date_series.dropna()
                
                # If most sample values are dates, format the entire column
                if len(valid_dates) > len(sample_data) * 0.8:  # 80% threshold
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    df[col] = df[col].fillna('')  # Replace NaT with empty string
                    
                    self.logger.debug(f"Formatted date column: {col}")
            
            except (ValueError, TypeError):
                # Skip columns that can't be processed
                continue
        
        return df
    
    @log_operation("write_csv_file", log_args=False)
    def _write_csv_file(
        self, 
        df: pd.DataFrame, 
        output_path: Path, 
        output_config: OutputConfig
    ) -> None:
        """Write DataFrame to CSV file.
        
        Args:
            df: DataFrame to write
            output_path: Path to output file
            output_config: Output configuration
            
        Raises:
            CSVGenerationError: If writing fails
        """
        with operation_context(
            "csv_file_writing",
            self.logger,
            output_path=str(output_path),
            dataframe_shape=df.shape,
            delimiter=output_config.delimiter,
            encoding=output_config.encoding
        ) as metrics:
            try:
                df.to_csv(
                    output_path,
                    sep=output_config.delimiter,
                    encoding=output_config.encoding,
                    index=False,  # Don't include row indices
                    header=output_config.include_headers,
                    na_rep='',  # Represent NaN as empty string
                    quoting=1,  # Quote all non-numeric values
                    escapechar='\\',  # Escape character for quotes in data
                    lineterminator='\n',  # Use consistent line endings
                    float_format='%.10g',  # Avoid scientific notation for small numbers
                )
                
                # Get file size after writing
                file_size = output_path.stat().st_size
                metrics.add_metadata("file_size_bytes", file_size)
                metrics.add_metadata("write_success", True)
                
                self.logger.debug(
                    f"Successfully wrote CSV file: {output_path} ({file_size:,} bytes)",
                    extra={
                        "structured": {
                            "operation": "csv_file_write_success",
                            "output_path": str(output_path),
                            "file_size_bytes": file_size,
                            "rows_written": len(df),
                            "columns_written": len(df.columns),
                            "delimiter": output_config.delimiter,
                            "encoding": output_config.encoding
                        }
                    }
                )
                
            except Exception as e:
                error_type = type(e).__name__
                metrics.add_metadata("write_success", False)
                metrics.add_metadata("error_type", error_type)
                
                self.logger.error(
                    f"Failed to write CSV file {output_path}: {e}",
                    extra={
                        "structured": {
                            "operation": "csv_file_write_failed",
                            "output_path": str(output_path),
                            "error_type": error_type,
                            "error_message": str(e),
                            "dataframe_shape": df.shape
                        }
                    }
                )
                raise CSVGenerationError(f"Failed to write CSV file {output_path}: {e}") from e
    
    def validate_output_path(self, output_path: Path) -> bool:
        """Validate that output path is writable.
        
        Args:
            output_path: Path to validate
            
        Returns:
            True if path is writable
        """
        try:
            # Check if directory exists or can be created
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions by creating a temporary file
            test_file = output_path.parent / f".test_write_{os.getpid()}"
            try:
                test_file.write_text("test", encoding='utf-8')
                test_file.unlink()  # Delete test file
                return True
            except (OSError, PermissionError):
                return False
                
        except (OSError, PermissionError):
            return False
    
    def estimate_csv_size(self, worksheet_data: WorksheetData) -> int:
        """Estimate the size of the generated CSV file.
        
        Args:
            worksheet_data: Worksheet data to estimate
            
        Returns:
            Estimated file size in bytes
        """
        df = worksheet_data.data
        
        # Rough estimation based on data size and formatting
        # Average bytes per cell (including delimiters and line endings)
        avg_bytes_per_cell = 10
        
        total_cells = df.size
        estimated_size = total_cells * avg_bytes_per_cell
        
        # Add overhead for headers if included
        if len(df.columns) > 0:
            header_size = len(df.columns) * 20  # Average header length
            estimated_size += header_size
        
        return estimated_size
    
    def generate_csv_preview(
        self, 
        worksheet_data: WorksheetData, 
        output_config: OutputConfig,
        max_rows: int = 10
    ) -> str:
        """Generate a preview of the CSV output.
        
        Args:
            worksheet_data: Worksheet data
            output_config: Output configuration
            max_rows: Maximum rows to include in preview
            
        Returns:
            CSV preview as string
        """
        # Prepare data (limited to max_rows)
        preview_data = worksheet_data.data.head(max_rows).copy()
        
        # Create temporary worksheet data for preview
        preview_worksheet = WorksheetData(
            source_file=worksheet_data.source_file,
            worksheet_name=worksheet_data.worksheet_name,
            data=preview_data,
            metadata=worksheet_data.metadata.copy(),
            confidence_score=worksheet_data.confidence_score
        )
        
        # Prepare CSV data
        csv_data = self._prepare_csv_data(preview_worksheet, output_config)
        
        # Convert to CSV string
        csv_string = csv_data.to_csv(
            sep=output_config.delimiter,
            index=False,
            header=output_config.include_headers,
            na_rep='',
            encoding=output_config.encoding
        )
        
        return csv_string