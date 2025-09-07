"""Core data models for Excel-to-CSV converter.

This module contains all the dataclasses and type definitions used throughout
the application for structured data representation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class HeaderInfo:
    """Information about worksheet headers.
    
    Attributes:
        has_headers: Whether the worksheet contains headers
        header_row: Row index of headers (0-based), None if no headers
        header_quality: Quality score of headers (0.0 to 1.0)
        column_names: List of column header names
    """
    has_headers: bool
    header_row: Optional[int]
    header_quality: float
    column_names: List[str]
    
    def __post_init__(self) -> None:
        """Validate header info after initialization."""
        if not 0.0 <= self.header_quality <= 1.0:
            raise ValueError("header_quality must be between 0.0 and 1.0")
        
        if self.has_headers and self.header_row is None:
            raise ValueError("header_row cannot be None when has_headers is True")
        
        if self.header_row is not None and self.header_row < 0:
            raise ValueError("header_row must be non-negative")


@dataclass
class ConfidenceScore:
    """Confidence score for worksheet data table detection.
    
    Attributes:
        overall_score: Overall confidence score (0.0 to 1.0)
        data_density: Data density component score (0.0 to 1.0)
        header_quality: Header quality component score (0.0 to 1.0)
        consistency_score: Data consistency component score (0.0 to 1.0)
        reasons: List of reasons explaining the score
    """

    overall_score: float = 0.0
    data_density: float = 0.0
    header_quality: float = 0.0
    consistency_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    threshold: float = 0.8
    
    def __post_init__(self) -> None:
        """Validate confidence score after initialization."""
        scores = [
            ("overall_score", self.overall_score),
            ("data_density", self.data_density),
            ("header_quality", self.header_quality),
            ("consistency_score", self.consistency_score)
        ]
        
        for name, score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
    
    @property
    def is_confident(self) -> bool:
        """Check if overall score indicates confidence."""
        return self.overall_score >= self.threshold
    
    def add_reason(self, reason: str) -> None:
        """Add a reason for the confidence score."""
        if reason and reason not in self.reasons:
            self.reasons.append(reason)


@dataclass
class WorksheetData:
    """Data container for Excel worksheet information.
    
    Attributes:
        source_file: Path to the source Excel file
        worksheet_name: Name of the worksheet
        data: DataFrame containing the worksheet data
        metadata: Additional metadata about the worksheet
        confidence_score: Optional confidence score for the worksheet
        archive_result: Optional result of archiving operation
        archived_at: Timestamp when the source file was archived
    """
    source_file: Path
    worksheet_name: str
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: Optional[ConfidenceScore] = None
    archive_result: Optional[ArchiveResult] = None
    archived_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Validate worksheet data after initialization."""
        if not isinstance(self.source_file, Path):
            self.source_file = Path(self.source_file)
        
        if not self.worksheet_name.strip():
            raise ValueError("worksheet_name cannot be empty")
        
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
    
    @property
    def row_count(self) -> int:
        """Number of rows in the worksheet data."""
        return len(self.data)
    
    @property
    def column_count(self) -> int:
        """Number of columns in the worksheet data."""
        return len(self.data.columns)
    
    @property
    def is_empty(self) -> bool:
        """Check if the worksheet data is empty."""
        return self.data.empty
    
    @property
    def non_empty_cell_count(self) -> int:
        """Count of non-empty cells in the worksheet."""
        return self.data.count().sum()
    
    @property
    def total_cell_count(self) -> int:
        """Total number of cells in the worksheet."""
        return self.row_count * self.column_count
    
    @property
    def data_density(self) -> float:
        """Calculate data density (non-empty cells / total cells)."""
        if self.total_cell_count == 0:
            return 0.0
        return self.non_empty_cell_count / self.total_cell_count


@dataclass
class OutputConfig:
    """Configuration for CSV output generation.
    
    Attributes:
        folder: Output directory path (None for same as source)
        naming_pattern: Pattern for output filename generation
        encoding: Character encoding for CSV files
        include_timestamp: Whether to include timestamp for duplicates
        delimiter: CSV field delimiter
        include_headers: Whether to include headers in CSV output
        timestamp_format: Format string for timestamps
    """
    folder: Optional[Path] = None
    naming_pattern: str = "{filename}_{worksheet}.csv"
    encoding: str = "utf-8"
    include_timestamp: bool = True
    delimiter: str = ","
    include_headers: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    def __post_init__(self) -> None:
        """Validate output configuration after initialization."""
        if self.folder is not None and not isinstance(self.folder, Path):
            self.folder = Path(self.folder)
        
        if not self.naming_pattern.strip():
            raise ValueError("naming_pattern cannot be empty")
        
        if not self.encoding.strip():
            raise ValueError("encoding cannot be empty")
        
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must be a single character")
    
    def generate_filename(
        self, 
        source_filename: str, 
        worksheet_name: str,
        timestamp: Optional[str] = None
    ) -> str:
        """Generate output filename based on pattern.
        
        Args:
            source_filename: Name of source Excel file (without extension)
            worksheet_name: Name of the worksheet
            timestamp: Optional timestamp string
            
        Returns:
            Generated filename
        """
        # Clean worksheet name for filesystem safety
        clean_worksheet = "".join(
            c for c in worksheet_name if c.isalnum() or c in "._- "
        ).strip()
        
        filename = self.naming_pattern.format(
            filename=source_filename,
            worksheet=clean_worksheet,
            timestamp=timestamp or ""
        )
        
        # Add timestamp if requested and provided
        if self.include_timestamp and timestamp:
            base, ext = filename.rsplit(".", 1) if "." in filename else (filename, "csv")
            filename = f"{base}_{timestamp}.{ext}"
        
        return filename


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Exponential backoff multiplier
        max_delay: Maximum delay between retries in seconds
    """
    max_attempts: int = 3
    delay: float = 5.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    
    def __post_init__(self) -> None:
        """Validate retry configuration after initialization."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        
        if self.delay <= 0:
            raise ValueError("delay must be positive")
        
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be at least 1")
        
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.
        
        Args:
            attempt: Attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if attempt <= 0:
            return self.delay
        
        delay = self.delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


@dataclass
class LoggingConfig:
    """Configuration for logging behavior.
    
    Attributes:
        level: Logging level
        format: Log message format string
        file_enabled: Whether to log to file
        file_path: Path for log file
        console_enabled: Whether to log to console
        structured_enabled: Whether to use structured JSON logging
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: Path = Path("./logs/excel_to_csv.log")
    console_enabled: bool = True
    structured_enabled: bool = False
    
    def __post_init__(self) -> None:
        """Validate logging configuration after initialization."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        
        self.level = self.level.upper()
        
        if not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
    
    @property
    def log_level(self) -> int:
        """Get numeric logging level."""
        return getattr(logging, self.level)


class ArchiveError(Exception):
    """Exception raised for archiving-related errors.
    
    This exception is raised when file archiving operations fail,
    such as permission errors, disk space issues, or file system problems.
    
    Attributes:
        message: Error message describing what went wrong
        file_path: Path to the file that caused the error (if applicable)
        error_type: Category of error (permission, filesystem, configuration)
    """
    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[Path] = None,
        error_type: str = "general"
    ):
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.error_type = error_type
    
    def __str__(self) -> str:
        if self.file_path:
            return f"ArchiveError[{self.error_type}]: {self.message} (File: {self.file_path})"
        return f"ArchiveError[{self.error_type}]: {self.message}"


@dataclass 
class ArchiveConfig:
    """Configuration for file archiving behavior.
    
    Attributes:
        enabled: Whether archiving is enabled
        archive_folder_name: Name of archive subfolder to create
        timestamp_format: Format string for conflict resolution timestamps
        handle_conflicts: Whether to handle filename conflicts with timestamps
        preserve_structure: Whether to maintain directory structure in archives
    """
    enabled: bool = True
    archive_folder_name: str = "archive"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    handle_conflicts: bool = True
    preserve_structure: bool = True
    
    def __post_init__(self) -> None:
        """Validate archive configuration after initialization."""
        if not self.archive_folder_name.strip():
            raise ValueError("archive_folder_name cannot be empty")
        
        if not self.timestamp_format.strip():
            raise ValueError("timestamp_format cannot be empty")
        
        # Test timestamp format validity
        try:
            datetime.now().strftime(self.timestamp_format)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timestamp_format: {e}")


@dataclass
class ArchiveResult:
    """Result container for archive operations.
    
    Attributes:
        success: Whether the archive operation succeeded
        source_path: Original path of the file
        archive_path: Path where the file was archived (if successful)
        timestamp_used: Timestamp suffix used for conflict resolution
        error_message: Error message if operation failed
        operation_time: Time taken for the archive operation in seconds
    """
    success: bool
    source_path: Path
    archive_path: Optional[Path] = None
    timestamp_used: Optional[str] = None
    error_message: Optional[str] = None
    operation_time: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate archive result after initialization."""
        if not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)
        
        if self.archive_path is not None and not isinstance(self.archive_path, Path):
            self.archive_path = Path(self.archive_path)
        
        if self.operation_time < 0:
            raise ValueError("operation_time cannot be negative")
    
    def was_successful(self) -> bool:
        """Check if the archive operation was successful."""
        return self.success
    
    def get_error_details(self) -> Dict[str, Any]:
        """Get detailed error information.
        
        Returns:
            Dictionary containing error details
        """
        return {
            "success": self.success,
            "source_path": str(self.source_path),
            "archive_path": str(self.archive_path) if self.archive_path else None,
            "error_message": self.error_message,
            "timestamp_used": self.timestamp_used,
            "operation_time": self.operation_time
        }


@dataclass
class Config:
    """Main configuration for Excel-to-CSV converter.
    
    Attributes:
        monitored_folders: List of directories to monitor
        confidence_threshold: Minimum confidence threshold for processing
        output_folder: Optional output directory path
        file_patterns: List of file patterns to match
        logging: Logging configuration
        retry_settings: Retry configuration
        output_config: Output generation configuration
        archive_config: File archiving configuration
        max_concurrent: Maximum concurrent processing operations
        max_file_size_mb: Maximum file size in MB to process
    """
    monitored_folders: List[Path]
    confidence_threshold: float = 0.9
    output_folder: Optional[Path] = None
    file_patterns: List[str] = field(default_factory=lambda: ["*.xlsx", "*.xls"])
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    retry_settings: RetryConfig = field(default_factory=RetryConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    archive_config: ArchiveConfig = field(default_factory=ArchiveConfig)
    max_concurrent: int = 5
    max_file_size_mb: int = 100
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert string paths to Path objects
        self.monitored_folders = [
            Path(folder) if not isinstance(folder, Path) else folder
            for folder in self.monitored_folders
        ]
        
        if self.output_folder is not None and not isinstance(self.output_folder, Path):
            self.output_folder = Path(self.output_folder)
        
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        # Validate file patterns
        if not self.file_patterns:
            raise ValueError("file_patterns cannot be empty")
        
        # Validate max_concurrent
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        
        # Validate max_file_size_mb
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    def add_monitored_folder(self, folder: Union[str, Path]) -> None:
        """Add a folder to monitor.
        
        Args:
            folder: Path to folder to monitor
        """
        folder_path = Path(folder) if not isinstance(folder, Path) else folder
        if folder_path not in self.monitored_folders:
            self.monitored_folders.append(folder_path)
    
    def remove_monitored_folder(self, folder: Union[str, Path]) -> bool:
        """Remove a folder from monitoring.
        
        Args:
            folder: Path to folder to remove
            
        Returns:
            True if folder was removed, False if not found
        """
        folder_path = Path(folder) if not isinstance(folder, Path) else folder
        try:
            self.monitored_folders.remove(folder_path)
            return True
        except ValueError:
            return False