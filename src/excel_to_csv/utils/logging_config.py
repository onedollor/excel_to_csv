"""Enhanced logging configuration with correlation tracking and daily rotation.

This module provides comprehensive logging configuration including:
- Correlation ID injection into all log records
- Daily log rotation with automatic archival
- Structured logging format support
- Configurable retention policies
"""

import gzip
import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

from .correlation import CorrelationContext


class CorrelationFormatter(logging.Formatter):
    """Custom formatter that injects correlation IDs into log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with correlation ID injection.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message string
        """
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = "NONE"
        
        # Add structured data if present
        if hasattr(record, 'structured') and isinstance(record.structured, dict):
            # For JSON output, we could serialize the structured data
            # For now, we'll just ensure it's available
            pass
        
        return super().format(record)


class DailyRotatingLogHandler:
    """Handler for daily log rotation with automatic archival and compression."""
    
    def __init__(
        self, 
        log_dir: Path, 
        base_filename: str = "excel_to_csv",
        retention_days: int = 30
    ):
        """Initialize daily rotating log handler.
        
        Args:
            log_dir: Directory to store log files
            base_filename: Base name for log files (without extension)
            retention_days: Number of days to retain archived logs
        """
        self.log_dir = Path(log_dir)
        self.base_filename = base_filename
        self.retention_days = retention_days
        self.current_date = None
        self.current_handler = None
        self._setup_log_directory()
    
    def _setup_log_directory(self) -> None:
        """Create log directory structure if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "archive").mkdir(exist_ok=True)
    
    def get_current_handler(self) -> logging.Handler:
        """Get handler for current date, rotating if needed.
        
        Returns:
            Current file handler for today's log
        """
        today = datetime.now().date()
        
        if self.current_date != today:
            self._rotate_logs(today)
        
        return self.current_handler
    
    def _rotate_logs(self, new_date: date) -> None:
        """Rotate to new log file and archive previous.
        
        Args:
            new_date: The new date to rotate to
        """
        # Archive current log if it exists
        if self.current_handler:
            self._archive_current_log()
            self._close_current_handler()
        
        # Clean up old archives
        self._cleanup_old_archives()
        
        # Create new log file for today
        log_filename = f"{self.base_filename}_{new_date.strftime('%Y%m%d')}.log"
        log_path = self.log_dir / log_filename
        
        self.current_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        self.current_date = new_date
    
    def _close_current_handler(self) -> None:
        """Safely close current handler."""
        if self.current_handler:
            # Remove from any loggers first to prevent writing during close
            root_logger = logging.getLogger()
            if self.current_handler in root_logger.handlers:
                root_logger.removeHandler(self.current_handler)
            
            self.current_handler.close()
    
    def _archive_current_log(self) -> None:
        """Archive current log file with compression."""
        if not self.current_handler:
            return
        
        current_log_path = Path(self.current_handler.baseFilename)
        if not current_log_path.exists() or current_log_path.stat().st_size == 0:
            # No log file or empty file, nothing to archive
            return
        
        try:
            # Create archive filename
            archive_filename = f"{current_log_path.stem}.gz"
            archive_path = self.log_dir / "archive" / archive_filename
            
            # Compress and move to archive
            with open(current_log_path, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original log file after successful compression
            current_log_path.unlink()
            
        except Exception as e:
            # If archival fails, log to stderr but continue
            print(f"Warning: Failed to archive log file {current_log_path}: {e}", 
                  file=__import__('sys').stderr)
    
    def _cleanup_old_archives(self) -> None:
        """Remove archived logs older than retention period."""
        if self.retention_days <= 0:
            return
        
        archive_dir = self.log_dir / "archive"
        if not archive_dir.exists():
            return
        
        cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
        
        try:
            for archive_file in archive_dir.glob(f"{self.base_filename}_*.gz"):
                if archive_file.stat().st_mtime < cutoff_time:
                    archive_file.unlink()
        except Exception as e:
            # If cleanup fails, log to stderr but continue
            print(f"Warning: Failed to cleanup old archives: {e}", 
                  file=__import__('sys').stderr)


# Standard log format with correlation ID
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"

# JSON-compatible log format (if needed)
JSON_LOG_FORMAT = {
    "timestamp": "%(asctime)s",
    "level": "%(levelname)s", 
    "correlation_id": "%(correlation_id)s",
    "logger": "%(name)s",
    "function": "%(funcName)s",
    "line": "%(lineno)d",
    "message": "%(message)s"
}


def setup_enhanced_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    structured_format: bool = True,
    daily_rotation: bool = True,
    retention_days: int = 30,
    console_output: bool = True
) -> logging.Logger:
    """Setup enhanced logging with daily rotation and archival.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (defaults to ./logs)
        structured_format: Whether to use structured format with correlation IDs
        daily_rotation: Whether to enable daily log rotation
        retention_days: Number of days to retain archived logs
        console_output: Whether to also log to console
        
    Returns:
        Configured root logger instance
    """
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Setup file handler
    if daily_rotation:
        rotating_handler_manager = DailyRotatingLogHandler(
            log_dir, retention_days=retention_days
        )
        file_handler = rotating_handler_manager.get_current_handler()
    else:
        log_file = log_dir / "excel_to_csv.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    # Setup formatter
    if structured_format:
        formatter = CorrelationFormatter(LOG_FORMAT)
    else:
        formatter = logging.Formatter(LOG_FORMAT.replace(" | %(correlation_id)s", ""))
    
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    return root_logger


def get_processing_logger(name: str) -> logging.Logger:
    """Get a logger for processing operations.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Convenience function for backward compatibility
setup_logging = setup_enhanced_logging