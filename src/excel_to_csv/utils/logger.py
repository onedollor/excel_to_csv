"""Logging utilities for Excel-to-CSV converter.

This module provides comprehensive logging setup with support for:
- Structured JSON logging
- File and console handlers with rotation
- Domain-specific logging methods for processing events
- Performance and monitoring logging
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from excel_to_csv.models.data_models import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add processing context if present
        for field in ["file_path", "worksheet_name", "confidence_score", 
                     "processing_time", "error_type"]:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ProcessingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for processing-specific logging.
    
    Adds processing context to log records for better traceability.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """Initialize processing logger adapter.
        
        Args:
            logger: Base logger instance
            extra: Extra context to add to all log records
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and kwargs.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (message, kwargs)
        """
        # Add extra context to the log record
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        kwargs["extra"].update(self.extra)
        return msg, kwargs
    
    def log_processing_start(
        self, 
        file_path: Union[str, Path], 
        file_size: Optional[int] = None
    ) -> None:
        """Log processing start event.
        
        Args:
            file_path: Path to file being processed
            file_size: File size in bytes
        """
        extra = {
            "event_type": "processing_start",
            "file_path": str(file_path),
            "file_size": file_size,
        }
        self.info(f"Started processing file: {file_path}", extra=extra)
    
    def log_processing_complete(
        self, 
        file_path: Union[str, Path], 
        worksheet_count: int,
        processing_time: float,
        csv_files_generated: int
    ) -> None:
        """Log processing completion event.
        
        Args:
            file_path: Path to processed file
            worksheet_count: Number of worksheets processed
            processing_time: Processing time in seconds
            csv_files_generated: Number of CSV files generated
        """
        extra = {
            "event_type": "processing_complete",
            "file_path": str(file_path),
            "worksheet_count": worksheet_count,
            "processing_time": processing_time,
            "csv_files_generated": csv_files_generated,
        }
        self.info(
            f"Completed processing {file_path}: "
            f"{csv_files_generated}/{worksheet_count} worksheets converted "
            f"in {processing_time:.2f}s",
            extra=extra
        )
    
    def log_worksheet_analysis(
        self, 
        worksheet_name: str, 
        confidence_score: float,
        decision: bool,
        reasons: Optional[list] = None
    ) -> None:
        """Log worksheet confidence analysis.
        
        Args:
            worksheet_name: Name of the worksheet
            confidence_score: Calculated confidence score
            decision: Whether worksheet was accepted for processing
            reasons: List of reasons for the decision
        """
        extra = {
            "event_type": "worksheet_analysis",
            "worksheet_name": worksheet_name,
            "confidence_score": confidence_score,
            "decision": decision,
            "reasons": reasons or [],
        }
        
        action = "accepted" if decision else "rejected"
        self.info(
            f"Worksheet '{worksheet_name}' {action} "
            f"(confidence: {confidence_score:.3f})",
            extra=extra
        )
    
    def log_csv_generation(
        self, 
        output_path: Union[str, Path], 
        record_count: int,
        file_size: Optional[int] = None
    ) -> None:
        """Log CSV file generation.
        
        Args:
            output_path: Path to generated CSV file
            record_count: Number of records in CSV
            file_size: Generated file size in bytes
        """
        extra = {
            "event_type": "csv_generation",
            "output_path": str(output_path),
            "record_count": record_count,
            "file_size": file_size,
        }
        self.info(
            f"Generated CSV: {output_path} ({record_count} records)",
            extra=extra
        )
    
    def log_error(
        self, 
        error_type: str, 
        message: str, 
        file_path: Optional[Union[str, Path]] = None,
        worksheet_name: Optional[str] = None,
        exc_info: bool = True
    ) -> None:
        """Log processing error with context.
        
        Args:
            error_type: Type of error
            message: Error message
            file_path: Optional file path where error occurred
            worksheet_name: Optional worksheet name where error occurred
            exc_info: Whether to include exception information
        """
        extra = {
            "event_type": "processing_error",
            "error_type": error_type,
        }
        
        if file_path:
            extra["file_path"] = str(file_path)
        if worksheet_name:
            extra["worksheet_name"] = worksheet_name
        
        self.error(message, extra=extra, exc_info=exc_info)
    
    def log_performance_warning(
        self, 
        metric: str, 
        value: Union[int, float], 
        threshold: Union[int, float],
        file_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Log performance warning.
        
        Args:
            metric: Performance metric name
            value: Current value
            threshold: Warning threshold
            file_path: Optional file path related to warning
        """
        extra = {
            "event_type": "performance_warning",
            "metric": metric,
            "value": value,
            "threshold": threshold,
        }
        
        if file_path:
            extra["file_path"] = str(file_path)
        
        self.warning(
            f"Performance warning: {metric}={value} exceeds threshold {threshold}",
            extra=extra
        )


class LoggerManager:
    """Manages logger setup and configuration.
    
    Provides centralized logger configuration with support for:
    - Multiple output handlers (file, console, structured)
    - Log rotation and retention
    - Performance monitoring
    - Processing event tracking
    """
    
    def __init__(self):
        """Initialize logger manager."""
        self._configured = False
        self._loggers: Dict[str, logging.Logger] = {}
        self._adapters: Dict[str, ProcessingLoggerAdapter] = {}
    
    def setup_logging(self, config: LoggingConfig) -> None:
        """Set up logging configuration.
        
        Args:
            config: Logging configuration
        """
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logging level
        root_logger.setLevel(config.log_level)
        
        # Set up console handler
        if config.console_enabled:
            self._setup_console_handler(config)
        
        # Set up file handler
        if config.file_enabled:
            self._setup_file_handler(config)
        
        # Set up structured logging
        if config.structured_enabled:
            self._setup_structured_handler(config)
        
        # Configure third-party loggers
        self._configure_third_party_loggers()
        
        self._configured = True
        
        # Log configuration completion
        logger = self.get_logger(__name__)
        logger.info("Logging configuration completed")
        logger.debug(f"Log level: {config.level}")
        logger.debug(f"Console enabled: {config.console_enabled}")
        logger.debug(f"File logging enabled: {config.file_enabled}")
        logger.debug(f"Structured logging enabled: {config.structured_enabled}")
    
    def _setup_console_handler(self, config: LoggingConfig) -> None:
        """Set up console logging handler.
        
        Args:
            config: Logging configuration
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.log_level)
        
        formatter = logging.Formatter(
            fmt=config.format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(console_handler)
    
    def _setup_file_handler(self, config: LoggingConfig) -> None:
        """Set up file logging handler with rotation.
        
        Args:
            config: Logging configuration
        """
        # Ensure log directory exists
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(config.log_level)
        
        formatter = logging.Formatter(
            fmt=config.format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
    
    def _setup_structured_handler(self, config: LoggingConfig) -> None:
        """Set up structured JSON logging handler.
        
        Args:
            config: Logging configuration
        """
        # Ensure log directory exists
        structured_path = config.file_path.parent / "structured.json"
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create structured logging handler
        structured_handler = logging.handlers.RotatingFileHandler(
            filename=structured_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        structured_handler.setLevel(config.log_level)
        structured_handler.setFormatter(JSONFormatter())
        
        logging.getLogger().addHandler(structured_handler)
    
    def _configure_third_party_loggers(self) -> None:
        """Configure third-party library loggers."""
        # Reduce verbosity of third-party libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("pandas").setLevel(logging.WARNING)
        logging.getLogger("openpyxl").setLevel(logging.WARNING)
        logging.getLogger("watchdog").setLevel(logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance by name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        
        return self._loggers[name]
    
    def get_processing_logger(
        self, 
        name: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessingLoggerAdapter:
        """Get processing logger adapter with context.
        
        Args:
            name: Logger name
            context: Additional context for all log records
            
        Returns:
            Processing logger adapter
        """
        cache_key = f"{name}:{hash(str(context))}"
        
        if cache_key not in self._adapters:
            base_logger = self.get_logger(name)
            self._adapters[cache_key] = ProcessingLoggerAdapter(base_logger, context)
        
        return self._adapters[cache_key]
    
    def shutdown(self) -> None:
        """Shutdown logging system gracefully."""
        logging.shutdown()
        self._configured = False
        self._loggers.clear()
        self._adapters.clear()


# Global logger manager instance
logger_manager = LoggerManager()


def setup_logging(config: LoggingConfig) -> None:
    """Set up application logging.
    
    Args:
        config: Logging configuration
    """
    logger_manager.setup_logging(config)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger_manager.get_logger(name)


def get_processing_logger(
    name: str, 
    context: Optional[Dict[str, Any]] = None
) -> ProcessingLoggerAdapter:
    """Get processing logger with context.
    
    Args:
        name: Logger name (typically __name__)
        context: Additional context for log records
        
    Returns:
        Processing logger adapter
    """
    return logger_manager.get_processing_logger(name, context)


def shutdown_logging() -> None:
    """Shutdown logging system."""
    logger_manager.shutdown()