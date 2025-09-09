"""Tests for enhanced logging configuration."""

import gzip
import logging
import os
import tempfile
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from excel_to_csv.utils.correlation import CorrelationContext
from excel_to_csv.utils.logging_config import (
    CorrelationFormatter,
    DailyRotatingLogHandler,
    setup_enhanced_logging
)


class TestCorrelationFormatter:
    """Test cases for CorrelationFormatter."""
    
    def test_format_with_correlation_id(self):
        """Test formatting with correlation ID."""
        formatter = CorrelationFormatter("%(correlation_id)s - %(message)s")
        
        # Set correlation ID
        CorrelationContext.set_correlation_id("test-id-123")
        
        # Create log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "test-id-123 - Test message" == formatted
    
    def test_format_without_correlation_id(self):
        """Test formatting without correlation ID."""
        formatter = CorrelationFormatter("%(correlation_id)s - %(message)s")
        
        # Save current context and clear correlation ID
        original_context = CorrelationContext.get_correlation_id()
        
        # Create a new context with no correlation ID
        with patch.object(CorrelationContext, 'get_correlation_id', return_value=None):
            # Create log record
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="test.py", 
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            assert "NONE - Test message" == formatted
    
    def test_format_with_structured_data(self):
        """Test formatting with structured data."""
        formatter = CorrelationFormatter("%(message)s")
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add structured data
        record.structured = {"operation": "test", "status": "SUCCESS"}
        
        formatted = formatter.format(record)
        assert formatted == "Test message"
        assert hasattr(record, 'structured')


class TestDailyRotatingLogHandler:
    """Test cases for DailyRotatingLogHandler."""
    
    def test_init(self):
        """Test handler initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            handler = DailyRotatingLogHandler(log_dir)
            
            assert handler.log_dir == log_dir
            assert handler.base_filename == "excel_to_csv"
            assert handler.retention_days == 30
            assert (log_dir / "archive").exists()
    
    def test_init_custom_params(self):
        """Test handler initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            handler = DailyRotatingLogHandler(
                log_dir, 
                base_filename="custom_log",
                retention_days=7
            )
            
            assert handler.base_filename == "custom_log"
            assert handler.retention_days == 7
    
    def test_get_current_handler_creates_new(self):
        """Test getting current handler creates new file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            handler = DailyRotatingLogHandler(log_dir)
            
            file_handler = handler.get_current_handler()
            
            assert isinstance(file_handler, logging.FileHandler)
            assert handler.current_date == datetime.now().date()
            
            # Check log file was created
            expected_filename = f"excel_to_csv_{datetime.now().strftime('%Y%m%d')}.log"
            expected_path = log_dir / expected_filename
            assert expected_path.exists()
    
    def test_rotation_on_date_change(self):
        """Test log rotation when date changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            handler = DailyRotatingLogHandler(log_dir)
            
            # Get handler for "today"
            today = date(2024, 1, 10)
            with patch('excel_to_csv.utils.logging_config.datetime') as mock_dt:
                mock_dt.now.return_value = datetime.combine(today, datetime.min.time())
                handler._rotate_logs(today)
            
            first_handler = handler.current_handler
            assert handler.current_date == today
            
            # Simulate date change
            tomorrow = date(2024, 1, 11)
            with patch('excel_to_csv.utils.logging_config.datetime') as mock_dt:
                mock_dt.now.return_value = datetime.combine(tomorrow, datetime.min.time())
                handler._rotate_logs(tomorrow)
            
            second_handler = handler.current_handler
            assert handler.current_date == tomorrow
            assert first_handler != second_handler
    
    def test_archive_current_log(self):
        """Test archiving of current log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            handler = DailyRotatingLogHandler(log_dir)
            
            # Create a log file with content
            log_file = log_dir / "excel_to_csv_20240110.log"
            log_file.write_text("Test log content\n")
            
            # Create file handler for the log file
            handler.current_handler = logging.FileHandler(log_file)
            
            # Archive the log
            handler._archive_current_log()
            
            # Check original file is gone
            assert not log_file.exists()
            
            # Check archive was created
            archive_file = log_dir / "archive" / "excel_to_csv_20240110.gz"
            assert archive_file.exists()
            
            # Check archive content
            with gzip.open(archive_file, 'rt') as f:
                content = f.read()
                assert content == "Test log content\n"
    
    def test_archive_empty_log_file(self):
        """Test archiving empty log file does nothing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            handler = DailyRotatingLogHandler(log_dir)
            
            # Create empty log file
            log_file = log_dir / "excel_to_csv_20240110.log"
            log_file.touch()
            
            handler.current_handler = logging.FileHandler(log_file)
            
            # Archive should do nothing
            handler._archive_current_log()
            
            # Empty file should still exist (nothing to archive)
            assert log_file.exists()
            
            # No archive should be created
            archive_file = log_dir / "archive" / "excel_to_csv_20240110.gz"
            assert not archive_file.exists()
    
    def test_cleanup_old_archives(self):
        """Test cleanup of old archived logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            archive_dir = log_dir / "archive"
            archive_dir.mkdir()
            
            handler = DailyRotatingLogHandler(log_dir, retention_days=7)
            
            # Create old and new archive files
            old_archive = archive_dir / "excel_to_csv_20240101.gz"
            recent_archive = archive_dir / "excel_to_csv_20240110.gz"
            
            old_archive.write_text("old content")
            recent_archive.write_text("recent content")
            
            # Set modification times
            old_time = time.time() - (10 * 24 * 60 * 60)  # 10 days ago
            recent_time = time.time() - (3 * 24 * 60 * 60)  # 3 days ago
            
            os.utime(old_archive, (old_time, old_time))
            os.utime(recent_archive, (recent_time, recent_time))
            
            # Cleanup old archives
            handler._cleanup_old_archives()
            
            # Old archive should be deleted, recent should remain
            assert not old_archive.exists()
            assert recent_archive.exists()
    
    def test_cleanup_disabled_with_zero_retention(self):
        """Test cleanup is disabled when retention_days is 0."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            archive_dir = log_dir / "archive"
            archive_dir.mkdir()
            
            handler = DailyRotatingLogHandler(log_dir, retention_days=0)
            
            # Create old archive file
            old_archive = archive_dir / "excel_to_csv_20240101.gz"
            old_archive.write_text("old content")
            
            old_time = time.time() - (365 * 24 * 60 * 60)  # 1 year ago
            os.utime(old_archive, (old_time, old_time))
            
            # Cleanup should do nothing
            handler._cleanup_old_archives()
            
            # File should still exist
            assert old_archive.exists()


class TestSetupEnhancedLogging:
    """Test cases for setup_enhanced_logging function."""
    
    def test_setup_basic_logging(self):
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            logger = setup_enhanced_logging(
                log_level="INFO",
                log_dir=log_dir,
                daily_rotation=False,
                console_output=False
            )
            
            assert logger.level == logging.INFO
            assert len(logger.handlers) == 1
            
            # Check log file was created
            log_file = log_dir / "excel_to_csv.log"
            assert log_file.exists()
    
    def test_setup_with_daily_rotation(self):
        """Test logging setup with daily rotation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            logger = setup_enhanced_logging(
                log_level="DEBUG",
                log_dir=log_dir,
                daily_rotation=True,
                console_output=False
            )
            
            assert logger.level == logging.DEBUG
            
            # Check today's log file was created
            today_filename = f"excel_to_csv_{datetime.now().strftime('%Y%m%d')}.log"
            today_log = log_dir / today_filename
            assert today_log.exists()
    
    def test_setup_with_console_output(self):
        """Test logging setup with console output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            logger = setup_enhanced_logging(
                log_level="WARNING",
                log_dir=log_dir,
                console_output=True
            )
            
            # Should have both file and console handlers
            assert len(logger.handlers) == 2
            
            handler_types = [type(handler).__name__ for handler in logger.handlers]
            assert "StreamHandler" in handler_types
            assert "FileHandler" in handler_types
    
    def test_setup_clears_existing_handlers(self):
        """Test that setup clears existing handlers."""
        # Add some dummy handlers
        root_logger = logging.getLogger()
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)
        
        initial_handler_count = len(root_logger.handlers)
        assert initial_handler_count >= 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            logger = setup_enhanced_logging(
                log_dir=log_dir,
                console_output=False
            )
            
            # Should only have the new handler
            assert len(logger.handlers) == 1
    
    def test_correlation_id_in_logs(self):
        """Test that correlation ID appears in log output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            setup_enhanced_logging(
                log_dir=log_dir,
                daily_rotation=False,
                console_output=False
            )
            
            # Set correlation ID and log message
            CorrelationContext.set_correlation_id("test-correlation-456")
            logger = logging.getLogger("test")
            logger.info("Test message with correlation")
            
            # Check log file content
            log_file = log_dir / "excel_to_csv.log"
            content = log_file.read_text()
            
            assert "test-correlation-456" in content
            assert "Test message with correlation" in content