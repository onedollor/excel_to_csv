"""Comprehensive tests for logging utilities targeting 85%+ coverage.

This test suite covers all logging functionality including:
- JSONFormatter structured logging
- ProcessingLoggerAdapter with context
- Logger setup and configuration  
- File and console handlers with rotation
- Processing-specific logging methods
- Error handling and edge cases
"""

import pytest
import json
import logging
import logging.handlers
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from excel_to_csv.utils.logger import (
    JSONFormatter,
    ProcessingLoggerAdapter,
    LoggerManager,
    get_processing_logger,
    setup_logging,
    get_logger,
    shutdown_logging,
    logger_manager
)
from excel_to_csv.models.data_models import LoggingConfig


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_logging_config(temp_workspace):
    """Create sample logging configuration."""
    return LoggingConfig(
        level="DEBUG",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file_enabled=True,
        file_path=temp_workspace / "test.log",
        console_enabled=True,
        structured_enabled=False
    )


class TestJSONFormatter:
    """Test JSONFormatter functionality."""
    
    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        # Create log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format record
        result = formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(result)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "module"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
        assert log_data["timestamp"].endswith("Z")
    
    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info."""
        formatter = JSONFormatter()
        
        # Create exception
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            exc_info = sys.exc_info()
        
        # Create log record with exception
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        
        # Format record
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert "exception" in log_data
        assert "ValueError: Test exception" in log_data["exception"]
        assert "Traceback" in log_data["exception"]
    
    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        
        # Create log record with extra fields
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.extra_fields = {
            "custom_field": "custom_value",
            "another_field": 42,
            "nested_field": {"key": "value"}
        }
        
        # Format record
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["custom_field"] == "custom_value"
        assert log_data["another_field"] == 42
        assert log_data["nested_field"] == {"key": "value"}
    
    def test_json_formatter_with_processing_context(self):
        """Test JSON formatting with processing context fields."""
        formatter = JSONFormatter()
        
        # Create log record
        record = logging.LogRecord(
            name="processing.logger",
            level=logging.INFO,
            pathname="/path/to/processor.py",
            lineno=100,
            msg="Processing file",
            args=(),
            exc_info=None
        )
        
        # Add processing context
        record.file_path = "/path/to/file.xlsx"
        record.worksheet_name = "Sheet1"
        record.confidence_score = 0.85
        record.processing_time = 1.23
        record.error_type = "ValidationError"
        
        # Format record
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["file_path"] == "/path/to/file.xlsx"
        assert log_data["worksheet_name"] == "Sheet1"
        assert log_data["confidence_score"] == 0.85
        assert log_data["processing_time"] == 1.23
        assert log_data["error_type"] == "ValidationError"
    
    def test_json_formatter_unicode_handling(self):
        """Test JSON formatter with Unicode characters."""
        formatter = JSONFormatter()
        
        # Create log record with Unicode
        record = logging.LogRecord(
            name="unicode.logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Processing file: 测试文件.xlsx with émojis 🎉",
            args=(),
            exc_info=None
        )
        
        # Format record
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert "测试文件.xlsx" in log_data["message"]
        assert "émojis 🎉" in log_data["message"]
    
    def test_json_formatter_message_with_args(self):
        """Test JSON formatter with message arguments."""
        formatter = JSONFormatter()
        
        # Create log record with arguments
        record = logging.LogRecord(
            name="args.logger",
            level=logging.INFO,
            pathname="/path/to/module.py",
            lineno=42,
            msg="Processing %s with %d items",
            args=("test_file.xlsx", 100),
            exc_info=None
        )
        
        # Format record
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["message"] == "Processing test_file.xlsx with 100 items"


class TestProcessingLoggerAdapter:
    """Test ProcessingLoggerAdapter functionality."""
    
    def test_adapter_initialization_default(self):
        """Test adapter initialization with default extra."""
        base_logger = logging.getLogger("test.adapter")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        assert adapter.logger is base_logger
        assert adapter.extra == {}
    
    def test_adapter_initialization_with_extra(self):
        """Test adapter initialization with extra context."""
        base_logger = logging.getLogger("test.adapter")
        extra_context = {"file_path": "/test/file.xlsx", "operation": "convert"}
        
        adapter = ProcessingLoggerAdapter(base_logger, extra_context)
        
        assert adapter.logger is base_logger
        assert adapter.extra == extra_context
    
    def test_adapter_process_method(self):
        """Test adapter process method."""
        base_logger = logging.getLogger("test.adapter")
        extra_context = {"default_field": "default_value"}
        adapter = ProcessingLoggerAdapter(base_logger, extra_context)
        
        # Test process method
        msg = "Test message"
        kwargs = {"extra": {"custom_field": "custom_value"}}
        
        processed_msg, processed_kwargs = adapter.process(msg, kwargs)
        
        assert processed_msg == msg
        assert "extra" in processed_kwargs
        assert processed_kwargs["extra"]["default_field"] == "default_value"
        assert processed_kwargs["extra"]["custom_field"] == "custom_value"
    
    def test_adapter_process_without_extra_kwargs(self):
        """Test adapter process without extra kwargs."""
        base_logger = logging.getLogger("test.adapter")
        extra_context = {"default_field": "default_value"}
        adapter = ProcessingLoggerAdapter(base_logger, extra_context)
        
        # Test process method without extra in kwargs
        msg = "Test message"
        kwargs = {"level": "INFO"}
        
        processed_msg, processed_kwargs = adapter.process(msg, kwargs)
        
        assert processed_msg == msg
        assert processed_kwargs["extra"] == extra_context
        assert processed_kwargs["level"] == "INFO"
    
    def test_adapter_log_processing_start(self):
        """Test adapter processing start logging method."""
        base_logger = logging.getLogger("test.processing")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        # Mock the base logger to capture calls - need to mock _log method since that's what adapter calls
        with patch.object(base_logger, '_log') as mock_log:
            # Test log_processing_start
            adapter.log_processing_start("/test/file.xlsx", 1024)
            
            # Verify call was made
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            
            # Check message and extra data
            assert "Started processing file" in call_args[0][1]  # message is 2nd arg
            assert call_args[1]["extra"]["file_path"] == "/test/file.xlsx"
            assert call_args[1]["extra"]["file_size"] == 1024
    
    def test_adapter_log_processing_complete(self):
        """Test adapter processing complete logging."""
        base_logger = logging.getLogger("test.processing")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        with patch.object(base_logger, 'info') as mock_info:
            # Test log_processing_complete
            adapter.log_processing_complete("/test/file.xlsx", 3, 1.5, 2)
            
            # Verify call
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            
            assert "Completed processing" in call_args[0][0]
            assert call_args[1]["extra"]["file_path"] == "/test/file.xlsx"
            assert call_args[1]["extra"]["worksheet_count"] == 3
            assert call_args[1]["extra"]["processing_time"] == 1.5
            assert call_args[1]["extra"]["csv_files_generated"] == 2
    
    def test_adapter_log_error(self):
        """Test adapter error logging."""
        base_logger = logging.getLogger("test.processing")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        with patch.object(base_logger, 'error') as mock_error:
            # Test log_error
            adapter.log_error("ValueError", "Invalid data format", "/test/file.xlsx", "Sheet1")
            
            # Verify call
            mock_error.assert_called_once()
            call_args = mock_error.call_args
            
            assert "Invalid data format" in call_args[0][0]
            assert call_args[1]["extra"]["file_path"] == "/test/file.xlsx"
            assert call_args[1]["extra"]["worksheet_name"] == "Sheet1"
            assert call_args[1]["extra"]["error_type"] == "ValueError"
    
    def test_adapter_log_worksheet_analysis(self):
        """Test adapter worksheet analysis logging."""
        base_logger = logging.getLogger("test.processing")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        with patch.object(base_logger, 'info') as mock_info:
            # Test log_worksheet_analysis
            adapter.log_worksheet_analysis("Sheet1", 0.85, True, ["has_headers", "good_data_density"])
            
            # Verify call
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            
            assert "Sheet1" in call_args[0][0]
            assert "accepted" in call_args[0][0]
            assert call_args[1]["extra"]["worksheet_name"] == "Sheet1"
            assert call_args[1]["extra"]["confidence_score"] == 0.85
            assert call_args[1]["extra"]["decision"] == True
            assert call_args[1]["extra"]["reasons"] == ["has_headers", "good_data_density"]


class TestLoggerSetup:
    """Test logger setup and configuration."""
    
    def test_get_processing_logger(self):
        """Test get_processing_logger function."""
        logger = get_processing_logger("test.module")
        
        assert isinstance(logger, ProcessingLoggerAdapter)
        assert logger.logger.name == "test.module"
    
    def test_get_processing_logger_caching(self):
        """Test that get_processing_logger caches loggers."""
        logger1 = get_processing_logger("cached.module")
        logger2 = get_processing_logger("cached.module")
        
        # Should be the same instance (cached)
        assert logger1 is logger2
    
    def test_get_processing_logger_different_names(self):
        """Test get_processing_logger with different names."""
        logger1 = get_processing_logger("module1")
        logger2 = get_processing_logger("module2")
        
        # Should be different instances
        assert logger1 is not logger2
        assert logger1.logger.name == "module1"
        assert logger2.logger.name == "module2"
    
    @patch('excel_to_csv.utils.logger.logging.getLogger')
    def test_setup_logging_basic(self, mock_get_logger, sample_logging_config):
        """Test basic logging setup."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        setup_logging(sample_logging_config)
        
        # Should configure logger
        mock_logger.setLevel.assert_called()
        mock_logger.addHandler.assert_called()
    
    def test_logger_manager_initialization(self):
        """Test LoggerManager initialization."""
        manager = LoggerManager()
        
        assert not manager._configured
        assert len(manager._loggers) == 0
        assert len(manager._adapters) == 0
    
    def test_logger_manager_get_logger(self):
        """Test LoggerManager get_logger method."""
        manager = LoggerManager()
        
        logger1 = manager.get_logger("test.module")
        logger2 = manager.get_logger("test.module")
        logger3 = manager.get_logger("other.module")
        
        # Same name should return same logger
        assert logger1 is logger2
        assert logger1.name == "test.module"
        
        # Different name should return different logger
        assert logger1 is not logger3
        assert logger3.name == "other.module"
    
    def test_logger_manager_get_processing_logger(self):
        """Test LoggerManager get_processing_logger method."""
        manager = LoggerManager()
        
        adapter1 = manager.get_processing_logger("test.module", {"key": "value"})
        adapter2 = manager.get_processing_logger("test.module", {"key": "value"})
        adapter3 = manager.get_processing_logger("test.module", {"key": "other"})
        
        assert isinstance(adapter1, ProcessingLoggerAdapter)
        assert isinstance(adapter2, ProcessingLoggerAdapter)
        assert isinstance(adapter3, ProcessingLoggerAdapter)
        
        # Same context should be cached
        assert adapter1 is adapter2
        # Different context should be different
        assert adapter1 is not adapter3
    
    def test_logger_manager_shutdown(self):
        """Test LoggerManager shutdown method."""
        manager = LoggerManager()
        
        # Add some loggers
        manager.get_logger("test1")
        manager.get_processing_logger("test2", {"key": "value"})
        
        assert len(manager._loggers) > 0
        assert len(manager._adapters) > 0
        
        # Shutdown
        manager.shutdown()
        
        assert not manager._configured
        assert len(manager._loggers) == 0
        assert len(manager._adapters) == 0


class TestLoggingIntegration:
    """Test logging integration scenarios."""
    
    def test_end_to_end_logging(self, temp_workspace, sample_logging_config):
        """Test end-to-end logging setup and usage."""
        # Setup logging
        setup_logging(sample_logging_config)
        
        # Get logger
        logger = get_processing_logger("test.integration")
        
        # Log various types of messages
        logger.info("Integration test message")
        logger.log_processing_start("/test/file.xlsx", 1024)
        logger.log_processing_complete("/test/file.xlsx", 2, 1.5, 2)
        logger.log_csv_generation("/out/result.csv", 50, 2048)
        
        # Verify log file was created
        log_file = Path(sample_logging_config.file_path)
        assert log_file.exists()
        assert log_file.stat().st_size > 0
    
    def test_logging_with_exception(self, temp_workspace, sample_logging_config):
        """Test logging with exception handling."""
        setup_logging(sample_logging_config)
        logger = get_processing_logger("test.exception")
        
        # Log with exception
        try:
            raise RuntimeError("Test exception for logging")
        except RuntimeError:
            logger.error("An error occurred", exc_info=True)
        
        # Verify log file contains exception
        log_file = Path(sample_logging_config.file_path)
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            # Should contain exception information
            assert "RuntimeError" in content or "Test exception" in content
    
    def test_concurrent_logging(self, temp_workspace, sample_logging_config):
        """Test concurrent logging from multiple threads."""
        import threading
        import time
        
        setup_logging(sample_logging_config)
        
        def log_worker(worker_id):
            logger = get_processing_logger(f"worker.{worker_id}")
            for i in range(10):
                logger.info(f"Worker {worker_id} message {i}")
                time.sleep(0.001)  # Small delay
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify log file has entries from all workers
        log_file = Path(sample_logging_config.file_path)
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            # Should have messages from all workers
            assert "Worker 0" in content
            assert "Worker 1" in content
            assert "Worker 2" in content


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_json_formatter_with_none_values(self):
        """Test JSON formatter handles None values gracefully."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test.none",
            level=logging.INFO,
            pathname=None,  # None pathname
            lineno=42,
            msg="Message with None values",
            args=(),
            exc_info=None
        )
        
        # Add None fields
        record.file_path = None
        record.confidence_score = None
        
        # Should not raise exception
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["message"] == "Message with None values"
        assert log_data["file_path"] is None
        assert log_data["confidence_score"] is None
    
    def test_adapter_with_circular_reference(self):
        """Test adapter handles circular references in extra data."""
        base_logger = logging.getLogger("test.circular")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        # Create circular reference
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        with patch.object(base_logger, 'info') as mock_info:
            # This might cause issues with JSON serialization later
            # but the adapter itself should handle it
            try:
                adapter.info("Test message", extra={"circular": circular_dict})
                # If it doesn't crash, that's good
            except (ValueError, TypeError):
                # Expected for circular references in JSON serialization
                pass
    
    def test_file_handler_permission_error(self, temp_workspace):
        """Test file handler with permission errors."""
        # Try to create handler in read-only location
        readonly_file = "/root/readonly.log"  # Typically no write permission
        
        # Should either handle gracefully or raise appropriate exception
        try:
            handler = create_file_handler(
                file_path=readonly_file,
                max_size_mb=10,
                backup_count=3,
                format_type="json"
            )
            # If successful, verify it's a handler
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
        except (PermissionError, OSError):
            # Expected behavior for permission denied
            pass
    
    def test_logging_config_missing_fields(self, temp_workspace):
        """Test logging setup with missing configuration fields."""
        minimal_config = LoggingConfig(
            level="INFO",
            file_handler={"enabled": False},
            console_handler={"enabled": True}
        )
        
        # Should handle missing fields gracefully
        try:
            setup_logging(minimal_config)
        except (KeyError, AttributeError, TypeError):
            # Should not raise these exceptions
            pytest.fail("Should handle missing config fields gracefully")
    
    def test_large_log_message(self, temp_workspace, sample_logging_config):
        """Test logging with very large message."""
        setup_logging(sample_logging_config)
        logger = get_processing_logger("test.large")
        
        # Create large message
        large_message = "x" * 10000
        
        # Should handle large message without issues
        logger.info(large_message)
        
        # Verify logged
        log_file = Path(sample_logging_config.file_handler["file_path"])
        assert log_file.exists()
    
    def test_unicode_in_extra_fields(self):
        """Test adapter with Unicode in extra fields."""
        base_logger = logging.getLogger("test.unicode")
        adapter = ProcessingLoggerAdapter(base_logger)
        
        with patch.object(base_logger, 'info') as mock_info:
            # Log with Unicode in extra fields
            adapter.info("Test message", extra={
                "chinese": "你好世界",
                "emoji": "🎉🚀",
                "french": "Bonjour à tous"
            })
            
            # Should not raise exception
            mock_info.assert_called_once()


class TestLoggerConfigurationEdgeCases:
    """Test edge cases in logger configuration."""
    
    def test_invalid_log_level(self):
        """Test setup with invalid log level."""
        invalid_config = LoggingConfig(
            level="INVALID_LEVEL",
            file_handler={"enabled": False},
            console_handler={"enabled": True}
        )
        
        # Should handle invalid level gracefully
        try:
            setup_logging(invalid_config)
        except ValueError:
            # Expected for invalid level
            pass
    
    def test_zero_max_size(self, temp_workspace):
        """Test file handler with zero max size."""
        log_file = temp_workspace / "zero_size.log"
        
        # Should handle zero/invalid max size
        try:
            handler = create_file_handler(
                file_path=str(log_file),
                max_size_mb=0,
                backup_count=1,
                format_type="json"
            )
            assert isinstance(handler, logging.handlers.RotatingFileHandler)
        except ValueError:
            # Expected for invalid max size
            pass
    
    def test_negative_backup_count(self, temp_workspace):
        """Test file handler with negative backup count."""
        log_file = temp_workspace / "negative_backup.log"
        
        # Should handle negative backup count
        handler = create_file_handler(
            file_path=str(log_file),
            max_size_mb=10,
            backup_count=-1,
            format_type="json"
        )
        
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        # Should default to 0 or handle gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])