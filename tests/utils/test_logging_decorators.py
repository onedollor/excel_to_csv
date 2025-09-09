"""Tests for logging decorators and utilities."""

import logging
import pytest
import time
from unittest.mock import Mock, patch

from excel_to_csv.utils.correlation import CorrelationContext
from excel_to_csv.utils.logging_decorators import (
    log_operation,
    log_method,
    operation_context,
    _sanitize_args,
    _sanitize_result
)
from excel_to_csv.utils.metrics import get_metrics_collector


class TestSanitizationFunctions:
    """Test cases for argument and result sanitization."""
    
    def test_sanitize_args_basic(self):
        """Test basic argument sanitization."""
        args = ("arg1", "arg2", "arg3")
        kwargs = {"param1": "value1", "param2": "value2"}
        
        result = _sanitize_args(args, kwargs)
        
        assert result["args_count"] == 3
        assert result["arg_0"] == "arg1"
        assert result["arg_1"] == "arg2"
        assert result["arg_2"] == "arg3"
        assert result["kwargs_count"] == 2
        assert result["param1"] == "value1"
        assert result["param2"] == "value2"
    
    def test_sanitize_args_sensitive_data(self):
        """Test sanitization of sensitive data."""
        args = ("username", "secret_password")
        kwargs = {"api_key": "secret123", "data": "normal_data"}
        
        result = _sanitize_args(args, kwargs)
        
        assert result["arg_0"] == "username"
        assert result["arg_1"] == "[REDACTED]"  # Contains "password"
        assert result["api_key"] == "[REDACTED]"  # Contains "key"
        assert result["data"] == "normal_data"
    
    def test_sanitize_args_long_values(self):
        """Test sanitization of long argument values."""
        long_string = "x" * 300
        args = (long_string,)
        kwargs = {"long_data": long_string}  # Use "long_data" instead of "long_key" to avoid redaction
        
        result = _sanitize_args(args, kwargs)
        
        assert result["arg_0"].endswith("...")
        assert len(result["arg_0"]) == 203  # 200 chars + "..."
        assert result["long_data"].endswith("...")
        assert len(result["long_data"]) == 203  # 200 chars + "..."
    
    def test_sanitize_args_limits(self):
        """Test that sanitization respects limits."""
        # More than 3 args
        args = tuple(f"arg{i}" for i in range(10))
        kwargs = {f"key{i}": f"value{i}" for i in range(10)}
        
        result = _sanitize_args(args, kwargs)
        
        assert result["args_count"] == 10
        assert "arg_2" in result  # Last included arg
        assert "arg_3" not in result  # Should be excluded
        
        assert result["kwargs_count"] == 10
        assert len([k for k in result.keys() if k.startswith("key")]) == 5  # Max 5 kwargs
    
    def test_sanitize_result_basic(self):
        """Test basic result sanitization."""
        result = _sanitize_result("simple result")
        assert result == "simple result"
        
        result = _sanitize_result(None)
        assert result is None
        
        result = _sanitize_result(42)
        assert result == "42"
    
    def test_sanitize_result_long_value(self):
        """Test sanitization of long result values."""
        long_result = "x" * 600
        result = _sanitize_result(long_result)
        
        assert result.endswith("...")
        assert len(result) == 503  # 500 chars + "..."


class TestLogOperationDecorator:
    """Test cases for log_operation decorator."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear metrics collector
        get_metrics_collector().clear_metrics()
        
        # Setup mock logger
        self.mock_logger = Mock()
        self.logger_patch = patch('excel_to_csv.utils.logging_decorators.logging.getLogger')
        self.mock_get_logger = self.logger_patch.start()
        self.mock_get_logger.return_value = self.mock_logger
    
    def teardown_method(self):
        """Teardown for each test method."""
        self.logger_patch.stop()
    
    def test_log_operation_success(self):
        """Test log_operation decorator with successful operation."""
        @log_operation("test_operation")
        def test_function(arg1, arg2="default"):
            return f"result: {arg1}, {arg2}"
        
        # Set correlation ID
        CorrelationContext.set_correlation_id("test-id-123")
        
        result = test_function("value1", arg2="value2")
        
        assert result == "result: value1, value2"
        
        # Check logging calls
        assert self.mock_logger.info.call_count == 2
        assert self.mock_logger.error.call_count == 0
        
        # Check start log
        start_call = self.mock_logger.info.call_args_list[0]
        assert "Operation started" in start_call[0][0]
        start_extra = start_call[1]["extra"]["structured"]
        assert start_extra["operation"] == "test_operation"
        assert start_extra["status"] == "START"
        
        # Check success log
        success_call = self.mock_logger.info.call_args_list[1]
        assert "Operation completed successfully" in success_call[0][0]
        success_extra = success_call[1]["extra"]["structured"]
        assert success_extra["operation"] == "test_operation"
        assert success_extra["status"] == "SUCCESS"
        assert "duration_ms" in success_extra
    
    def test_log_operation_failure(self):
        """Test log_operation decorator with failed operation."""
        @log_operation("test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        CorrelationContext.set_correlation_id("test-id-456")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # Check logging calls
        assert self.mock_logger.info.call_count == 1  # Only start log
        assert self.mock_logger.error.call_count == 1
        
        # Check error log
        error_call = self.mock_logger.error.call_args_list[0]
        assert "Operation failed" in error_call[0][0]
        error_extra = error_call[1]["extra"]["structured"]
        assert error_extra["operation"] == "test_operation"
        assert error_extra["status"] == "ERROR"
        assert error_extra["error_type"] == "ValueError"
        assert error_extra["error_message"] == "Test error"
    
    def test_log_operation_without_args_logging(self):
        """Test log_operation decorator without argument logging."""
        @log_operation("test_operation", log_args=False)
        def test_function(secret_arg):
            return "result"
        
        result = test_function("sensitive_data")
        
        # Check that args are not logged
        start_call = self.mock_logger.info.call_args_list[0]
        start_extra = start_call[1]["extra"]["structured"]
        assert "args" not in start_extra
    
    def test_log_operation_with_result_logging(self):
        """Test log_operation decorator with result logging."""
        @log_operation("test_operation", log_result=True)
        def test_function():
            return {"key": "value", "number": 42}
        
        result = test_function()
        
        # Check that result is logged
        success_call = self.mock_logger.info.call_args_list[1]
        success_extra = success_call[1]["extra"]["structured"]
        assert "result" in success_extra
        assert "key" in str(success_extra["result"])
    
    def test_log_operation_metrics_collection(self):
        """Test that metrics are collected properly."""
        @log_operation("test_operation")
        def test_function():
            time.sleep(0.01)  # Small delay for measurable duration
            return "success"
        
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        result = test_function()
        
        # Check metrics were recorded
        assert len(collector.metrics) == initial_count + 1
        
        metrics = collector.metrics[-1]
        assert metrics.operation_name == "test_operation"
        assert metrics.success is True
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_log_operation_no_metrics_collection(self):
        """Test log_operation decorator without metrics collection."""
        @log_operation("test_operation", collect_metrics=False)
        def test_function():
            return "success"
        
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        result = test_function()
        
        # Check no metrics were recorded
        assert len(collector.metrics) == initial_count


class TestLogMethodDecorator:
    """Test cases for log_method decorator."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_logger = Mock()
        self.logger_patch = patch('excel_to_csv.utils.logging_decorators.logging.getLogger')
        self.mock_get_logger = self.logger_patch.start()
        self.mock_get_logger.return_value = self.mock_logger
    
    def teardown_method(self):
        """Teardown for each test method."""
        self.logger_patch.stop()
    
    def test_log_method_basic(self):
        """Test log_method decorator basic functionality."""
        class TestClass:
            @log_method()
            def test_method(self, arg):
                return f"processed: {arg}"
        
        obj = TestClass()
        result = obj.test_method("test_value")
        
        assert result == "processed: test_value"
        
        # Check operation name includes class and method
        start_call = self.mock_logger.info.call_args_list[0]
        start_extra = start_call[1]["extra"]["structured"]
        assert start_extra["operation"] == "TestClass.test_method"


class TestOperationContext:
    """Test cases for operation_context context manager."""
    
    def setup_method(self):
        """Setup for each test method."""
        get_metrics_collector().clear_metrics()
        
        self.mock_logger = Mock()
    
    def test_operation_context_success(self):
        """Test operation_context with successful operation."""
        with operation_context("test_operation", self.mock_logger) as metrics:
            assert metrics is not None
            assert metrics.operation_name == "test_operation"
            time.sleep(0.01)  # Small delay
        
        # Check logging
        assert self.mock_logger.info.call_count == 2
        
        start_call = self.mock_logger.info.call_args_list[0]
        assert "Operation context started" in start_call[0][0]
        
        success_call = self.mock_logger.info.call_args_list[1]
        assert "Operation context completed successfully" in success_call[0][0]
        
        # Check metrics were recorded
        collector = get_metrics_collector()
        assert len(collector.metrics) >= 1
        recorded_metrics = collector.metrics[-1]
        assert recorded_metrics.success is True
        assert recorded_metrics.duration_ms is not None
    
    def test_operation_context_failure(self):
        """Test operation_context with failed operation."""
        with pytest.raises(ValueError, match="Test error"):
            with operation_context("test_operation", self.mock_logger) as metrics:
                raise ValueError("Test error")
        
        # Check error logging
        assert self.mock_logger.info.call_count == 1  # Only start
        assert self.mock_logger.error.call_count == 1
        
        error_call = self.mock_logger.error.call_args_list[0]
        assert "Operation context failed" in error_call[0][0]
        
        # Check metrics show failure
        collector = get_metrics_collector()
        recorded_metrics = collector.metrics[-1]
        assert recorded_metrics.success is False
        assert recorded_metrics.error_type == "ValueError"
    
    def test_operation_context_with_metadata(self):
        """Test operation_context with metadata."""
        with operation_context(
            "test_operation", 
            self.mock_logger,
            file_name="test.xlsx",
            worksheet_count=3
        ) as metrics:
            assert metrics.metadata["file_name"] == "test.xlsx"
            assert metrics.metadata["worksheet_count"] == 3
        
        # Check metadata in logging
        start_call = self.mock_logger.info.call_args_list[0]
        start_extra = start_call[1]["extra"]["structured"]
        assert start_extra["file_name"] == "test.xlsx"
        assert start_extra["worksheet_count"] == 3
    
    def test_operation_context_no_metrics(self):
        """Test operation_context without metrics collection."""
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        with operation_context("test_operation", self.mock_logger, collect_metrics=False):
            pass
        
        # No metrics should be recorded
        assert len(collector.metrics) == initial_count
    
    def test_operation_context_default_logger(self):
        """Test operation_context with default logger."""
        with patch('excel_to_csv.utils.logging_decorators.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with operation_context("test_operation") as metrics:
                pass
            
            # Should have used default logger
            mock_get_logger.assert_called_once()
            assert mock_logger.info.call_count == 2