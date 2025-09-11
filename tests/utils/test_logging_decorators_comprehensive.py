"""Comprehensive tests for logging decorators targeting 90%+ coverage.

This test suite covers all edge cases and uncovered code paths including:
- Sensitive data detection edge cases
- Error handling in all scenarios
- Context manager edge cases
- Metrics integration
- Logger fallbacks and error scenarios
"""

import logging
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from excel_to_csv.utils.correlation import CorrelationContext
from excel_to_csv.utils.logging_decorators import (
    log_operation,
    log_method,
    operation_context,
    _sanitize_args,
    _sanitize_result
)
from excel_to_csv.utils.metrics import get_metrics_collector, OperationMetrics


class TestSanitizationComprehensive:
    """Comprehensive sanitization testing."""
    
    def test_sanitize_args_empty_inputs(self):
        """Test sanitization with empty inputs."""
        result = _sanitize_args((), {})
        assert result == {}
    
    def test_sanitize_args_sensitive_patterns_comprehensive(self):
        """Test all sensitive data patterns."""
        # Test various sensitive patterns
        args = (
            "password123",          # contains "password"
            "my_token_value",       # contains "token"
            "secret_data",          # contains "secret"
            "key",                  # exact match "key"
            "normal_data"          # not sensitive
        )
        
        kwargs = {
            "user_password": "secret123",      # contains "password"
            "api_token": "token123",           # contains "token"
            "secret_key": "mysecret",          # contains "secret"
            "auth_key": "authvalue",           # ends with "_key"
            "encryption_key": "enckey",        # ends with "_key"
            "normal_param": "normalvalue",     # not sensitive
            "key": "exactkey",                 # exact match "key"
            "keychain": "notkey"               # contains "key" but not exact or ending pattern
        }
        
        result = _sanitize_args(args, kwargs)
        
        # Check args
        assert result["arg_0"] == "[REDACTED]"  # password123
        assert result["arg_1"] == "[REDACTED]"  # my_token_value
        assert result["arg_2"] == "[REDACTED]"  # secret_data
        assert result["arg_3"] == "[REDACTED]"  # key (exact match)
        # Only first 3 args are logged, so arg_4 won't be in result
        
        # Check kwargs
        assert result["user_password"] == "[REDACTED]"
        assert result["api_token"] == "[REDACTED]"
        assert result["secret_key"] == "[REDACTED]"
        assert result["auth_key"] == "[REDACTED]"
        assert result["encryption_key"] == "[REDACTED]"
        assert result["normal_param"] == "normalvalue"
        assert result["key"] == "[REDACTED]"
        assert result["keychain"] == "notkey"  # Not a sensitive pattern
    
    def test_sanitize_args_case_sensitivity(self):
        """Test case sensitivity in sensitive data detection."""
        args = ("PASSWORD", "Token", "SECRET")
        kwargs = {
            "API_KEY": "value",
            "Secret_Token": "value",
            "NORMAL": "value"
        }
        
        result = _sanitize_args(args, kwargs)
        
        # Should be case insensitive
        assert result["arg_0"] == "[REDACTED]"  # PASSWORD
        assert result["arg_1"] == "[REDACTED]"  # Token
        assert result["arg_2"] == "[REDACTED]"  # SECRET
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Secret_Token"] == "[REDACTED]"
        assert result["NORMAL"] == "value"
    
    def test_sanitize_args_max_limits_exceeded(self):
        """Test behavior when limits are exceeded."""
        # Test with more than 3 args
        args = tuple(f"arg_{i}" for i in range(10))
        # Test with more than 5 kwargs
        kwargs = {f"param_{i}": f"value_{i}" for i in range(10)}
        
        result = _sanitize_args(args, kwargs)
        
        # Should have exactly 3 args logged
        assert result["args_count"] == 10
        assert "arg_0" in result
        assert "arg_1" in result
        assert "arg_2" in result
        assert "arg_3" not in result
        
        # Should have exactly 5 kwargs logged
        assert result["kwargs_count"] == 10
        logged_kwargs = [k for k in result.keys() if k.startswith("param_")]
        assert len(logged_kwargs) == 5
    
    def test_sanitize_args_value_conversion_edge_cases(self):
        """Test value conversion edge cases."""
        # Test various data types
        args = (
            None,
            42,
            3.14,
            True,
            ["list", "data"],
            {"dict": "data"},
            object(),
        )
        
        result = _sanitize_args(args, {})
        
        # All should be converted to strings
        assert "None" in result["arg_0"]
        assert "42" in result["arg_1"]
        assert "3.14" in result["arg_2"]
    
    def test_sanitize_result_edge_cases(self):
        """Test result sanitization edge cases."""
        # Test various data types
        assert _sanitize_result(None) is None
        assert _sanitize_result(42) == "42"
        assert _sanitize_result([1, 2, 3]) == "[1, 2, 3]"
        assert _sanitize_result({"key": "value"}) == "{'key': 'value'}"
        
        # Test very long result
        long_result = "x" * 1000
        sanitized = _sanitize_result(long_result)
        assert len(sanitized) == 503  # 500 + "..."
        assert sanitized.endswith("...")


class TestLogOperationComprehensive:
    """Comprehensive log_operation decorator testing."""
    
    def setup_method(self):
        """Setup for each test method."""
        get_metrics_collector().clear_metrics()
        self.mock_logger = Mock()
        self.logger_patch = patch('excel_to_csv.utils.logging_decorators.logging.getLogger')
        self.mock_get_logger = self.logger_patch.start()
        self.mock_get_logger.return_value = self.mock_logger
    
    def teardown_method(self):
        """Teardown for each test method."""
        self.logger_patch.stop()
    
    def test_log_operation_correlation_id_generation(self):
        """Test correlation ID generation when none exists."""
        # Clear any existing correlation ID
        CorrelationContext.clear_correlation_id()
        
        @log_operation("test_operation")
        def test_function():
            return "success"
        
        # Mock correlation ID generation
        with patch.object(CorrelationContext, 'ensure_correlation_id') as mock_ensure:
            mock_ensure.return_value = "generated-id-123"
            
            result = test_function()
            
            # Should have called ensure_correlation_id
            mock_ensure.assert_called_once()
    
    def test_log_operation_all_parameters_combination(self):
        """Test all parameter combinations."""
        @log_operation(
            "complex_operation",
            log_args=True,
            log_result=True,
            collect_metrics=True
        )
        def test_function(arg1, arg2="default", **kwargs):
            time.sleep(0.001)  # Small delay for metrics
            return {"processed": arg1, "default": arg2, "extras": kwargs}
        
        result = test_function("test_value", arg2="custom", extra_param="extra")
        
        # Check all logging occurred
        assert self.mock_logger.info.call_count == 2  # start + success
        
        # Check args were logged
        start_call = self.mock_logger.info.call_args_list[0]
        start_extra = start_call[1]["extra"]["structured"]
        assert "args" in start_extra
        
        # Check result was logged
        success_call = self.mock_logger.info.call_args_list[1]
        success_extra = success_call[1]["extra"]["structured"]
        assert "result" in success_extra
        assert "duration_ms" in success_extra
        
        # Check metrics were collected
        collector = get_metrics_collector()
        assert len(collector.metrics) >= 1
        metrics = collector.metrics[-1]
        assert metrics.success is True
        assert metrics.operation_name == "complex_operation"
    
    def test_log_operation_exception_handling_with_metrics(self):
        """Test exception handling preserves metrics."""
        @log_operation("failing_operation", collect_metrics=True)
        def failing_function():
            time.sleep(0.001)
            raise RuntimeError("Custom error message")
        
        with pytest.raises(RuntimeError, match="Custom error message"):
            failing_function()
        
        # Check error was logged
        assert self.mock_logger.error.call_count == 1
        error_call = self.mock_logger.error.call_args_list[0]
        error_extra = error_call[1]["extra"]["structured"]
        assert error_extra["error_type"] == "RuntimeError"
        assert error_extra["error_message"] == "Custom error message"
        assert "duration_ms" in error_extra
        
        # Check metrics recorded failure
        collector = get_metrics_collector()
        metrics = collector.metrics[-1]
        assert metrics.success is False
        assert metrics.error_type == "RuntimeError"
    
    def test_log_operation_metrics_disabled(self):
        """Test operation with metrics collection disabled."""
        @log_operation("no_metrics_operation", collect_metrics=False)
        def test_function():
            return "success"
        
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        result = test_function()
        
        # No metrics should be recorded
        assert len(collector.metrics) == initial_count
        
        # But logging should still work
        assert self.mock_logger.info.call_count == 2


class TestLogMethodComprehensive:
    """Comprehensive log_method decorator testing."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.mock_logger = Mock()
        self.logger_patch = patch('excel_to_csv.utils.logging_decorators.logging.getLogger')
        self.mock_get_logger = self.logger_patch.start()
        self.mock_get_logger.return_value = self.mock_logger
    
    def teardown_method(self):
        """Teardown for each test method."""
        self.logger_patch.stop()
    
    def test_log_method_with_all_options(self):
        """Test log_method with all options enabled."""
        class TestClass:
            @log_method(log_args=True, log_result=True, collect_metrics=True)
            def process_data(self, data, option="default"):
                return f"processed_{data}_{option}"
        
        obj = TestClass()
        result = obj.process_data("test_data", option="custom")
        
        assert result == "processed_test_data_custom"
        
        # Check operation name format
        start_call = self.mock_logger.info.call_args_list[0]
        start_extra = start_call[1]["extra"]["structured"]
        assert start_extra["operation"] == "TestClass.process_data"
    
    def test_log_method_inheritance(self):
        """Test log_method with class inheritance."""
        class BaseClass:
            @log_method()
            def base_method(self):
                return "base_result"
        
        class DerivedClass(BaseClass):
            @log_method()
            def derived_method(self):
                return "derived_result"
        
        obj = DerivedClass()
        
        # Test base method
        result1 = obj.base_method()
        assert result1 == "base_result"
        
        # Test derived method
        result2 = obj.derived_method()
        assert result2 == "derived_result"
        
        # Check operation names include correct class names
        calls = self.mock_logger.info.call_args_list
        # Should have 4 calls: 2 start + 2 success
        assert len(calls) >= 4
        
        # Find the operation names in the calls
        operation_names = []
        for call in calls:
            if len(call) > 1 and "extra" in call[1]:
                structured = call[1]["extra"].get("structured", {})
                if "operation" in structured:
                    operation_names.append(structured["operation"])
        
        assert "BaseClass.base_method" in operation_names
        assert "DerivedClass.derived_method" in operation_names
    
    def test_log_method_error_handling(self):
        """Test log_method error handling."""
        class TestClass:
            @log_method(collect_metrics=True)
            def failing_method(self):
                raise ValueError("Method failed")
        
        obj = TestClass()
        
        with pytest.raises(ValueError, match="Method failed"):
            obj.failing_method()
        
        # Check error was logged correctly
        assert self.mock_logger.error.call_count == 1
        error_call = self.mock_logger.error.call_args_list[0]
        error_extra = error_call[1]["extra"]["structured"]
        assert error_extra["operation"] == "TestClass.failing_method"


class TestOperationContextComprehensive:
    """Comprehensive operation_context testing."""
    
    def setup_method(self):
        """Setup for each test method."""
        get_metrics_collector().clear_metrics()
        self.mock_logger = Mock()
    
    def test_operation_context_metrics_only_mode(self):
        """Test operation context with metrics but no custom metadata."""
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        with operation_context("metrics_operation", self.mock_logger) as metrics:
            assert metrics is not None
            assert isinstance(metrics, OperationMetrics)
            time.sleep(0.001)
        
        # Check metrics were recorded
        assert len(collector.metrics) == initial_count + 1
        recorded_metrics = collector.metrics[-1]
        assert recorded_metrics.operation_name == "metrics_operation"
        assert recorded_metrics.success is True
    
    def test_operation_context_no_logger_provided(self):
        """Test operation context with default logger."""
        with patch('excel_to_csv.utils.logging_decorators.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with operation_context("default_logger_operation") as metrics:
                pass
            
            # Should have used default logger from module name
            mock_get_logger.assert_called_once_with('excel_to_csv.utils.logging_decorators')
            assert mock_logger.info.call_count == 2
    
    def test_operation_context_complex_metadata(self):
        """Test operation context with complex metadata."""
        complex_metadata = {
            "file_path": "/path/to/file.xlsx",
            "sheet_count": 5,
            "processing_mode": "batch",
            "nested_data": {"key": "value", "number": 42}
        }
        
        with operation_context("complex_operation", self.mock_logger, **complex_metadata) as metrics:
            # Verify metadata was added to metrics
            for key, value in complex_metadata.items():
                assert metrics.metadata[key] == value
        
        # Verify metadata appears in logging
        start_call = self.mock_logger.info.call_args_list[0]
        start_extra = start_call[1]["extra"]["structured"]
        
        for key, value in complex_metadata.items():
            assert start_extra[key] == value
    
    def test_operation_context_exception_propagation(self):
        """Test that exceptions are properly propagated."""
        class CustomException(Exception):
            pass
        
        with pytest.raises(CustomException):
            with operation_context("exception_operation", self.mock_logger) as metrics:
                raise CustomException("Custom error")
        
        # Check error handling
        assert self.mock_logger.error.call_count == 1
        error_call = self.mock_logger.error.call_args_list[0]
        error_extra = error_call[1]["extra"]["structured"]
        assert error_extra["error_type"] == "CustomException"
        assert error_extra["error_message"] == "Custom error"
    
    def test_operation_context_metrics_disabled(self):
        """Test operation context with metrics disabled."""
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        with operation_context("no_metrics", self.mock_logger, collect_metrics=False) as metrics:
            # Metrics object should be None when disabled
            assert metrics is None
        
        # No metrics should be recorded
        assert len(collector.metrics) == initial_count
        
        # But logging should still work
        assert self.mock_logger.info.call_count == 2
    
    def test_operation_context_error_in_metrics_disabled_mode(self):
        """Test error handling when metrics are disabled."""
        with pytest.raises(ValueError):
            with operation_context("error_no_metrics", self.mock_logger, collect_metrics=False) as metrics:
                assert metrics is None
                raise ValueError("Error without metrics")
        
        # Error should still be logged
        assert self.mock_logger.error.call_count == 1


class TestEdgeCasesAndErrorScenarios:
    """Test edge cases and error scenarios."""
    
    def test_function_name_preservation(self):
        """Test that decorated functions preserve original names and metadata."""
        @log_operation("test_op")
        def original_function(arg):
            """Original docstring."""
            return arg
        
        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."
    
    def test_complex_nested_decorators(self):
        """Test interaction with other decorators."""
        call_order = []
        
        def tracking_decorator(name):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    call_order.append(f"{name}_start")
                    result = func(*args, **kwargs)
                    call_order.append(f"{name}_end")
                    return result
                return wrapper
            return decorator
        
        @tracking_decorator("outer")
        @log_operation("nested_test")
        @tracking_decorator("inner")
        def nested_function():
            call_order.append("function_body")
            return "nested_result"
        
        with patch('excel_to_csv.utils.logging_decorators.logging.getLogger'):
            result = nested_function()
        
        assert result == "nested_result"
        # Verify decorator execution order
        assert "outer_start" in call_order
        assert "inner_start" in call_order
        assert "function_body" in call_order
        assert "inner_end" in call_order
        assert "outer_end" in call_order
    
    def test_correlation_context_integration(self):
        """Test integration with correlation context."""
        # Test with existing correlation ID
        CorrelationContext.set_correlation_id("existing-id-123")
        
        @log_operation("correlation_test")
        def test_function():
            current_id = CorrelationContext.get_correlation_id()
            return current_id
        
        with patch('excel_to_csv.utils.logging_decorators.logging.getLogger'):
            result = test_function()
        
        assert result == "existing-id-123"
        
        # Test with no existing correlation ID
        CorrelationContext.clear_correlation_id()
        
        with patch.object(CorrelationContext, 'ensure_correlation_id') as mock_ensure:
            mock_ensure.return_value = "new-generated-id"
            
            with patch('excel_to_csv.utils.logging_decorators.logging.getLogger'):
                test_function()
            
            mock_ensure.assert_called_once()
    
    def test_metrics_collector_integration(self):
        """Test proper integration with metrics collector."""
        collector = get_metrics_collector()
        initial_metrics = len(collector.metrics)
        
        @log_operation("metrics_integration_test")
        def test_function():
            time.sleep(0.001)
            return "success"
        
        @log_operation("another_operation")
        def another_function():
            raise Exception("Test exception")
        
        with patch('excel_to_csv.utils.logging_decorators.logging.getLogger'):
            # Successful operation
            test_function()
            
            # Failed operation
            try:
                another_function()
            except Exception:
                pass
        
        # Should have 2 new metrics entries
        assert len(collector.metrics) == initial_metrics + 2
        
        # Check success metric
        success_metric = collector.metrics[-2]
        assert success_metric.success is True
        assert success_metric.operation_name == "metrics_integration_test"
        assert success_metric.duration_ms is not None
        
        # Check failure metric
        failure_metric = collector.metrics[-1]
        assert failure_metric.success is False
        assert failure_metric.operation_name == "another_operation"
        assert failure_metric.error_type == "Exception"
    
    def test_large_data_sanitization_performance(self):
        """Test sanitization with large data sets."""
        # Create large arguments
        large_data = "x" * 10000
        many_kwargs = {f"param_{i}": f"value_{i}" * 100 for i in range(20)}
        
        # Should complete without timeout
        result = _sanitize_args((large_data,), many_kwargs)
        
        # Should handle large data appropriately
        assert len(result["arg_0"]) <= 203  # Truncated
        assert result["kwargs_count"] == 20
        # Only first 5 kwargs should be logged
        logged_kwargs = [k for k in result.keys() if k.startswith("param_")]
        assert len(logged_kwargs) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])