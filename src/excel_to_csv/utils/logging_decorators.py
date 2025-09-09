"""Logging decorators and utilities for operation tracking.

This module provides decorators and context managers for automatic
operation logging with correlation tracking and metrics collection.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable, Generator

from .correlation import CorrelationContext
from .metrics import create_operation_metrics, get_metrics_collector, OperationMetrics


def _sanitize_args(args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize function arguments for logging.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Sanitized arguments dictionary
    """
    sanitized = {}
    
    # Handle positional args (limit to first few to avoid huge logs)
    if args:
        sanitized["args_count"] = len(args)
        # Only log first 3 args and sanitize sensitive data
        for i, arg in enumerate(args[:3]):
            arg_str = str(arg)
            if len(arg_str) > 200:
                arg_str = arg_str[:200] + "..."
            # Don't log sensitive data (check for exact sensitive patterns)
            arg_lower = str(arg).lower()
            if any(word in arg_lower for word in ["password", "token", "secret"]) or arg_lower == "key":
                arg_str = "[REDACTED]"
            sanitized[f"arg_{i}"] = arg_str
    
    # Handle keyword arguments
    if kwargs:
        sanitized["kwargs_count"] = len(kwargs)
        for key, value in list(kwargs.items())[:5]:  # Limit to 5 kwargs
            key_lower = key.lower()
            if any(word in key_lower for word in ["password", "token", "secret"]) or key_lower.endswith("key") or key_lower.endswith("_key"):
                sanitized[key] = "[REDACTED]"
            else:
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                sanitized[key] = value_str
    
    return sanitized


def _sanitize_result(result: Any) -> Any:
    """Sanitize function result for logging.
    
    Args:
        result: Function return value
        
    Returns:
        Sanitized result for logging
    """
    if result is None:
        return None
    
    result_str = str(result)
    if len(result_str) > 500:
        result_str = result_str[:500] + "..."
    
    return result_str


def log_operation(
    operation_name: str, 
    log_args: bool = True, 
    log_result: bool = False,
    collect_metrics: bool = True
) -> Callable:
    """Decorator for automatic operation logging with metrics collection.
    
    Args:
        operation_name: Name of the operation being logged
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        collect_metrics: Whether to collect performance metrics
        
    Returns:
        Decorated function with logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)
            correlation_id = CorrelationContext.ensure_correlation_id()
            
            # Start metrics collection
            metrics = None
            if collect_metrics:
                metrics = create_operation_metrics(operation_name, correlation_id)
            
            # Log operation start
            log_data = {
                "operation": operation_name,
                "status": "START"
            }
            if log_args:
                log_data["args"] = _sanitize_args(args, kwargs)
            
            logger.info("Operation started", extra={"structured": log_data})
            
            try:
                result = func(*args, **kwargs)
                
                # Complete metrics
                if metrics:
                    metrics.complete(success=True)
                    get_metrics_collector().record_operation(metrics)
                
                # Log success
                success_data = {
                    "operation": operation_name,
                    "status": "SUCCESS"
                }
                if metrics:
                    success_data["duration_ms"] = metrics.duration_ms
                if log_result:
                    success_data["result"] = _sanitize_result(result)
                
                logger.info("Operation completed successfully", extra={"structured": success_data})
                return result
                
            except Exception as e:
                # Complete metrics with error
                if metrics:
                    metrics.complete(success=False, error_type=type(e).__name__)
                    get_metrics_collector().record_operation(metrics)
                
                # Log error
                error_data = {
                    "operation": operation_name,
                    "status": "ERROR",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                if metrics:
                    error_data["duration_ms"] = metrics.duration_ms
                
                logger.error("Operation failed", extra={"structured": error_data}, exc_info=True)
                raise
        
        return wrapper
    return decorator


@contextmanager
def operation_context(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    collect_metrics: bool = True,
    **metadata: Any
) -> Generator[OperationMetrics, None, None]:
    """Context manager for operation tracking with logging and metrics.
    
    Args:
        operation_name: Name of the operation
        logger: Logger to use (defaults to generic logger)
        collect_metrics: Whether to collect metrics
        **metadata: Additional metadata to include
        
    Yields:
        OperationMetrics instance for the operation
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    correlation_id = CorrelationContext.ensure_correlation_id()
    
    # Start metrics collection
    metrics = None
    if collect_metrics:
        metrics = create_operation_metrics(operation_name, correlation_id)
        for key, value in metadata.items():
            metrics.add_metadata(key, value)
    
    # Log operation start
    start_data = {
        "operation": operation_name,
        "status": "START",
        **metadata
    }
    logger.info("Operation context started", extra={"structured": start_data})
    
    try:
        yield metrics
        
        # Complete metrics on success
        if metrics:
            metrics.complete(success=True)
            get_metrics_collector().record_operation(metrics)
        
        # Log success
        success_data = {
            "operation": operation_name,
            "status": "SUCCESS"
        }
        if metrics:
            success_data["duration_ms"] = metrics.duration_ms
        
        logger.info("Operation context completed successfully", extra={"structured": success_data})
        
    except Exception as e:
        # Complete metrics with error
        if metrics:
            metrics.complete(success=False, error_type=type(e).__name__)
            get_metrics_collector().record_operation(metrics)
        
        # Log error
        error_data = {
            "operation": operation_name,
            "status": "ERROR",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        if metrics:
            error_data["duration_ms"] = metrics.duration_ms
        
        logger.error("Operation context failed", extra={"structured": error_data}, exc_info=True)
        raise


def log_method(
    log_args: bool = True,
    log_result: bool = False,
    collect_metrics: bool = True
) -> Callable:
    """Decorator for logging class methods with automatic operation naming.
    
    Args:
        log_args: Whether to log method arguments
        log_result: Whether to log method result
        collect_metrics: Whether to collect metrics
        
    Returns:
        Decorated method with logging
    """
    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs) -> Any:
            # Create operation name from class and method
            operation_name = f"{self.__class__.__name__}.{method.__name__}"
            
            # Use the log_operation decorator
            logged_method = log_operation(
                operation_name=operation_name,
                log_args=log_args,
                log_result=log_result,
                collect_metrics=collect_metrics
            )(method)
            
            return logged_method(self, *args, **kwargs)
        
        return wrapper
    return decorator