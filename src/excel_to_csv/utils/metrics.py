"""Operation metrics tracking system for performance analysis and monitoring.

This module provides utilities for tracking operation performance, success rates,
and detailed metrics across the excel_to_csv system.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from threading import Lock


@dataclass
class OperationMetrics:
    """Tracks metrics for a single operation."""
    
    operation_name: str
    correlation_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool, error_type: Optional[str] = None) -> None:
        """Mark operation as complete and calculate duration.
        
        Args:
            success: Whether the operation succeeded
            error_type: Type of error if operation failed
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_type = error_type
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization.
        
        Returns:
            Dictionary representation of metrics
        """
        return {
            "operation_name": self.operation_name,
            "correlation_id": self.correlation_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_type": self.error_type,
            "metadata": self.metadata.copy()
        }


class MetricsCollector:
    """Collects and aggregates operation metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[OperationMetrics] = []
        self._lock = Lock()
    
    def record_operation(self, metrics: OperationMetrics) -> None:
        """Record completed operation metrics.
        
        Args:
            metrics: The completed operation metrics
        """
        with self._lock:
            self.metrics.append(metrics)
    
    def get_metrics_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for operations.
        
        Args:
            operation_name: Optional filter by operation name
            
        Returns:
            Summary statistics dictionary
        """
        with self._lock:
            filtered_metrics = self.metrics
            if operation_name:
                filtered_metrics = [m for m in self.metrics if m.operation_name == operation_name]
        
        if not filtered_metrics:
            return {"total_operations": 0}
        
        completed_metrics = [m for m in filtered_metrics if m.success is not None]
        successful_metrics = [m for m in completed_metrics if m.success]
        failed_metrics = [m for m in completed_metrics if not m.success]
        
        durations = [m.duration_ms for m in completed_metrics if m.duration_ms is not None]
        
        summary = {
            "total_operations": len(filtered_metrics),
            "completed_operations": len(completed_metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(failed_metrics),
            "success_rate": len(successful_metrics) / len(completed_metrics) if completed_metrics else 0,
        }
        
        if durations:
            summary.update({
                "avg_duration_ms": sum(durations) / len(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "total_duration_ms": sum(durations)
            })
        
        # Error breakdown
        if failed_metrics:
            error_counts = {}
            for metrics in failed_metrics:
                error_type = metrics.error_type or "Unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            summary["error_breakdown"] = error_counts
        
        return summary
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self.metrics.clear()
    
    def get_recent_metrics(self, limit: int = 100) -> List[OperationMetrics]:
        """Get most recent metrics.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of recent operation metrics
        """
        with self._lock:
            return self.metrics[-limit:] if len(self.metrics) > limit else self.metrics.copy()


# Global metrics collector instance
_global_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.
    
    Returns:
        Global metrics collector
    """
    return _global_metrics_collector


def create_operation_metrics(operation_name: str, correlation_id: str) -> OperationMetrics:
    """Create new operation metrics instance.
    
    Args:
        operation_name: Name of the operation being tracked
        correlation_id: Correlation ID for the operation
        
    Returns:
        New OperationMetrics instance
    """
    return OperationMetrics(
        operation_name=operation_name,
        correlation_id=correlation_id,
        start_time=time.time()
    )