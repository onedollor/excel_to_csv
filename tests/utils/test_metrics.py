"""Tests for operation metrics system."""

import pytest
import time
import threading
from excel_to_csv.utils.metrics import (
    OperationMetrics,
    MetricsCollector,
    create_operation_metrics,
    get_metrics_collector
)


class TestOperationMetrics:
    """Test cases for OperationMetrics."""
    
    def test_init(self):
        """Test OperationMetrics initialization."""
        start_time = time.time()
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=start_time
        )
        
        assert metrics.operation_name == "test_operation"
        assert metrics.correlation_id == "test-id-123"
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.duration_ms is None
        assert metrics.success is None
        assert metrics.error_type is None
        assert isinstance(metrics.metadata, dict)
        assert len(metrics.metadata) == 0
    
    def test_complete_success(self):
        """Test completing metrics successfully."""
        start_time = time.time()
        metrics = OperationMetrics("test_op", "test-id", start_time)
        
        time.sleep(0.01)  # Small delay to ensure duration > 0
        metrics.complete(success=True)
        
        assert metrics.success is True
        assert metrics.error_type is None
        assert metrics.end_time is not None
        assert metrics.end_time > start_time
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_complete_failure(self):
        """Test completing metrics with failure."""
        start_time = time.time()
        metrics = OperationMetrics("test_op", "test-id", start_time)
        
        time.sleep(0.01)
        metrics.complete(success=False, error_type="ValueError")
        
        assert metrics.success is False
        assert metrics.error_type == "ValueError"
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_add_metadata(self):
        """Test adding metadata to metrics."""
        metrics = OperationMetrics("test_op", "test-id", time.time())
        
        metrics.add_metadata("file_size", 1024)
        metrics.add_metadata("worksheet_count", 3)
        
        assert metrics.metadata["file_size"] == 1024
        assert metrics.metadata["worksheet_count"] == 3
        assert len(metrics.metadata) == 2
    
    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        start_time = time.time()
        metrics = OperationMetrics("test_op", "test-id", start_time)
        metrics.add_metadata("test_key", "test_value")
        metrics.complete(success=True)
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["operation_name"] == "test_op"
        assert result["correlation_id"] == "test-id"
        assert result["start_time"] == start_time
        assert result["success"] is True
        assert result["error_type"] is None
        assert result["metadata"]["test_key"] == "test_value"
        assert "duration_ms" in result
        assert "end_time" in result


class TestMetricsCollector:
    """Test cases for MetricsCollector."""
    
    def test_init(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        
        assert isinstance(collector.metrics, list)
        assert len(collector.metrics) == 0
        assert hasattr(collector, '_lock')
    
    def test_record_operation(self):
        """Test recording operation metrics."""
        collector = MetricsCollector()
        metrics = create_operation_metrics("test_op", "test-id")
        metrics.complete(success=True)
        
        collector.record_operation(metrics)
        
        assert len(collector.metrics) == 1
        assert collector.metrics[0] == metrics
    
    def test_get_metrics_summary_empty(self):
        """Test getting summary with no metrics."""
        collector = MetricsCollector()
        
        summary = collector.get_metrics_summary()
        
        assert summary["total_operations"] == 0
    
    def test_get_metrics_summary_basic(self):
        """Test getting basic metrics summary."""
        collector = MetricsCollector()
        
        # Add successful operation
        metrics1 = create_operation_metrics("test_op", "id1")
        metrics1.complete(success=True)
        collector.record_operation(metrics1)
        
        # Add failed operation
        metrics2 = create_operation_metrics("test_op", "id2")
        metrics2.complete(success=False, error_type="ValueError")
        collector.record_operation(metrics2)
        
        summary = collector.get_metrics_summary()
        
        assert summary["total_operations"] == 2
        assert summary["completed_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 1
        assert summary["success_rate"] == 0.5
        assert "avg_duration_ms" in summary
        assert "min_duration_ms" in summary
        assert "max_duration_ms" in summary
        assert summary["error_breakdown"]["ValueError"] == 1
    
    def test_get_metrics_summary_filtered(self):
        """Test getting filtered metrics summary."""
        collector = MetricsCollector()
        
        # Add metrics for different operations
        metrics1 = create_operation_metrics("op_a", "id1")
        metrics1.complete(success=True)
        collector.record_operation(metrics1)
        
        metrics2 = create_operation_metrics("op_b", "id2") 
        metrics2.complete(success=True)
        collector.record_operation(metrics2)
        
        metrics3 = create_operation_metrics("op_a", "id3")
        metrics3.complete(success=False, error_type="IOError")
        collector.record_operation(metrics3)
        
        # Filter for op_a only
        summary = collector.get_metrics_summary("op_a")
        
        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 1
        assert summary["error_breakdown"]["IOError"] == 1
    
    def test_get_metrics_summary_incomplete_operations(self):
        """Test summary with incomplete operations."""
        collector = MetricsCollector()
        
        # Add incomplete operation
        metrics1 = create_operation_metrics("test_op", "id1")
        # Don't complete it
        collector.record_operation(metrics1)
        
        # Add completed operation
        metrics2 = create_operation_metrics("test_op", "id2")
        metrics2.complete(success=True)
        collector.record_operation(metrics2)
        
        summary = collector.get_metrics_summary()
        
        assert summary["total_operations"] == 2
        assert summary["completed_operations"] == 1
        assert summary["successful_operations"] == 1
        assert summary["success_rate"] == 1.0
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()
        
        # Add some metrics
        metrics = create_operation_metrics("test_op", "test-id")
        collector.record_operation(metrics)
        assert len(collector.metrics) == 1
        
        # Clear metrics
        collector.clear_metrics()
        assert len(collector.metrics) == 0
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        collector = MetricsCollector()
        
        # Add more metrics than limit
        for i in range(150):
            metrics = create_operation_metrics(f"op_{i}", f"id_{i}")
            collector.record_operation(metrics)
        
        # Get recent metrics with default limit
        recent = collector.get_recent_metrics()
        assert len(recent) == 100  # Default limit
        
        # Get recent metrics with custom limit
        recent = collector.get_recent_metrics(50)
        assert len(recent) == 50
        
        # Get recent metrics when total is less than limit
        collector.clear_metrics()
        for i in range(25):
            metrics = create_operation_metrics(f"op_{i}", f"id_{i}")
            collector.record_operation(metrics)
        
        recent = collector.get_recent_metrics(100)
        assert len(recent) == 25
    
    def test_thread_safety(self):
        """Test thread safety of metrics collector."""
        collector = MetricsCollector()
        
        def worker(thread_id: int):
            for i in range(10):
                metrics = create_operation_metrics(f"op_{thread_id}_{i}", f"id_{thread_id}_{i}")
                metrics.complete(success=True)
                collector.record_operation(metrics)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(collector.metrics) == 50  # 5 threads * 10 operations each
        
        # Verify thread safety of summary
        summary = collector.get_metrics_summary()
        assert summary["total_operations"] == 50
        assert summary["successful_operations"] == 50


class TestHelperFunctions:
    """Test cases for helper functions."""
    
    def test_create_operation_metrics(self):
        """Test create_operation_metrics function."""
        metrics = create_operation_metrics("test_operation", "test-correlation-id")
        
        assert isinstance(metrics, OperationMetrics)
        assert metrics.operation_name == "test_operation"
        assert metrics.correlation_id == "test-correlation-id"
        assert metrics.start_time is not None
        assert metrics.start_time <= time.time()
    
    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns the same instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
        assert isinstance(collector1, MetricsCollector)
    
    def test_global_metrics_collector_persistence(self):
        """Test that global metrics collector persists across calls."""
        collector = get_metrics_collector()
        
        # Add some metrics
        metrics = create_operation_metrics("test_op", "test-id")
        collector.record_operation(metrics)
        
        # Get collector again and verify metrics persist
        collector2 = get_metrics_collector()
        assert len(collector2.metrics) == 1
        assert collector2.metrics[0].operation_name == "test_op"