"""Comprehensive tests for metrics tracking system targeting 90%+ coverage.

This test suite covers all aspects of the metrics system including:
- OperationMetrics lifecycle and metadata handling
- MetricsCollector aggregation and statistics
- Thread safety and concurrent operations
- Global metrics collector instance management
- Edge cases and error scenarios
"""

import pytest
import time
import threading
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from excel_to_csv.utils.metrics import (
    OperationMetrics,
    MetricsCollector,
    get_metrics_collector,
    create_operation_metrics
)


class TestOperationMetrics:
    """Test OperationMetrics class functionality."""
    
    def test_operation_metrics_initialization(self):
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
        assert metrics.metadata == {}
    
    def test_operation_metrics_complete_success(self):
        """Test completing operation with success."""
        start_time = time.time()
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=start_time
        )
        
        time.sleep(0.001)  # Small delay for measurable duration
        metrics.complete(success=True)
        
        assert metrics.success is True
        assert metrics.error_type is None
        assert metrics.end_time is not None
        assert metrics.end_time > start_time
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_operation_metrics_complete_failure(self):
        """Test completing operation with failure."""
        start_time = time.time()
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=start_time
        )
        
        time.sleep(0.001)
        metrics.complete(success=False, error_type="ValueError")
        
        assert metrics.success is False
        assert metrics.error_type == "ValueError"
        assert metrics.end_time is not None
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
    
    def test_add_metadata(self):
        """Test adding metadata to operation."""
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=time.time()
        )
        
        metrics.add_metadata("file_name", "test.xlsx")
        metrics.add_metadata("row_count", 1000)
        metrics.add_metadata("complex_data", {"nested": {"value": 42}})
        
        assert metrics.metadata["file_name"] == "test.xlsx"
        assert metrics.metadata["row_count"] == 1000
        assert metrics.metadata["complex_data"] == {"nested": {"value": 42}}
    
    def test_metadata_overwrites(self):
        """Test that metadata can be overwritten."""
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=time.time()
        )
        
        metrics.add_metadata("key", "original_value")
        assert metrics.metadata["key"] == "original_value"
        
        metrics.add_metadata("key", "new_value")
        assert metrics.metadata["key"] == "new_value"
    
    def test_to_dict_complete_metrics(self):
        """Test converting completed metrics to dictionary."""
        start_time = time.time()
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=start_time
        )
        
        metrics.add_metadata("file_name", "test.xlsx")
        metrics.add_metadata("row_count", 500)
        time.sleep(0.001)
        metrics.complete(success=True)
        
        result_dict = metrics.to_dict()
        
        assert result_dict["operation_name"] == "test_operation"
        assert result_dict["correlation_id"] == "test-id-123"
        assert result_dict["start_time"] == start_time
        assert result_dict["end_time"] is not None
        assert result_dict["duration_ms"] is not None
        assert result_dict["success"] is True
        assert result_dict["error_type"] is None
        assert result_dict["metadata"]["file_name"] == "test.xlsx"
        assert result_dict["metadata"]["row_count"] == 500
        
        # Ensure metadata is a copy, not reference
        assert result_dict["metadata"] is not metrics.metadata
    
    def test_to_dict_incomplete_metrics(self):
        """Test converting incomplete metrics to dictionary."""
        metrics = OperationMetrics(
            operation_name="incomplete_operation",
            correlation_id="test-id-456",
            start_time=time.time()
        )
        
        result_dict = metrics.to_dict()
        
        assert result_dict["operation_name"] == "incomplete_operation"
        assert result_dict["end_time"] is None
        assert result_dict["duration_ms"] is None
        assert result_dict["success"] is None
        assert result_dict["error_type"] is None
        assert result_dict["metadata"] == {}


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.collector = MetricsCollector()
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        
        assert collector.metrics == []
        assert hasattr(collector, '_lock')
    
    def test_record_operation(self):
        """Test recording operation metrics."""
        metrics = OperationMetrics(
            operation_name="test_operation",
            correlation_id="test-id-123",
            start_time=time.time()
        )
        metrics.complete(success=True)
        
        self.collector.record_operation(metrics)
        
        assert len(self.collector.metrics) == 1
        assert self.collector.metrics[0] is metrics
    
    def test_record_multiple_operations(self):
        """Test recording multiple operations."""
        metrics1 = OperationMetrics("op1", "id1", time.time())
        metrics1.complete(success=True)
        
        metrics2 = OperationMetrics("op2", "id2", time.time())
        metrics2.complete(success=False, error_type="Error")
        
        self.collector.record_operation(metrics1)
        self.collector.record_operation(metrics2)
        
        assert len(self.collector.metrics) == 2
        assert self.collector.metrics[0] is metrics1
        assert self.collector.metrics[1] is metrics2
    
    def test_get_metrics_summary_empty(self):
        """Test metrics summary with no operations."""
        summary = self.collector.get_metrics_summary()
        
        assert summary == {"total_operations": 0}
    
    def test_get_metrics_summary_basic(self):
        """Test basic metrics summary."""
        # Add successful operation
        success_metrics = OperationMetrics("success_op", "id1", time.time())
        success_metrics.complete(success=True)
        
        # Add failed operation
        failure_metrics = OperationMetrics("failure_op", "id2", time.time())
        failure_metrics.complete(success=False, error_type="ValueError")
        
        # Add incomplete operation
        incomplete_metrics = OperationMetrics("incomplete_op", "id3", time.time())
        
        self.collector.record_operation(success_metrics)
        self.collector.record_operation(failure_metrics)
        self.collector.record_operation(incomplete_metrics)
        
        summary = self.collector.get_metrics_summary()
        
        assert summary["total_operations"] == 3
        assert summary["completed_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["failed_operations"] == 1
        assert summary["success_rate"] == 0.5  # 1 success / 2 completed
    
    def test_get_metrics_summary_with_durations(self):
        """Test metrics summary includes duration statistics."""
        # Create metrics with known durations
        metrics1 = OperationMetrics("op1", "id1", 100.0)  # start_time
        metrics1.end_time = 100.1  # end_time
        metrics1.duration_ms = 100.0  # 100ms
        metrics1.success = True
        
        metrics2 = OperationMetrics("op2", "id2", 200.0)
        metrics2.end_time = 200.05
        metrics2.duration_ms = 50.0  # 50ms
        metrics2.success = True
        
        metrics3 = OperationMetrics("op3", "id3", 300.0)
        metrics3.end_time = 300.2
        metrics3.duration_ms = 200.0  # 200ms
        metrics3.success = True
        
        self.collector.record_operation(metrics1)
        self.collector.record_operation(metrics2)
        self.collector.record_operation(metrics3)
        
        summary = self.collector.get_metrics_summary()
        
        assert summary["avg_duration_ms"] == 116.66666666666667  # (100+50+200)/3
        assert summary["min_duration_ms"] == 50.0
        assert summary["max_duration_ms"] == 200.0
        assert summary["total_duration_ms"] == 350.0
    
    def test_get_metrics_summary_with_errors(self):
        """Test metrics summary includes error breakdown."""
        # Different error types
        error1 = OperationMetrics("op1", "id1", time.time())
        error1.complete(success=False, error_type="ValueError")
        
        error2 = OperationMetrics("op2", "id2", time.time())
        error2.complete(success=False, error_type="ValueError")
        
        error3 = OperationMetrics("op3", "id3", time.time())
        error3.complete(success=False, error_type="IOError")
        
        error4 = OperationMetrics("op4", "id4", time.time())
        error4.complete(success=False, error_type=None)  # Unknown error
        
        self.collector.record_operation(error1)
        self.collector.record_operation(error2)
        self.collector.record_operation(error3)
        self.collector.record_operation(error4)
        
        summary = self.collector.get_metrics_summary()
        
        assert "error_breakdown" in summary
        assert summary["error_breakdown"]["ValueError"] == 2
        assert summary["error_breakdown"]["IOError"] == 1
        assert summary["error_breakdown"]["Unknown"] == 1
    
    def test_get_metrics_summary_filtered_by_operation(self):
        """Test filtering metrics summary by operation name."""
        # Add operations with different names
        op1_metrics = OperationMetrics("target_operation", "id1", time.time())
        op1_metrics.complete(success=True)
        
        op2_metrics = OperationMetrics("other_operation", "id2", time.time())
        op2_metrics.complete(success=False, error_type="Error")
        
        op3_metrics = OperationMetrics("target_operation", "id3", time.time())
        op3_metrics.complete(success=True)
        
        self.collector.record_operation(op1_metrics)
        self.collector.record_operation(op2_metrics)
        self.collector.record_operation(op3_metrics)
        
        # Filter by target_operation
        summary = self.collector.get_metrics_summary("target_operation")
        
        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 2
        assert summary["failed_operations"] == 0
        assert summary["success_rate"] == 1.0
    
    def test_get_metrics_summary_filtered_no_matches(self):
        """Test filtering with no matching operations."""
        metrics = OperationMetrics("some_operation", "id1", time.time())
        metrics.complete(success=True)
        self.collector.record_operation(metrics)
        
        summary = self.collector.get_metrics_summary("nonexistent_operation")
        
        assert summary == {"total_operations": 0}
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        # Add some metrics
        metrics1 = OperationMetrics("op1", "id1", time.time())
        metrics2 = OperationMetrics("op2", "id2", time.time())
        
        self.collector.record_operation(metrics1)
        self.collector.record_operation(metrics2)
        
        assert len(self.collector.metrics) == 2
        
        self.collector.clear_metrics()
        
        assert len(self.collector.metrics) == 0
    
    def test_get_recent_metrics_within_limit(self):
        """Test getting recent metrics within limit."""
        # Add metrics
        for i in range(5):
            metrics = OperationMetrics(f"op{i}", f"id{i}", time.time())
            self.collector.record_operation(metrics)
        
        recent = self.collector.get_recent_metrics(limit=3)
        
        assert len(recent) == 3
        # Should get the last 3 metrics
        assert recent[0].operation_name == "op2"
        assert recent[1].operation_name == "op3"
        assert recent[2].operation_name == "op4"
        
        # Should be copies, not references
        assert recent is not self.collector.metrics[-3:]
    
    def test_get_recent_metrics_exceeds_limit(self):
        """Test getting recent metrics when total is less than limit."""
        # Add only 2 metrics
        metrics1 = OperationMetrics("op1", "id1", time.time())
        metrics2 = OperationMetrics("op2", "id2", time.time())
        
        self.collector.record_operation(metrics1)
        self.collector.record_operation(metrics2)
        
        recent = self.collector.get_recent_metrics(limit=5)
        
        assert len(recent) == 2
        assert recent[0] is not metrics1  # Should be copy
        assert recent[1] is not metrics2  # Should be copy
        assert recent[0].operation_name == "op1"
        assert recent[1].operation_name == "op2"
    
    def test_get_recent_metrics_empty_collector(self):
        """Test getting recent metrics from empty collector."""
        recent = self.collector.get_recent_metrics(limit=10)
        
        assert recent == []


class TestThreadSafety:
    """Test thread safety of MetricsCollector."""
    
    def test_concurrent_recording(self):
        """Test concurrent operation recording."""
        collector = MetricsCollector()
        
        def record_operations(start_index, count):
            for i in range(count):
                metrics = OperationMetrics(
                    f"op_{start_index}_{i}",
                    f"id_{start_index}_{i}",
                    time.time()
                )
                metrics.complete(success=True)
                collector.record_operation(metrics)
        
        # Use multiple threads to record operations
        threads = []
        operations_per_thread = 50
        thread_count = 4
        
        for thread_id in range(thread_count):
            thread = threading.Thread(
                target=record_operations,
                args=(thread_id, operations_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have all operations recorded
        expected_total = thread_count * operations_per_thread
        assert len(collector.metrics) == expected_total
        
        # All operations should be unique
        operation_names = [m.operation_name for m in collector.metrics]
        assert len(set(operation_names)) == expected_total
    
    def test_concurrent_summary_generation(self):
        """Test concurrent summary generation while recording."""
        collector = MetricsCollector()
        
        # Pre-populate with some metrics
        for i in range(10):
            metrics = OperationMetrics(f"initial_op_{i}", f"initial_id_{i}", time.time())
            metrics.complete(success=True)
            collector.record_operation(metrics)
        
        summaries = []
        
        def generate_summaries():
            for _ in range(10):
                summary = collector.get_metrics_summary()
                summaries.append(summary)
                time.sleep(0.001)
        
        def record_more_operations():
            for i in range(20):
                metrics = OperationMetrics(f"concurrent_op_{i}", f"concurrent_id_{i}", time.time())
                metrics.complete(success=True)
                collector.record_operation(metrics)
                time.sleep(0.001)
        
        # Run both operations concurrently
        thread1 = threading.Thread(target=generate_summaries)
        thread2 = threading.Thread(target=record_more_operations)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Should have generated summaries without errors
        assert len(summaries) == 10
        # All summaries should be valid
        for summary in summaries:
            assert "total_operations" in summary
            assert summary["total_operations"] >= 10  # At least initial operations


class TestGlobalMetricsCollector:
    """Test global metrics collector functionality."""
    
    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns the same instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
        assert isinstance(collector1, MetricsCollector)
    
    def test_global_collector_persistence(self):
        """Test that global collector persists data across calls."""
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        # Add a metric
        metrics = OperationMetrics("persistent_test", "test_id", time.time())
        metrics.complete(success=True)
        collector.record_operation(metrics)
        
        # Get collector again and verify metric is still there
        collector2 = get_metrics_collector()
        assert len(collector2.metrics) == initial_count + 1
        assert collector2.metrics[-1].operation_name == "persistent_test"
    
    def test_create_operation_metrics_function(self):
        """Test create_operation_metrics convenience function."""
        start_time_before = time.time()
        metrics = create_operation_metrics("test_operation", "test_correlation_id")
        start_time_after = time.time()
        
        assert isinstance(metrics, OperationMetrics)
        assert metrics.operation_name == "test_operation"
        assert metrics.correlation_id == "test_correlation_id"
        assert start_time_before <= metrics.start_time <= start_time_after
        assert metrics.end_time is None
        assert metrics.success is None
        assert metrics.metadata == {}


class TestEdgeCasesAndErrorScenarios:
    """Test edge cases and error scenarios."""
    
    def test_metrics_with_zero_duration(self):
        """Test metrics with zero or very small duration."""
        metrics = OperationMetrics("instant_op", "id", time.time())
        # Complete immediately
        metrics.complete(success=True)
        
        # Duration might be 0 or very small
        assert metrics.duration_ms is not None
        assert metrics.duration_ms >= 0
    
    def test_metrics_with_negative_duration(self):
        """Test handling of negative duration (clock adjustments)."""
        # Simulate clock adjustment by setting end time before start time
        metrics = OperationMetrics("clock_adjustment_op", "id", 1000.0)
        metrics.end_time = 999.0  # Before start time
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.success = True
        
        # Should handle negative duration gracefully
        assert metrics.duration_ms < 0
        
        collector = MetricsCollector()
        collector.record_operation(metrics)
        
        summary = collector.get_metrics_summary()
        # Summary should still be generated without errors
        assert summary["total_operations"] == 1
        assert summary["successful_operations"] == 1
    
    def test_metadata_with_complex_types(self):
        """Test metadata with complex data types."""
        metrics = OperationMetrics("complex_metadata", "id", time.time())
        
        # Add various complex types
        metrics.add_metadata("list_data", [1, 2, 3, {"nested": "value"}])
        metrics.add_metadata("dict_data", {"key": "value", "nested": {"deep": "data"}})
        metrics.add_metadata("none_value", None)
        metrics.add_metadata("bool_value", True)
        metrics.add_metadata("float_value", 3.14159)
        
        # Should handle all types
        result_dict = metrics.to_dict()
        assert result_dict["metadata"]["list_data"] == [1, 2, 3, {"nested": "value"}]
        assert result_dict["metadata"]["dict_data"]["nested"]["deep"] == "data"
        assert result_dict["metadata"]["none_value"] is None
        assert result_dict["metadata"]["bool_value"] is True
        assert result_dict["metadata"]["float_value"] == 3.14159
    
    def test_large_number_of_metrics(self):
        """Test handling large number of metrics."""
        collector = MetricsCollector()
        
        # Add many metrics
        count = 1000
        for i in range(count):
            metrics = OperationMetrics(f"bulk_op_{i}", f"bulk_id_{i}", time.time())
            metrics.complete(success=i % 2 == 0)  # Alternate success/failure
            collector.record_operation(metrics)
        
        # Should handle large dataset
        summary = collector.get_metrics_summary()
        assert summary["total_operations"] == count
        assert summary["successful_operations"] == count // 2
        assert summary["failed_operations"] == count // 2
        
        # Recent metrics should work
        recent = collector.get_recent_metrics(limit=10)
        assert len(recent) == 10
    
    def test_empty_string_operation_names(self):
        """Test handling of empty or unusual operation names."""
        collector = MetricsCollector()
        
        # Test various edge case names
        edge_cases = ["", " ", "\n", "\t", "very_long_" + "x" * 100]
        
        for name in edge_cases:
            metrics = OperationMetrics(name, "id", time.time())
            metrics.complete(success=True)
            collector.record_operation(metrics)
        
        summary = collector.get_metrics_summary()
        assert summary["total_operations"] == len(edge_cases)
        
        # Test filtering by edge case names
        for name in edge_cases:
            filtered_summary = collector.get_metrics_summary(name)
            assert filtered_summary["total_operations"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])