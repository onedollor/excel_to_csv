"""Tests for correlation ID management."""

import pytest
import threading
import time
from excel_to_csv.utils.correlation import CorrelationContext


class TestCorrelationContext:
    """Test cases for CorrelationContext."""
    
    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        id1 = CorrelationContext.generate_correlation_id()
        id2 = CorrelationContext.generate_correlation_id()
        
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-123"
        
        # Initially should be None
        assert CorrelationContext.get_correlation_id() is None
        
        # Set and retrieve
        CorrelationContext.set_correlation_id(test_id)
        assert CorrelationContext.get_correlation_id() == test_id
    
    def test_ensure_correlation_id_new(self):
        """Test ensure_correlation_id creates new ID when none exists."""
        # Clear any existing correlation ID
        try:
            CorrelationContext._context.get()
        except LookupError:
            pass
        
        correlation_id = CorrelationContext.ensure_correlation_id()
        
        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
        assert CorrelationContext.get_correlation_id() == correlation_id
    
    def test_ensure_correlation_id_existing(self):
        """Test ensure_correlation_id returns existing ID."""
        existing_id = "existing-test-id"
        CorrelationContext.set_correlation_id(existing_id)
        
        correlation_id = CorrelationContext.ensure_correlation_id()
        
        assert correlation_id == existing_id
    
    def test_context_manager_new_id(self):
        """Test context manager with new correlation ID."""
        # Clear any existing correlation ID first
        original_id = CorrelationContext.get_correlation_id()
        
        with CorrelationContext() as correlation_id:
            assert correlation_id is not None
            assert CorrelationContext.get_correlation_id() == correlation_id
        
        # Should be restored to original state after context
        restored_id = CorrelationContext.get_correlation_id()
        assert restored_id == original_id
    
    def test_context_manager_provided_id(self):
        """Test context manager with provided correlation ID."""
        test_id = "provided-test-id"
        
        with CorrelationContext(test_id) as correlation_id:
            assert correlation_id == test_id
            assert CorrelationContext.get_correlation_id() == test_id
    
    def test_context_manager_nesting(self):
        """Test nested context managers."""
        outer_id = "outer-id"
        inner_id = "inner-id"
        
        with CorrelationContext(outer_id):
            assert CorrelationContext.get_correlation_id() == outer_id
            
            with CorrelationContext(inner_id):
                assert CorrelationContext.get_correlation_id() == inner_id
            
            # Should restore outer ID
            assert CorrelationContext.get_correlation_id() == outer_id
    
    def test_thread_isolation(self):
        """Test that correlation IDs are isolated between threads."""
        results = {}
        
        def worker(thread_id: int):
            correlation_id = f"thread-{thread_id}-id"
            CorrelationContext.set_correlation_id(correlation_id)
            time.sleep(0.1)  # Allow other threads to set their IDs
            results[thread_id] = CorrelationContext.get_correlation_id()
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have its own correlation ID
        assert len(results) == 5
        for thread_id, correlation_id in results.items():
            assert correlation_id == f"thread-{thread_id}-id"
    
    def test_context_manager_exception_handling(self):
        """Test context manager properly handles exceptions."""
        original_id = "original-id"
        CorrelationContext.set_correlation_id(original_id)
        
        try:
            with CorrelationContext("temp-id"):
                assert CorrelationContext.get_correlation_id() == "temp-id"
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should restore original ID even after exception
        assert CorrelationContext.get_correlation_id() == original_id
    
    def test_multiple_calls_ensure_same_id(self):
        """Test multiple ensure_correlation_id calls return same ID."""
        id1 = CorrelationContext.ensure_correlation_id()
        id2 = CorrelationContext.ensure_correlation_id()
        id3 = CorrelationContext.ensure_correlation_id()
        
        assert id1 == id2 == id3