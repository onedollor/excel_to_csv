"""Performance tests for archiving functionality."""

import time
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any
import concurrent.futures
from unittest.mock import Mock, patch

from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.models.data_models import ArchiveConfig, ArchiveResult


class TestArchivingPerformance:
    """Performance test cases for archiving functionality."""
    
    @pytest.fixture
    def archive_manager(self):
        """Create ArchiveManager instance for testing."""
        return ArchiveManager()
    
    @pytest.fixture
    def archive_config(self):
        """Create default archive configuration for testing."""
        return ArchiveConfig(
            enabled=True,
            archive_folder_name="archive",
            timestamp_format="%Y%m%d_%H%M%S",
            handle_conflicts=True,
            preserve_structure=True
        )
    
    @pytest.fixture
    def test_files_small(self, tmp_path):
        """Create small test files for performance testing."""
        files = []
        for i in range(10):
            test_file = tmp_path / f"small_file_{i}.xlsx"
            test_file.write_text("A" * 1024)  # 1KB files
            files.append(test_file)
        return files
    
    @pytest.fixture
    def test_files_medium(self, tmp_path):
        """Create medium test files for performance testing."""
        files = []
        for i in range(5):
            test_file = tmp_path / f"medium_file_{i}.xlsx"
            test_file.write_text("A" * (1024 * 1024))  # 1MB files
            files.append(test_file)
        return files
    
    @pytest.fixture
    def test_file_large(self, tmp_path):
        """Create large test file for performance testing."""
        test_file = tmp_path / "large_file.xlsx"
        test_file.write_text("A" * (10 * 1024 * 1024))  # 10MB file
        return test_file
    
    def test_single_file_archiving_overhead(self, archive_manager, archive_config, test_file_large):
        """Test archiving overhead is less than 5% for single file operations."""
        # Baseline: Time file copy without archiving
        baseline_file = test_file_large.parent / "baseline_copy.xlsx"
        
        start_time = time.perf_counter()
        baseline_file.write_bytes(test_file_large.read_bytes())
        baseline_time = time.perf_counter() - start_time
        baseline_file.unlink()  # Clean up
        
        # Archiving: Time file archiving operation
        start_time = time.perf_counter()
        result = archive_manager.archive_file(test_file_large, archive_config)
        archiving_time = time.perf_counter() - start_time
        
        # Verify successful archiving
        assert result.success
        assert result.archived_path.exists()
        
        # Calculate overhead percentage
        overhead_percentage = ((archiving_time - baseline_time) / baseline_time) * 100
        
        # Verify overhead is less than 5%
        assert overhead_percentage < 5.0, f"Archiving overhead {overhead_percentage:.2f}% exceeds 5% limit"
        
        print(f"Baseline time: {baseline_time:.4f}s, Archiving time: {archiving_time:.4f}s, Overhead: {overhead_percentage:.2f}%")
    
    def test_multiple_files_archiving_performance(self, archive_manager, archive_config, test_files_medium):
        """Test archiving performance with multiple medium-sized files."""
        start_time = time.perf_counter()
        
        results = []
        for test_file in test_files_medium:
            result = archive_manager.archive_file(test_file, archive_config)
            results.append(result)
        
        total_time = time.perf_counter() - start_time
        
        # Verify all files were archived successfully
        assert all(result.success for result in results)
        
        # Calculate average time per file
        avg_time_per_file = total_time / len(test_files_medium)
        
        # Performance expectation: should handle medium files (1MB) in under 0.5 seconds each
        assert avg_time_per_file < 0.5, f"Average archiving time {avg_time_per_file:.4f}s per file exceeds 0.5s limit"
        
        print(f"Total time: {total_time:.4f}s, Average per file: {avg_time_per_file:.4f}s")
    
    def test_concurrent_archiving_performance(self, archive_manager, archive_config, test_files_small):
        """Test concurrent archiving performance."""
        def archive_single_file(file_path):
            """Archive a single file and return result with timing."""
            start = time.perf_counter()
            result = archive_manager.archive_file(file_path, archive_config)
            end = time.perf_counter()
            return result, end - start
        
        # Sequential archiving baseline
        start_time = time.perf_counter()
        sequential_results = []
        for test_file in test_files_small:
            result, duration = archive_single_file(test_file)
            sequential_results.append((result, duration))
        sequential_total_time = time.perf_counter() - start_time
        
        # Create fresh test files for concurrent test
        concurrent_test_files = []
        for i, original_file in enumerate(test_files_small):
            concurrent_file = original_file.parent / f"concurrent_{i}.xlsx"
            concurrent_file.write_bytes(original_file.read_bytes())
            concurrent_test_files.append(concurrent_file)
        
        # Concurrent archiving
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_futures = [
                executor.submit(archive_single_file, test_file) 
                for test_file in concurrent_test_files
            ]
            concurrent_results = [future.result() for future in concurrent_futures]
        concurrent_total_time = time.perf_counter() - start_time
        
        # Verify all operations succeeded
        assert all(result[0].success for result in sequential_results)
        assert all(result[0].success for result in concurrent_results)
        
        # Concurrent should be faster than sequential for multiple files
        speedup_ratio = sequential_total_time / concurrent_total_time
        assert speedup_ratio > 1.5, f"Concurrent archiving speedup {speedup_ratio:.2f}x is less than expected 1.5x"
        
        print(f"Sequential time: {sequential_total_time:.4f}s, Concurrent time: {concurrent_total_time:.4f}s, Speedup: {speedup_ratio:.2f}x")
    
    def test_archiving_memory_usage(self, archive_manager, archive_config, test_file_large):
        """Test that archiving doesn't consume excessive memory."""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Get baseline memory usage
        baseline_snapshot = tracemalloc.take_snapshot()
        
        # Perform archiving operation
        result = archive_manager.archive_file(test_file_large, archive_config)
        
        # Get memory usage after archiving
        final_snapshot = tracemalloc.take_snapshot()
        
        # Stop memory tracing
        tracemalloc.stop()
        
        # Verify archiving succeeded
        assert result.success
        
        # Calculate memory increase
        top_stats = final_snapshot.compare_to(baseline_snapshot, 'lineno')
        total_memory_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        
        # File size for comparison
        file_size = test_file_large.stat().st_size
        
        # Memory usage should be reasonable relative to file size
        memory_ratio = total_memory_increase / file_size
        assert memory_ratio < 0.1, f"Memory usage ratio {memory_ratio:.4f} exceeds 10% of file size"
        
        print(f"File size: {file_size} bytes, Memory increase: {total_memory_increase} bytes, Ratio: {memory_ratio:.4f}")
    
    def test_archive_folder_creation_performance(self, archive_manager, tmp_path):
        """Test performance of archive folder creation operations."""
        archive_config = ArchiveConfig(
            enabled=True,
            archive_folder_name="performance_test_archive",
            preserve_structure=True
        )
        
        # Create nested directory structure
        nested_input_path = tmp_path / "level1" / "level2" / "level3"
        nested_input_path.mkdir(parents=True, exist_ok=True)
        
        test_file = nested_input_path / "nested_test.xlsx"
        test_file.write_text("test content")
        
        # Time archive folder creation
        start_time = time.perf_counter()
        result = archive_manager.archive_file(test_file, archive_config)
        creation_time = time.perf_counter() - start_time
        
        # Verify success
        assert result.success
        
        # Archive folder creation should be fast (under 0.1 seconds)
        assert creation_time < 0.1, f"Archive folder creation time {creation_time:.4f}s exceeds 0.1s limit"
        
        print(f"Archive folder creation time: {creation_time:.4f}s")
    
    def test_conflict_resolution_performance(self, archive_manager, archive_config, tmp_path):
        """Test performance of naming conflict resolution."""
        # Create test file
        test_file = tmp_path / "conflict_test.xlsx"
        test_file.write_text("test content")
        
        # Pre-create archive folder with existing file
        archive_folder = tmp_path / archive_config.archive_folder_name
        archive_folder.mkdir(exist_ok=True)
        existing_file = archive_folder / "conflict_test.xlsx"
        existing_file.write_text("existing content")
        
        # Time conflict resolution
        start_time = time.perf_counter()
        result = archive_manager.archive_file(test_file, archive_config)
        resolution_time = time.perf_counter() - start_time
        
        # Verify success and conflict was resolved
        assert result.success
        assert result.archived_path != existing_file  # Should have different name
        assert result.archived_path.exists()
        assert existing_file.exists()  # Original should still exist
        
        # Conflict resolution should be fast (under 0.05 seconds)
        assert resolution_time < 0.05, f"Conflict resolution time {resolution_time:.4f}s exceeds 0.05s limit"
        
        print(f"Conflict resolution time: {resolution_time:.4f}s")
    
    def test_batch_archiving_scalability(self, archive_manager, archive_config, tmp_path):
        """Test scalability of batch archiving operations."""
        # Create varying batch sizes
        batch_sizes = [10, 50, 100]
        timing_results = {}
        
        for batch_size in batch_sizes:
            # Create batch of files
            test_files = []
            for i in range(batch_size):
                test_file = tmp_path / f"batch_{batch_size}" / f"file_{i}.xlsx"
                test_file.parent.mkdir(exist_ok=True)
                test_file.write_text("A" * (1024 * 10))  # 10KB files
                test_files.append(test_file)
            
            # Time batch archiving
            start_time = time.perf_counter()
            results = []
            for test_file in test_files:
                result = archive_manager.archive_file(test_file, archive_config)
                results.append(result)
            batch_time = time.perf_counter() - start_time
            
            # Verify all succeeded
            assert all(result.success for result in results)
            
            # Calculate time per file
            time_per_file = batch_time / batch_size
            timing_results[batch_size] = time_per_file
            
            print(f"Batch size {batch_size}: {batch_time:.4f}s total, {time_per_file:.6f}s per file")
        
        # Verify scalability: time per file shouldn't increase significantly with batch size
        time_10 = timing_results[10]
        time_100 = timing_results[100]
        scalability_ratio = time_100 / time_10
        
        # Time per file shouldn't more than double when going from 10 to 100 files
        assert scalability_ratio < 2.0, f"Scalability ratio {scalability_ratio:.2f} exceeds 2.0x limit"
        
        print(f"Scalability ratio (100 vs 10 files): {scalability_ratio:.2f}x")
    
    @pytest.mark.parametrize("file_size_mb", [1, 5, 10, 25])
    def test_archiving_performance_by_file_size(self, archive_manager, archive_config, tmp_path, file_size_mb):
        """Test archiving performance across different file sizes."""
        # Create test file of specified size
        test_file = tmp_path / f"size_test_{file_size_mb}mb.xlsx"
        test_file.write_text("A" * (file_size_mb * 1024 * 1024))
        
        # Time archiving operation
        start_time = time.perf_counter()
        result = archive_manager.archive_file(test_file, archive_config)
        archiving_time = time.perf_counter() - start_time
        
        # Verify success
        assert result.success
        
        # Calculate throughput (MB/s)
        throughput = file_size_mb / archiving_time
        
        # Performance expectation: should maintain reasonable throughput
        # (This is a rough estimate, actual values will vary by system)
        min_throughput = 50  # MB/s
        assert throughput > min_throughput, f"Throughput {throughput:.2f} MB/s is below minimum {min_throughput} MB/s"
        
        print(f"File size: {file_size_mb}MB, Time: {archiving_time:.4f}s, Throughput: {throughput:.2f} MB/s")