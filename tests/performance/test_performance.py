"""Performance tests for Excel-to-CSV converter.

This module contains performance benchmarks and stress tests to ensure
the system meets performance requirements specified in the design.

Performance Requirements:
- Process 50MB+ Excel files within reasonable time limits
- Handle concurrent file processing efficiently  
- Maintain stable memory usage during bulk processing
- Scale to multiple monitored directories
"""

import pytest
import time
import threading
import psutil
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.models.data_models import Config
from excel_to_csv.processors.excel_processor import ExcelProcessor
from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
from excel_to_csv.generators.csv_generator import CSVGenerator
from excel_to_csv.monitoring.file_monitor import FileMonitor


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.slow,  # These tests take longer to run
]


class PerformanceTracker:
    """Utility class to track performance metrics during tests."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None
    
    def start(self):
        """Start performance tracking."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def stop(self):
        """Stop performance tracking."""
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    @property
    def duration(self):
        """Get processing duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def memory_increase(self):
        """Get memory increase during processing."""
        if self.start_memory and self.end_memory:
            return self.end_memory - self.start_memory
        return None
    
    @property
    def peak_memory_increase(self):
        """Get peak memory increase during processing."""
        if self.start_memory and self.peak_memory:
            return self.peak_memory - self.start_memory
        return None


class TestFileGenerator:
    """Utility class to generate test files of various sizes."""
    
    @staticmethod
    def create_large_excel_file(file_path: Path, target_size_mb: float = 50):
        """Create large Excel file for performance testing.
        
        Args:
            file_path: Path where to create the file
            target_size_mb: Target file size in megabytes
        """
        # Estimate rows needed for target size
        # Rough estimate: 1MB ≈ 10,000 rows with 10 columns
        estimated_rows = int(target_size_mb * 10000)
        cols_count = 10
        
        # Generate data in chunks to manage memory
        chunk_size = 10000
        total_data = []
        
        for i in range(0, estimated_rows, chunk_size):
            current_chunk_size = min(chunk_size, estimated_rows - i)
            chunk_data = {
                'ID': range(i, i + current_chunk_size),
                'Name': [f'Item_{j}' for j in range(i, i + current_chunk_size)],
                'Category': ['Category_A', 'Category_B', 'Category_C'] * (current_chunk_size // 3 + 1),
                'Value': np.random.uniform(0, 1000, current_chunk_size),
                'Description': [f'Description for item {j} with some longer text content' 
                               for j in range(i, i + current_chunk_size)],
                'Date': pd.date_range('2023-01-01', periods=current_chunk_size, freq='D'),
                'Score': np.random.uniform(0, 100, current_chunk_size),
                'Active': [True, False] * (current_chunk_size // 2 + 1),
                'Tags': [f'tag1,tag2,tag3_{j%100}' for j in range(i, i + current_chunk_size)],
                'Notes': [f'Additional notes for record {j}' * 3 
                         for j in range(i, i + current_chunk_size)]
            }
            
            # Trim lists to exact size
            for key, values in chunk_data.items():
                chunk_data[key] = values[:current_chunk_size]
            
            chunk_df = pd.DataFrame(chunk_data)
            total_data.append(chunk_df)
        
        # Combine all chunks
        full_df = pd.concat(total_data, ignore_index=True)
        
        # Write to Excel file
        full_df.to_excel(file_path, index=False)
        
        # Return actual file size
        actual_size_mb = file_path.stat().st_size / 1024 / 1024
        return actual_size_mb, len(full_df)
    
    @staticmethod
    def create_multi_sheet_file(file_path: Path, num_sheets: int = 5):
        """Create Excel file with multiple worksheets."""
        with pd.ExcelWriter(file_path) as writer:
            for i in range(num_sheets):
                data = pd.DataFrame({
                    'Column_A': range(1000 * i, 1000 * (i + 1)),
                    'Column_B': [f'Data_{j}' for j in range(1000)],
                    'Column_C': np.random.uniform(0, 100, 1000)
                })
                data.to_excel(writer, sheet_name=f'Sheet_{i+1}', index=False)
        
        return file_path.stat().st_size / 1024 / 1024  # Size in MB


@pytest.fixture
def performance_tracker():
    """Fixture to provide performance tracking."""
    return PerformanceTracker()


@pytest.fixture
def large_excel_file(temp_dir: Path):
    """Fixture to create a large Excel file for testing."""
    file_path = temp_dir / "large_test_file.xlsx"
    actual_size, row_count = TestFileGenerator.create_large_excel_file(file_path, 50.0)
    
    return {
        'path': file_path,
        'size_mb': actual_size,
        'row_count': row_count
    }


@pytest.fixture
def multi_sheet_file(temp_dir: Path):
    """Fixture to create multi-sheet Excel file."""
    file_path = temp_dir / "multi_sheet_test.xlsx"
    size_mb = TestFileGenerator.create_multi_sheet_file(file_path, 8)
    
    return {
        'path': file_path,
        'size_mb': size_mb,
        'sheet_count': 8
    }


class TestLargeFileProcessing:
    """Test performance with large Excel files."""
    
    def test_large_file_processing_time(self, temp_dir: Path, large_excel_file: Dict, 
                                      performance_tracker: PerformanceTracker):
        """Test that large files are processed within acceptable time limits."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,  # Lower threshold for performance testing
            max_concurrent=1
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Track performance
        performance_tracker.start()
        
        result = converter.process_single_file(large_excel_file['path'])
        
        performance_tracker.stop()
        
        # Assertions
        assert result is not None
        assert result.success, f"Processing failed: {result.errors}"
        
        # Performance requirements
        max_processing_time = 300  # 5 minutes for 50MB+ file
        assert performance_tracker.duration < max_processing_time, \
            f"Processing took {performance_tracker.duration:.2f}s, expected < {max_processing_time}s"
        
        # Memory usage should be reasonable
        max_memory_increase = 500  # 500MB increase
        assert performance_tracker.peak_memory_increase < max_memory_increase, \
            f"Peak memory increase {performance_tracker.peak_memory_increase:.2f}MB, expected < {max_memory_increase}MB"
        
        print(f"\nLarge file performance metrics:")
        print(f"  File size: {large_excel_file['size_mb']:.2f} MB")
        print(f"  Row count: {large_excel_file['row_count']:,}")
        print(f"  Processing time: {performance_tracker.duration:.2f} seconds")
        print(f"  Peak memory increase: {performance_tracker.peak_memory_increase:.2f} MB")
    
    def test_excel_processor_large_file(self, large_excel_file: Dict, 
                                      performance_tracker: PerformanceTracker):
        """Test Excel processor with large files."""
        processor = ExcelProcessor()
        
        performance_tracker.start()
        
        worksheets = processor.process_excel_file(large_excel_file['path'])
        
        performance_tracker.stop()
        
        assert len(worksheets) > 0
        assert worksheets[0].row_count > 10000  # Should have substantial data
        
        # Should complete within reasonable time
        max_time = 60  # 1 minute for Excel reading
        assert performance_tracker.duration < max_time, \
            f"Excel processing took {performance_tracker.duration:.2f}s, expected < {max_time}s"
        
        print(f"\nExcel processor performance:")
        print(f"  Processing time: {performance_tracker.duration:.2f} seconds")
        print(f"  Memory increase: {performance_tracker.memory_increase:.2f} MB")


class TestConcurrentProcessing:
    """Test performance with concurrent file processing."""
    
    def test_concurrent_file_processing(self, temp_dir: Path, 
                                      performance_tracker: PerformanceTracker):
        """Test processing multiple files concurrently."""
        output_dir = temp_dir / "concurrent_output"
        output_dir.mkdir()
        
        # Create multiple smaller test files
        test_files = []
        for i in range(5):
            file_path = temp_dir / f"concurrent_test_{i}.xlsx"
            # Create smaller files for concurrent testing
            actual_size, _ = TestFileGenerator.create_large_excel_file(file_path, 5.0)
            test_files.append(file_path)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.3,
            max_concurrent=3  # Test concurrent processing
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        performance_tracker.start()
        
        # Process files concurrently using ThreadPoolExecutor
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(converter.process_single_file, file_path) 
                      for file_path in test_files]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                performance_tracker.update_peak_memory()
        
        performance_tracker.stop()
        
        # All files should be processed successfully
        successful_results = [r for r in results if r and r.success]
        assert len(successful_results) >= len(test_files) - 1  # Allow one potential failure
        
        # Concurrent processing should be faster than sequential
        max_concurrent_time = 180  # 3 minutes for 5 files concurrently
        assert performance_tracker.duration < max_concurrent_time, \
            f"Concurrent processing took {performance_tracker.duration:.2f}s, expected < {max_concurrent_time}s"
        
        print(f"\nConcurrent processing performance:")
        print(f"  Files processed: {len(successful_results)}")
        print(f"  Total time: {performance_tracker.duration:.2f} seconds")
        print(f"  Average time per file: {performance_tracker.duration/len(test_files):.2f} seconds")
        print(f"  Peak memory increase: {performance_tracker.peak_memory_increase:.2f} MB")
    
    def test_file_monitor_multiple_directories(self, temp_dir: Path, 
                                             performance_tracker: PerformanceTracker):
        """Test file monitoring performance with multiple directories."""
        num_directories = 10
        directories = []
        
        # Create multiple directories
        for i in range(num_directories):
            dir_path = temp_dir / f"monitor_dir_{i}"
            dir_path.mkdir()
            directories.append(dir_path)
        
        callback_count = 0
        
        def test_callback(file_path):
            nonlocal callback_count
            callback_count += 1
        
        monitor = FileMonitor(directories, test_callback, debounce_seconds=0.1)
        
        performance_tracker.start()
        
        try:
            monitor.start_monitoring()
            
            # Create files in different directories
            for i, directory in enumerate(directories):
                test_file = directory / f"test_{i}.xlsx"
                # Create small test file
                pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}).to_excel(test_file, index=False)
            
            # Wait for file detection
            time.sleep(2.0)
            
            performance_tracker.stop()
            
            # Should detect most files
            assert callback_count >= num_directories - 2  # Allow some missed events
            
            # Should complete quickly
            max_monitoring_setup = 10  # 10 seconds for setup and detection
            assert performance_tracker.duration < max_monitoring_setup, \
                f"Monitoring setup took {performance_tracker.duration:.2f}s, expected < {max_monitoring_setup}s"
        
        finally:
            monitor.stop_monitoring()
        
        print(f"\nMultiple directory monitoring performance:")
        print(f"  Directories monitored: {num_directories}")
        print(f"  Files detected: {callback_count}")
        print(f"  Setup time: {performance_tracker.duration:.2f} seconds")


class TestMemoryUsage:
    """Test memory usage patterns during processing."""
    
    def test_memory_stability_bulk_processing(self, temp_dir: Path, 
                                            performance_tracker: PerformanceTracker):
        """Test memory stability during bulk file processing."""
        output_dir = temp_dir / "bulk_output"
        output_dir.mkdir()
        
        # Create multiple medium-sized files
        test_files = []
        for i in range(10):
            file_path = temp_dir / f"bulk_test_{i}.xlsx"
            # Create 5MB files for bulk testing
            TestFileGenerator.create_large_excel_file(file_path, 5.0)
            test_files.append(file_path)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.3,
            max_concurrent=2
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        performance_tracker.start()
        memory_samples = []
        
        # Process files one by one and track memory
        for i, file_path in enumerate(test_files):
            result = converter.process_single_file(file_path)
            assert result.success, f"File {i} processing failed"
            
            # Sample memory usage
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            performance_tracker.update_peak_memory()
        
        performance_tracker.stop()
        
        # Memory should not grow excessively
        initial_memory = memory_samples[0]
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        
        max_memory_growth = 200  # 200MB growth for 10 files
        assert memory_growth < max_memory_growth, \
            f"Memory grew by {memory_growth:.2f}MB, expected < {max_memory_growth}MB"
        
        # Memory usage should be relatively stable (not constantly growing)
        memory_variance = np.var(memory_samples)
        max_variance = 1000  # Memory variance should be reasonable
        assert memory_variance < max_variance, \
            f"Memory variance {memory_variance:.2f} too high, expected < {max_variance}"
        
        print(f"\nBulk processing memory stability:")
        print(f"  Files processed: {len(test_files)}")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory growth: {memory_growth:.2f} MB")
        print(f"  Peak memory: {performance_tracker.peak_memory:.2f} MB")
        print(f"  Memory variance: {memory_variance:.2f}")
    
    def test_confidence_analyzer_memory_efficiency(self, temp_dir: Path):
        """Test confidence analyzer memory efficiency with large datasets."""
        analyzer = ConfidenceAnalyzer()
        
        # Create large dataset in memory
        large_data = pd.DataFrame({
            'ID': range(100000),
            'Value': np.random.uniform(0, 1000, 100000),
            'Category': ['A', 'B', 'C'] * 33334,
            'Description': [f'Description_{i}' for i in range(100000)]
        })
        
        worksheet_data = type('WorksheetData', (), {
            'name': 'LargeSheet',
            'data': large_data,
            'file_path': temp_dir / 'large.xlsx',
            'row_count': 100000,
            'column_count': 4
        })()
        
        # Measure memory before analysis
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Analyze confidence
        start_time = time.time()
        confidence = analyzer.analyze_worksheet(worksheet_data)
        end_time = time.time()
        
        # Measure memory after analysis
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        # Analysis should complete quickly and use minimal additional memory
        max_analysis_time = 30  # 30 seconds for 100k rows
        assert end_time - start_time < max_analysis_time, \
            f"Analysis took {end_time - start_time:.2f}s, expected < {max_analysis_time}s"
        
        max_memory_increase = 100  # 100MB increase
        assert memory_increase < max_memory_increase, \
            f"Memory increased by {memory_increase:.2f}MB, expected < {max_memory_increase}MB"
        
        assert isinstance(confidence.overall_score, (int, float))
        assert 0 <= confidence.overall_score <= 1
        
        print(f"\nConfidence analyzer performance:")
        print(f"  Rows analyzed: 100,000")
        print(f"  Analysis time: {end_time - start_time:.2f} seconds")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Confidence score: {confidence.overall_score:.3f}")


class TestScalabilityLimits:
    """Test system behavior at scale limits."""
    
    @pytest.mark.slow
    def test_very_large_file_handling(self, temp_dir: Path):
        """Test handling of very large Excel files (100MB+)."""
        # Create very large file (this test is marked as slow)
        large_file_path = temp_dir / "very_large.xlsx"
        
        # Create 100MB file
        actual_size, row_count = TestFileGenerator.create_large_excel_file(large_file_path, 100.0)
        
        processor = ExcelProcessor(max_file_size_mb=150)  # Allow large files
        
        start_time = time.time()
        
        # This might fail on systems with limited memory - that's expected
        try:
            worksheets = processor.process_excel_file(large_file_path)
            end_time = time.time()
            
            assert len(worksheets) > 0
            processing_time = end_time - start_time
            
            # Should complete within reasonable time (10 minutes max)
            max_time = 600
            assert processing_time < max_time, \
                f"Very large file processing took {processing_time:.2f}s, expected < {max_time}s"
            
            print(f"\nVery large file processing:")
            print(f"  File size: {actual_size:.2f} MB")
            print(f"  Row count: {row_count:,}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            
        except MemoryError:
            # Expected on systems with limited memory
            pytest.skip("Insufficient memory for very large file test")
        except Exception as e:
            # Other errors should be handled gracefully
            assert "memory" in str(e).lower() or "size" in str(e).lower(), \
                f"Unexpected error (not memory-related): {e}"
    
    def test_file_size_limit_enforcement(self, temp_dir: Path):
        """Test that file size limits are properly enforced."""
        # Create file larger than limit
        large_file_path = temp_dir / "oversized.xlsx"
        TestFileGenerator.create_large_excel_file(large_file_path, 10.0)
        
        # Set small file size limit
        processor = ExcelProcessor(max_file_size_mb=5.0)
        
        with pytest.raises(ValueError, match="File size.*exceeds maximum"):
            processor.process_excel_file(large_file_path)
    
    def test_concurrent_limit_behavior(self, temp_dir: Path):
        """Test behavior when concurrent processing limits are reached."""
        output_dir = temp_dir / "concurrent_limit_output"
        output_dir.mkdir()
        
        # Create many small files
        test_files = []
        for i in range(20):  # More files than concurrent limit
            file_path = temp_dir / f"concurrent_limit_test_{i}.xlsx"
            pd.DataFrame({
                'ID': range(100),
                'Value': range(100, 200)
            }).to_excel(file_path, index=False)
            test_files.append(file_path)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.3,
            max_concurrent=3  # Limited concurrency
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        start_time = time.time()
        
        # Process all files
        results = []
        for file_path in test_files:
            result = converter.process_single_file(file_path)
            results.append(result)
        
        end_time = time.time()
        
        # Most files should be processed successfully
        successful_results = [r for r in results if r and r.success]
        assert len(successful_results) >= len(test_files) - 2
        
        # Should complete within reasonable time despite many files
        max_total_time = 120  # 2 minutes for 20 small files
        assert end_time - start_time < max_total_time, \
            f"Processing {len(test_files)} files took {end_time - start_time:.2f}s, expected < {max_total_time}s"
        
        print(f"\nConcurrent limit test:")
        print(f"  Files to process: {len(test_files)}")
        print(f"  Successfully processed: {len(successful_results)}")
        print(f"  Total time: {end_time - start_time:.2f} seconds")
        print(f"  Concurrent limit: 3")


# Performance benchmarking utilities
def benchmark_operation(operation_func, *args, **kwargs):
    """Utility to benchmark any operation."""
    tracker = PerformanceTracker()
    tracker.start()
    
    result = operation_func(*args, **kwargs)
    
    tracker.stop()
    
    return {
        'result': result,
        'duration': tracker.duration,
        'memory_increase': tracker.memory_increase,
        'peak_memory_increase': tracker.peak_memory_increase
    }


@pytest.mark.performance
def test_performance_regression_suite(temp_dir: Path):
    """Comprehensive performance regression test suite."""
    print("\n" + "="*80)
    print("PERFORMANCE REGRESSION TEST SUITE")
    print("="*80)
    
    # Create test files of different sizes
    small_file = temp_dir / "small.xlsx"
    medium_file = temp_dir / "medium.xlsx"
    
    # Small file: ~1MB
    TestFileGenerator.create_large_excel_file(small_file, 1.0)
    # Medium file: ~10MB  
    TestFileGenerator.create_large_excel_file(medium_file, 10.0)
    
    # Benchmark different operations
    processor = ExcelProcessor()
    analyzer = ConfidenceAnalyzer()
    
    # Test 1: Small file processing
    small_bench = benchmark_operation(processor.process_excel_file, small_file)
    print(f"\nSmall file processing (1MB):")
    print(f"  Time: {small_bench['duration']:.3f}s")
    print(f"  Memory: {small_bench['memory_increase']:.2f}MB")
    
    # Test 2: Medium file processing
    medium_bench = benchmark_operation(processor.process_excel_file, medium_file)
    print(f"\nMedium file processing (10MB):")
    print(f"  Time: {medium_bench['duration']:.3f}s")
    print(f"  Memory: {medium_bench['memory_increase']:.2f}MB")
    
    # Performance assertions
    assert small_bench['duration'] < 10, "Small file should process in under 10s"
    assert medium_bench['duration'] < 60, "Medium file should process in under 60s"
    assert small_bench['memory_increase'] < 50, "Small file should use under 50MB"
    assert medium_bench['memory_increase'] < 200, "Medium file should use under 200MB"
    
    print(f"\n✓ All performance benchmarks passed!")
    print("="*80)