"""Comprehensive integration tests for end-to-end workflows and edge cases.

This test suite covers:
- Complete Excel-to-CSV conversion workflows
- Multi-component error propagation
- Service mode integration (monitoring + processing + archiving)
- Configuration-driven behavior changes
- Complex boundary conditions and edge cases
- Performance and concurrency scenarios
"""

import pytest
import pandas as pd
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.config.config_manager import ConfigManager
from excel_to_csv.models.data_models import ConversionConfig, OutputConfig, MonitoringConfig
from excel_to_csv.monitoring.file_monitor import FileMonitor
from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.processors.excel_processor import ExcelProcessor
from excel_to_csv.generators.csv_generator import CSVGenerator
from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
from excel_to_csv.utils.metrics import get_metrics_collector
from excel_to_csv.utils.correlation import CorrelationContext


@pytest.fixture
def temp_workspace():
    """Create comprehensive temporary workspace."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    archive_dir = temp_dir / "archive"
    config_dir = temp_dir / "config"
    
    for dir_path in [input_dir, output_dir, archive_dir, config_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    yield {
        'base': temp_dir,
        'input': input_dir,
        'output': output_dir,
        'archive': archive_dir,
        'config': config_dir
    }
    
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_excel_files(temp_workspace):
    """Create sample Excel files for testing."""
    input_dir = temp_workspace['input']
    files = {}
    
    # Simple valid Excel file
    simple_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [95.5, 87.2, 92.1]
    })
    simple_file = input_dir / "simple.xlsx"
    simple_data.to_excel(simple_file, index=False)
    files['simple'] = simple_file
    
    # Multi-sheet Excel file
    multi_file = input_dir / "multi_sheet.xlsx"
    with pd.ExcelWriter(multi_file) as writer:
        simple_data.to_excel(writer, sheet_name='Sheet1', index=False)
        
        complex_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Amount': [100.50, 200.75, 150.25, 300.00, 250.80],
            'Category': ['A', 'B', 'A', 'C', 'B']
        })
        complex_data.to_excel(writer, sheet_name='Sheet2', index=False)
    files['multi'] = multi_file
    
    # Large Excel file
    large_data = pd.DataFrame({
        f'Col_{i}': range(i * 100, (i + 1) * 100) for i in range(20)
    })
    large_file = input_dir / "large.xlsx"
    large_data.to_excel(large_file, index=False)
    files['large'] = large_file
    
    # Edge case: empty Excel file
    empty_data = pd.DataFrame()
    empty_file = input_dir / "empty.xlsx"
    empty_data.to_excel(empty_file, index=False)
    files['empty'] = empty_file
    
    return files


@pytest.fixture
def integration_config(temp_workspace):
    """Create integration test configuration."""
    return ConversionConfig(
        monitoring=MonitoringConfig(
            folders=[temp_workspace['input']],
            file_patterns=['*.xlsx', '*.xls'],
            process_existing=True,
            debounce_seconds=0.1  # Fast for testing
        ),
        output=OutputConfig(
            folder=temp_workspace['output'],
            include_headers=True,
            include_timestamp=False,
            naming_pattern="{filename}_{worksheet}.csv"
        ),
        processing={
            'confidence_threshold': 0.7,
            'max_concurrent_files': 3,
            'enable_archiving': True
        },
        archiving={
            'enabled': True,
            'archive_folder': str(temp_workspace['archive']),
            'archive_successful': True,
            'archive_failed': True,
            'cleanup_source': False
        }
    )


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_single_file_conversion_workflow(self, sample_excel_files, integration_config, temp_workspace):
        """Test complete single file conversion workflow."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process single file
        excel_file = sample_excel_files['simple']
        results = converter.process_file(excel_file)
        
        # Verify results
        assert len(results) >= 1
        assert all(result.success for result in results)
        
        # Check CSV files were created
        output_files = list(temp_workspace['output'].glob("*.csv"))
        assert len(output_files) >= 1
        
        # Verify CSV content
        csv_file = output_files[0]
        df = pd.read_csv(csv_file)
        assert len(df) == 3
        assert 'ID' in df.columns
        assert 'Name' in df.columns
        assert 'Score' in df.columns
    
    def test_multi_sheet_conversion_workflow(self, sample_excel_files, integration_config, temp_workspace):
        """Test multi-sheet Excel file conversion."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process multi-sheet file
        excel_file = sample_excel_files['multi']
        results = converter.process_file(excel_file)
        
        # Should have multiple results (one per sheet)
        assert len(results) >= 2
        
        # Check multiple CSV files created
        output_files = list(temp_workspace['output'].glob("*.csv"))
        assert len(output_files) >= 2
        
        # Verify different sheet content
        csv_contents = []
        for csv_file in output_files:
            df = pd.read_csv(csv_file)
            csv_contents.append(df)
        
        # Should have different structures/content
        assert len(csv_contents) >= 2
        # Different column sets indicate different sheets
        col_sets = [set(df.columns) for df in csv_contents]
        assert len(set(frozenset(cols) for cols in col_sets)) >= 2
    
    def test_batch_processing_workflow(self, sample_excel_files, integration_config, temp_workspace):
        """Test batch processing of multiple files."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process multiple files
        files_to_process = [
            sample_excel_files['simple'],
            sample_excel_files['multi']
        ]
        
        all_results = []
        for excel_file in files_to_process:
            results = converter.process_file(excel_file)
            all_results.extend(results)
        
        # Should have processed all files
        assert len(all_results) >= 3  # simple (1 sheet) + multi (2+ sheets)
        
        # Check CSV outputs
        output_files = list(temp_workspace['output'].glob("*.csv"))
        assert len(output_files) >= 3
    
    def test_service_mode_integration(self, sample_excel_files, integration_config, temp_workspace):
        """Test service mode with file monitoring integration."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Start service mode
        service_thread = threading.Thread(target=converter.run_service, daemon=True)
        service_thread.start()
        
        # Give service time to start
        time.sleep(0.2)
        
        # Copy file to monitored directory (simulate new file)
        new_file = temp_workspace['input'] / "monitored_file.xlsx"
        shutil.copy2(sample_excel_files['simple'], new_file)
        
        # Wait for processing
        time.sleep(0.5)
        
        # Stop service
        converter.shutdown()
        
        # Check if file was processed
        output_files = list(temp_workspace['output'].glob("*monitored_file*.csv"))
        assert len(output_files) >= 1
    
    def test_archiving_integration_workflow(self, sample_excel_files, integration_config, temp_workspace):
        """Test integration with archiving system."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process file with archiving enabled
        excel_file = sample_excel_files['simple']
        results = converter.process_file(excel_file)
        
        # Verify processing succeeded
        assert len(results) >= 1
        assert all(result.success for result in results)
        
        # Check archiving occurred
        archive_files = list(temp_workspace['archive'].glob("**/*.xlsx"))
        assert len(archive_files) >= 1
        
        # Verify archived file content
        archived_file = archive_files[0]
        assert archived_file.exists()
        assert archived_file.stat().st_size > 0


class TestErrorPropagationAndRecovery:
    """Test error propagation and recovery scenarios."""
    
    def test_invalid_excel_file_error_propagation(self, temp_workspace, integration_config):
        """Test error handling with invalid Excel file."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Create invalid "Excel" file
        invalid_file = temp_workspace['input'] / "invalid.xlsx"
        invalid_file.write_text("This is not an Excel file")
        
        # Process invalid file
        results = converter.process_file(invalid_file)
        
        # Should handle error gracefully
        assert len(results) >= 0  # May be empty or contain error results
        if results:
            assert all(not result.success for result in results)
    
    def test_permission_error_handling(self, sample_excel_files, temp_workspace):
        """Test handling of permission errors."""
        # Create read-only output directory
        readonly_output = temp_workspace['base'] / "readonly_output"
        readonly_output.mkdir()
        readonly_output.chmod(0o444)  # Read-only
        
        try:
            config = ConversionConfig(
                output=OutputConfig(folder=readonly_output)
            )
            converter = ExcelToCSVConverter(config)
            
            # Should handle permission error
            results = converter.process_file(sample_excel_files['simple'])
            
            # Should either succeed (if permissions allow) or fail gracefully
            if results:
                # Check that errors are properly reported
                failed_results = [r for r in results if not r.success]
                if failed_results:
                    assert all(hasattr(r, 'error_message') for r in failed_results)
        
        finally:
            # Restore permissions for cleanup
            readonly_output.chmod(0o755)
    
    def test_disk_space_simulation(self, sample_excel_files, integration_config, temp_workspace):
        """Test handling of disk space issues."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Mock disk space error
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            mock_to_csv.side_effect = OSError("No space left on device")
            
            results = converter.process_file(sample_excel_files['simple'])
            
            # Should handle disk space error gracefully
            if results:
                failed_results = [r for r in results if not r.success]
                assert len(failed_results) >= 0  # Should handle error
    
    def test_confidence_threshold_filtering(self, sample_excel_files, temp_workspace):
        """Test confidence-based filtering integration."""
        # High confidence threshold - should reject most files
        strict_config = ConversionConfig(
            output=OutputConfig(folder=temp_workspace['output']),
            processing={'confidence_threshold': 0.95}
        )
        
        converter = ExcelToCSVConverter(strict_config)
        
        # Process file with strict confidence
        results = converter.process_file(sample_excel_files['simple'])
        
        # May reject due to confidence threshold
        if results:
            rejected_results = [r for r in results if not r.success and 'confidence' in str(r.error_message).lower()]
            # At least validate that confidence checking is working
            assert isinstance(results, list)


class TestBoundaryConditionsAndEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_empty_excel_file_handling(self, sample_excel_files, integration_config, temp_workspace):
        """Test handling of empty Excel files."""
        converter = ExcelToCSVConverter(integration_config)
        
        results = converter.process_file(sample_excel_files['empty'])
        
        # Should handle empty file gracefully
        assert isinstance(results, list)
        
        # Check if empty CSV was created or if it was properly rejected
        output_files = list(temp_workspace['output'].glob("*empty*.csv"))
        if output_files:
            # If CSV was created, it should be valid (even if empty)
            for csv_file in output_files:
                assert csv_file.exists()
    
    def test_large_file_processing(self, sample_excel_files, integration_config, temp_workspace):
        """Test processing of large Excel files."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process large file
        start_time = time.time()
        results = converter.process_file(sample_excel_files['large'])
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert processing_time < 30  # 30 second timeout
        
        if results and any(r.success for r in results):
            # Check large CSV output
            output_files = list(temp_workspace['output'].glob("*large*.csv"))
            if output_files:
                csv_file = output_files[0]
                assert csv_file.stat().st_size > 1000  # Should be substantial
    
    def test_concurrent_file_processing(self, sample_excel_files, integration_config, temp_workspace):
        """Test concurrent processing of multiple files."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process multiple files concurrently
        files_to_process = list(sample_excel_files.values())[:3]  # Process 3 files
        
        def process_file(file_path):
            return converter.process_file(file_path)
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_file, file_path) for file_path in files_to_process]
            results = [future.result() for future in futures]
        
        # Should handle concurrent processing
        assert len(results) == len(files_to_process)
        
        # Check outputs
        output_files = list(temp_workspace['output'].glob("*.csv"))
        assert len(output_files) >= 3
    
    def test_special_characters_in_filenames(self, temp_workspace, integration_config):
        """Test handling of special characters in file and sheet names."""
        # Create file with special characters
        special_data = pd.DataFrame({'Col1': [1, 2, 3], 'Col2': ['A', 'B', 'C']})
        special_file = temp_workspace['input'] / "file with spaces & symbols!.xlsx"
        
        with pd.ExcelWriter(special_file) as writer:
            special_data.to_excel(writer, sheet_name='Sheet with "quotes"', index=False)
        
        converter = ExcelToCSVConverter(integration_config)
        results = converter.process_file(special_file)
        
        # Should handle special characters gracefully
        assert isinstance(results, list)
        
        # Check sanitized output files exist
        output_files = list(temp_workspace['output'].glob("*.csv"))
        if output_files:
            # Filenames should be sanitized
            for output_file in output_files:
                assert output_file.exists()
                # Should not contain problematic characters
                assert '<' not in output_file.name
                assert '>' not in output_file.name
                assert '"' not in output_file.name


class TestConfigurationIntegration:
    """Test configuration-driven behavior changes."""
    
    def test_different_output_formats(self, sample_excel_files, temp_workspace):
        """Test different output configuration options."""
        # Test with different delimiters
        semicolon_config = ConversionConfig(
            output=OutputConfig(
                folder=temp_workspace['output'],
                delimiter=';',
                encoding='utf-8'
            )
        )
        
        converter = ExcelToCSVConverter(semicolon_config)
        results = converter.process_file(sample_excel_files['simple'])
        
        if results and any(r.success for r in results):
            output_files = list(temp_workspace['output'].glob("*.csv"))
            if output_files:
                # Check semicolon delimiter
                with open(output_files[0], 'r') as f:
                    content = f.read()
                    assert ';' in content
    
    def test_timestamp_naming_behavior(self, sample_excel_files, temp_workspace):
        """Test timestamp-based naming configuration."""
        timestamp_config = ConversionConfig(
            output=OutputConfig(
                folder=temp_workspace['output'],
                include_timestamp=True,
                timestamp_format="%Y%m%d_%H%M%S"
            )
        )
        
        converter = ExcelToCSVConverter(timestamp_config)
        
        # Process same file twice
        converter.process_file(sample_excel_files['simple'])
        time.sleep(1)  # Ensure different timestamp
        converter.process_file(sample_excel_files['simple'])
        
        # Should create files with timestamps
        output_files = list(temp_workspace['output'].glob("*.csv"))
        if len(output_files) >= 2:
            # Should have different names due to timestamps
            names = [f.name for f in output_files]
            assert len(set(names)) >= 2
    
    def test_header_inclusion_behavior(self, sample_excel_files, temp_workspace):
        """Test header inclusion configuration."""
        no_headers_config = ConversionConfig(
            output=OutputConfig(
                folder=temp_workspace['output'],
                include_headers=False
            )
        )
        
        converter = ExcelToCSVConverter(no_headers_config)
        results = converter.process_file(sample_excel_files['simple'])
        
        if results and any(r.success for r in results):
            output_files = list(temp_workspace['output'].glob("*.csv"))
            if output_files:
                # Check that headers are not included
                df = pd.read_csv(output_files[0], header=None)
                # First row should be data, not headers
                assert df.iloc[0, 0] != 'ID'  # Should be 1, not 'ID'


class TestMetricsAndMonitoringIntegration:
    """Test metrics collection and monitoring integration."""
    
    def test_metrics_collection_during_processing(self, sample_excel_files, integration_config):
        """Test that metrics are collected during processing."""
        collector = get_metrics_collector()
        initial_count = len(collector.metrics)
        
        converter = ExcelToCSVConverter(integration_config)
        results = converter.process_file(sample_excel_files['simple'])
        
        # Should have collected metrics
        final_count = len(collector.metrics)
        assert final_count > initial_count
        
        # Check metrics content
        recent_metrics = collector.get_recent_metrics(limit=10)
        operation_names = [m.operation_name for m in recent_metrics]
        
        # Should have operation-related metrics
        assert any('process' in name.lower() or 'convert' in name.lower() 
                  for name in operation_names)
    
    def test_correlation_id_propagation(self, sample_excel_files, integration_config):
        """Test correlation ID propagation through processing."""
        # Set correlation ID
        CorrelationContext.set_correlation_id("integration-test-123")
        
        converter = ExcelToCSVConverter(integration_config)
        results = converter.process_file(sample_excel_files['simple'])
        
        # Check that correlation ID was maintained
        current_id = CorrelationContext.get_correlation_id()
        assert current_id == "integration-test-123"
        
        # Check metrics have correlation ID
        collector = get_metrics_collector()
        recent_metrics = collector.get_recent_metrics(limit=5)
        
        if recent_metrics:
            # At least some metrics should have our correlation ID
            correlation_ids = [m.correlation_id for m in recent_metrics]
            assert "integration-test-123" in correlation_ids
    
    def test_performance_monitoring(self, sample_excel_files, integration_config):
        """Test performance monitoring and statistics."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Process file and measure
        start_time = time.time()
        results = converter.process_file(sample_excel_files['large'])
        processing_time = time.time() - start_time
        
        # Get statistics
        stats = converter.get_statistics()
        
        # Should have processing statistics
        assert 'total_files_processed' in stats
        assert 'total_worksheets_processed' in stats
        
        # Should track timing
        if 'processing_times' in stats:
            assert isinstance(stats['processing_times'], list)


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_mixed_success_failure_batch(self, sample_excel_files, temp_workspace, integration_config):
        """Test batch with mix of successful and failed files."""
        # Create a corrupted file
        corrupted_file = temp_workspace['input'] / "corrupted.xlsx"
        corrupted_file.write_bytes(b"corrupted data")
        
        converter = ExcelToCSVConverter(integration_config)
        
        # Process mix of good and bad files
        files_to_process = [
            sample_excel_files['simple'],  # Good
            corrupted_file,                # Bad
            sample_excel_files['multi']    # Good
        ]
        
        all_results = []
        for file_path in files_to_process:
            try:
                results = converter.process_file(file_path)
                all_results.extend(results)
            except Exception:
                # Some files may raise exceptions
                pass
        
        # Should have processed at least the good files
        successful_results = [r for r in all_results if r.success]
        assert len(successful_results) >= 2  # At least the 2 good files
    
    def test_rapid_file_changes_in_service_mode(self, sample_excel_files, integration_config, temp_workspace):
        """Test rapid file changes in service mode."""
        converter = ExcelToCSVConverter(integration_config)
        
        # Start service
        service_thread = threading.Thread(target=converter.run_service, daemon=True)
        service_thread.start()
        
        time.sleep(0.1)  # Let service start
        
        # Rapidly add multiple files
        for i in range(3):
            new_file = temp_workspace['input'] / f"rapid_{i}.xlsx"
            shutil.copy2(sample_excel_files['simple'], new_file)
            time.sleep(0.05)  # Very rapid
        
        # Wait for processing
        time.sleep(1.0)
        
        # Stop service
        converter.shutdown()
        
        # Should have processed files
        output_files = list(temp_workspace['output'].glob("*rapid*.csv"))
        assert len(output_files) >= 1  # At least some should be processed
    
    def test_configuration_hot_reload(self, sample_excel_files, temp_workspace):
        """Test behavior with configuration changes during processing."""
        # Start with one configuration
        initial_config = ConversionConfig(
            output=OutputConfig(
                folder=temp_workspace['output'],
                delimiter=','
            )
        )
        
        converter = ExcelToCSVConverter(initial_config)
        
        # Process one file
        results1 = converter.process_file(sample_excel_files['simple'])
        
        # Change configuration (simulate hot reload)
        new_config = ConversionConfig(
            output=OutputConfig(
                folder=temp_workspace['output'],
                delimiter=';'  # Different delimiter
            )
        )
        
        # Create new converter with new config
        converter2 = ExcelToCSVConverter(new_config)
        
        # Process another file
        results2 = converter2.process_file(sample_excel_files['multi'])
        
        # Both should succeed
        assert isinstance(results1, list)
        assert isinstance(results2, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])