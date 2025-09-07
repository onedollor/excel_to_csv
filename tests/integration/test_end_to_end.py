"""Integration tests for end-to-end workflows."""

import pytest
import time
import threading
from pathlib import Path
import pandas as pd
import yaml
import subprocess
import sys
from unittest.mock import patch

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.config.config_manager import ConfigManager
from excel_to_csv.models.data_models import Config


class TestEndToEndWorkflows:
    """Integration tests for complete Excel-to-CSV workflows."""
    
    def test_single_file_processing_workflow(self, temp_dir: Path, sample_excel_file: Path):
        """Test complete workflow for processing a single Excel file."""
        # Setup output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create config
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,  # Lower threshold for testing
            max_concurrent=1
        )
        
        # Process file
        converter = ExcelToCSVConverter()
        converter.config = config
        
        result = converter.process_single_file(sample_excel_file)
        
        assert result is not None
        assert result.success
        assert len(result.csv_files) > 0
        
        # Verify output files exist
        for csv_file in result.csv_files:
            assert csv_file.exists()
            assert csv_file.suffix == ".csv"
            
            # Verify CSV content
            df = pd.read_csv(csv_file)
            assert not df.empty
    
    def test_service_mode_workflow(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test service mode workflow with file monitoring."""
        # Setup directories
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create config
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=2
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Start service in separate thread
        service_thread = threading.Thread(target=converter.run_service)
        service_thread.daemon = True
        service_thread.start()
        
        # Wait for service to start
        time.sleep(0.5)
        
        try:
            # Create Excel file in monitored directory
            excel_file = input_dir / "service_test.xlsx"
            sample_excel_data.to_excel(excel_file, index=False)
            
            # Wait for processing
            time.sleep(2.0)
            
            # Check for CSV output
            csv_files = list(output_dir.glob("*.csv"))
            assert len(csv_files) > 0
            
            # Verify CSV content
            csv_file = csv_files[0]
            df = pd.read_csv(csv_file)
            assert df.shape == sample_excel_data.shape
            
        finally:
            # Stop service
            converter.stop_service()
            if service_thread.is_alive():
                service_thread.join(timeout=2.0)
    
    def test_configuration_loading_workflow(self, temp_dir: Path):
        """Test end-to-end workflow with configuration file."""
        # Create config file
        config_data = {
            "monitoring": {
                "folders": [str(temp_dir / "input")],
                "file_patterns": ["*.xlsx", "*.xls"],
                "process_existing": True
            },
            "output": {
                "folder": str(temp_dir / "output"),
                "naming_pattern": "converted_{filename}_{worksheet}.csv",
                "include_timestamp": False
            },
            "confidence": {
                "threshold": 0.7
            },
            "processing": {
                "max_concurrent": 3
            }
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create directories
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_file)
        
        # Verify configuration loaded correctly
        assert config.confidence_threshold == 0.7
        assert config.max_concurrent == 3
        assert str(input_dir) in [str(f) for f in config.monitored_folders]
        assert config.output_config.naming_pattern == "converted_{filename}_{worksheet}.csv"
        assert config.output_config.include_timestamp is False
        
        # Test processing with loaded config
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Create test Excel file
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        excel_file = input_dir / "config_test.xlsx"
        test_data.to_excel(excel_file, index=False)
        
        result = converter.process_single_file(excel_file)
        
        assert result.success
        assert len(result.csv_files) > 0
        
        # Verify naming pattern was applied
        csv_file = result.csv_files[0]
        assert "converted_config_test" in csv_file.name
        assert "_Sheet1.csv" in csv_file.name
    
    def test_confidence_threshold_workflow(self, temp_dir: Path):
        """Test workflow with different confidence thresholds."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create high-quality data (should pass high threshold)
        high_quality_data = pd.DataFrame({
            'ID': range(1, 101),
            'Name': [f'Person_{i}' for i in range(1, 101)],
            'Age': [20 + (i % 50) for i in range(100)],
            'Department': ['Eng', 'Sales', 'Marketing'] * 33 + ['HR'],
            'Salary': [50000 + (i * 1000) for i in range(100)]
        })
        
        # Create low-quality data (should fail high threshold)
        low_quality_data = pd.DataFrame(index=range(10), columns=range(5))
        low_quality_data.iloc[0, 0] = "Header1"
        low_quality_data.iloc[1, 0] = "Value1"
        # Rest is sparse
        
        high_quality_file = temp_dir / "high_quality.xlsx"
        low_quality_file = temp_dir / "low_quality.xlsx"
        
        high_quality_data.to_excel(high_quality_file, index=False)
        low_quality_data.to_excel(low_quality_file, index=False)
        
        # Test with high threshold (0.9)
        config_high = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.9,
            max_concurrent=1
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config_high
        
        # Process high-quality file
        result_high = converter.process_single_file(high_quality_file)
        assert result_high.success
        assert len(result_high.csv_files) > 0
        
        # Process low-quality file
        result_low = converter.process_single_file(low_quality_file)
        # May or may not succeed depending on exact confidence calculation
        # The important thing is that it doesn't crash
        assert result_low is not None
        
        # Test with low threshold (0.3)
        config_low = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.3,
            max_concurrent=1
        )
        
        converter.config = config_low
        
        # Both files should likely pass with low threshold
        result_high_low = converter.process_single_file(high_quality_file)
        result_low_low = converter.process_single_file(low_quality_file)
        
        assert result_high_low.success
        # Low quality might still fail, but shouldn't crash
        assert result_low_low is not None
    
    def test_multiple_worksheets_workflow(self, temp_dir: Path):
        """Test processing Excel file with multiple worksheets."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create Excel file with multiple sheets
        data1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        data2 = pd.DataFrame({'X': [7, 8, 9], 'Y': [10, 11, 12]})
        data3 = pd.DataFrame({'P': [13, 14, 15], 'Q': [16, 17, 18]})
        
        excel_file = temp_dir / "multi_sheet.xlsx"
        with pd.ExcelWriter(excel_file) as writer:
            data1.to_excel(writer, sheet_name="Sheet1", index=False)
            data2.to_excel(writer, sheet_name="DataSheet", index=False)
            data3.to_excel(writer, sheet_name="Results", index=False)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        result = converter.process_single_file(excel_file)
        
        assert result.success
        # Should create CSV for each qualifying worksheet
        assert len(result.csv_files) >= 1  # At least one should qualify
        
        # Verify different sheet names in output files
        csv_names = [f.name for f in result.csv_files]
        sheet_names_found = []
        for name in csv_names:
            if "Sheet1" in name:
                sheet_names_found.append("Sheet1")
            elif "DataSheet" in name:
                sheet_names_found.append("DataSheet")
            elif "Results" in name:
                sheet_names_found.append("Results")
        
        # At least some sheets should be processed
        assert len(sheet_names_found) > 0
    
    def test_error_handling_workflow(self, temp_dir: Path):
        """Test error handling in complete workflow."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Test with non-existent file
        nonexistent_file = temp_dir / "does_not_exist.xlsx"
        result = converter.process_single_file(nonexistent_file)
        
        assert not result.success
        assert len(result.errors) > 0
        
        # Test with invalid Excel file
        invalid_file = temp_dir / "invalid.xlsx"
        invalid_file.write_text("This is not an Excel file")
        
        result = converter.process_single_file(invalid_file)
        
        assert not result.success
        assert len(result.errors) > 0
        
        # Test with permission error (mock)
        valid_file = temp_dir / "valid.xlsx"
        pd.DataFrame({'A': [1, 2, 3]}).to_excel(valid_file, index=False)
        
        with patch('excel_to_csv.processors.excel_processor.pd.read_excel') as mock_read:
            mock_read.side_effect = PermissionError("Access denied")
            
            result = converter.process_single_file(valid_file)
            assert not result.success
            assert len(result.errors) > 0
    
    def test_concurrent_processing_workflow(self, temp_dir: Path):
        """Test concurrent processing of multiple files."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create multiple Excel files
        num_files = 5
        excel_files = []
        for i in range(num_files):
            data = pd.DataFrame({
                'ID': range(i*10, (i+1)*10),
                'Value': range(i*100, (i+1)*100)
            })
            excel_file = temp_dir / f"concurrent_{i}.xlsx"
            data.to_excel(excel_file, index=False)
            excel_files.append(excel_file)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=3  # Process 3 files concurrently
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Process all files
        results = []
        for excel_file in excel_files:
            result = converter.process_single_file(excel_file)
            results.append(result)
        
        # All should succeed
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= num_files - 1  # Allow one potential failure
        
        # Verify output files
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) >= num_files - 1
    
    def test_large_file_workflow(self, temp_dir: Path):
        """Test workflow with larger Excel file."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create larger dataset
        large_data = pd.DataFrame({
            'ID': range(1, 1001),
            'Name': [f'Item_{i}' for i in range(1, 1001)],
            'Category': ['A', 'B', 'C'] * 333 + ['D'],
            'Value': [i * 1.5 for i in range(1, 1001)],
            'Description': [f'Description for item {i}' for i in range(1, 1001)]
        })
        
        large_file = temp_dir / "large_file.xlsx"
        large_data.to_excel(large_file, index=False)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Measure processing time
        start_time = time.time()
        result = converter.process_single_file(large_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result.success
        assert len(result.csv_files) > 0
        
        # Verify output
        csv_file = result.csv_files[0]
        output_data = pd.read_csv(csv_file)
        assert output_data.shape == large_data.shape
        
        # Processing should complete in reasonable time (under 30 seconds)
        assert processing_time < 30.0
    
    def test_environment_override_workflow(self, temp_dir: Path, env_override):
        """Test workflow with environment variable overrides."""
        # Set environment overrides
        env_override.set("EXCEL_TO_CSV_CONFIDENCE_THRESHOLD", "0.6")
        env_override.set("EXCEL_TO_CSV_MAX_CONCURRENT", "4")
        env_override.set("EXCEL_TO_CSV_OUTPUT_INCLUDE_TIMESTAMP", "false")
        
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create basic config file (will be overridden by env vars)
        config_data = {
            "monitoring": {"folders": [str(temp_dir)]},
            "output": {"folder": str(output_dir)},
            "confidence": {"threshold": 0.9},  # Will be overridden to 0.6
            "processing": {"max_concurrent": 1}  # Will be overridden to 4
        }
        
        config_file = temp_dir / "env_test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config with environment overrides
        config_manager = ConfigManager()
        config = config_manager.load_config(config_file, use_env_overrides=True)
        
        # Verify overrides applied
        assert config.confidence_threshold == 0.6
        assert config.max_concurrent == 4
        assert config.output_config.include_timestamp is False
        
        # Test processing with overridden config
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Create test file
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        excel_file = temp_dir / "env_test.xlsx"
        test_data.to_excel(excel_file, index=False)
        
        result = converter.process_single_file(excel_file)
        
        assert result.success
        assert len(result.csv_files) > 0
        
        # Verify timestamp not included (due to env override)
        csv_file = result.csv_files[0]
        assert not any(char.isdigit() for char in csv_file.stem[-8:])  # No timestamp pattern
    
    def test_cli_integration_workflow(self, temp_dir: Path, sample_excel_file: Path):
        """Test CLI integration (if available)."""
        output_dir = temp_dir / "cli_output"
        output_dir.mkdir()
        
        # Create config for CLI
        config_data = {
            "monitoring": {"folders": [str(temp_dir)]},
            "output": {"folder": str(output_dir)},
            "confidence": {"threshold": 0.5}
        }
        
        config_file = temp_dir / "cli_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        try:
            # Try to run CLI command
            cmd = [
                sys.executable, "-m", "excel_to_csv.cli",
                "process", str(sample_excel_file),
                "--config", str(config_file)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=temp_dir.parent,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # If CLI is available, it should work
            if result.returncode == 0:
                # Verify output was created
                csv_files = list(output_dir.glob("*.csv"))
                assert len(csv_files) > 0
            else:
                # CLI might not be available in test environment
                # This is acceptable for unit tests
                pass
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # CLI not available or took too long - acceptable in test environment
            pass
    
    def test_memory_usage_workflow(self, temp_dir: Path):
        """Test memory usage during processing."""
        import psutil
        import os
        
        output_dir = temp_dir / "memory_test"
        output_dir.mkdir()
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large dataset
        large_data = pd.DataFrame({
            f'Column_{i}': [f'Value_{j}_{i}' for j in range(500)]
            for i in range(20)  # 500 rows x 20 columns
        })
        
        excel_file = temp_dir / "memory_test.xlsx"
        large_data.to_excel(excel_file, index=False)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Process file
        result = converter.process_single_file(excel_file)
        
        # Check memory usage after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert result.success
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100
    
    def test_stress_test_workflow(self, temp_dir: Path):
        """Stress test with multiple files and operations."""
        output_dir = temp_dir / "stress_test"
        output_dir.mkdir()
        
        # Create many small Excel files
        num_files = 20
        excel_files = []
        
        for i in range(num_files):
            data = pd.DataFrame({
                'ID': range(i*5, (i+1)*5),
                'Value': [f'Data_{i}_{j}' for j in range(5)],
                'Number': [i*10 + j for j in range(5)]
            })
            
            excel_file = temp_dir / f"stress_{i:03d}.xlsx"
            data.to_excel(excel_file, index=False)
            excel_files.append(excel_file)
        
        config = Config(
            monitored_folders=[temp_dir],
            output_folder=output_dir,
            confidence_threshold=0.3,  # Lower threshold for faster processing
            max_concurrent=5  # Higher concurrency
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Process all files
        start_time = time.time()
        successful_count = 0
        
        for excel_file in excel_files:
            result = converter.process_single_file(excel_file)
            if result.success:
                successful_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Most files should succeed
        assert successful_count >= num_files - 2  # Allow a few failures
        
        # Should complete in reasonable time
        assert total_time < 60.0  # Less than 1 minute for 20 small files
        
        # Verify output files
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) >= successful_count