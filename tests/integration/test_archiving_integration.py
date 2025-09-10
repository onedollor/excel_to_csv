"""Integration tests for file archiving functionality."""

import pytest
import time
import threading
from pathlib import Path
import pandas as pd
import yaml
from unittest.mock import patch

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.config.config_manager import ConfigManager
from excel_to_csv.models.data_models import Config, ArchiveConfig
from excel_to_csv.archiving.archive_manager import ArchiveManager


class TestArchivingIntegration:
    """Integration tests for complete archiving workflows."""
    
    def test_end_to_end_archiving_workflow(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test complete workflow with archiving enabled."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create test Excel file
        excel_file = input_dir / "test_data.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)
        
        # Create config with archiving enabled
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,  # Lower threshold for testing
            max_concurrent=1,
            archive_config=ArchiveConfig(
                enabled=True,
                archive_folder_name="archive",
                handle_conflicts=True
            )
        )
        
        # Process file
        converter = ExcelToCSVConverter()
        converter.config = config
        
        result = converter._process_file_pipeline(excel_file)
        
        assert result is True
        
        # Verify CSV was generated
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0
        
        # Verify original Excel file was archived
        archive_dir = input_dir / "archive"
        assert archive_dir.exists()
        assert archive_dir.is_dir()
        
        archived_file = archive_dir / "test_data.xlsx"
        assert archived_file.exists()
        assert not excel_file.exists()  # Original should be moved
        
        # Verify statistics were updated
        stats = converter.get_statistics()
        assert stats['files_archived'] == 1
        assert stats['archive_failures'] == 0
        assert stats['csv_files_generated'] >= 1
    
    def test_archiving_disabled_workflow(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test workflow with archiving disabled."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create test Excel file
        excel_file = input_dir / "test_data.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)
        
        # Create config with archiving disabled
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1,
            archive_config=ArchiveConfig(enabled=False)
        )
        
        # Process file
        converter = ExcelToCSVConverter()
        converter.config = config
        
        result = converter._process_file_pipeline(excel_file)
        
        assert result is True
        
        # Verify CSV was generated
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0
        
        # Verify original Excel file was NOT archived
        assert excel_file.exists()  # Original should remain
        archive_dir = input_dir / "archive"
        assert not archive_dir.exists()  # No archive folder created
        
        # Verify statistics
        stats = converter.get_statistics()
        assert stats['files_archived'] == 0
        assert stats['archive_failures'] == 0
    
    def test_archiving_with_naming_conflicts(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test archiving with filename conflicts."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        archive_dir = input_dir / "archive"
        archive_dir.mkdir()
        
        # Create test Excel file
        excel_file = input_dir / "test_data.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)
        
        # Create conflicting file in archive
        conflicting_file = archive_dir / "test_data.xlsx"
        conflicting_file.write_text("existing content")
        
        # Create config with archiving enabled
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1,
            archive_config=ArchiveConfig(
                enabled=True,
                handle_conflicts=True,
                timestamp_format="%Y%m%d_%H%M%S"
            )
        )
        
        # Process file
        converter = ExcelToCSVConverter()
        converter.config = config
        
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            result = converter._process_file_pipeline(excel_file)
        
        assert result is True
        
        # Verify original file was archived with timestamp
        assert not excel_file.exists()  # Original moved
        assert conflicting_file.exists()  # Original conflict file untouched
        
        timestamped_file = archive_dir / "test_data_20240101_120000.xlsx"
        assert timestamped_file.exists()
        
        # Verify statistics
        stats = converter.get_statistics()
        assert stats['files_archived'] == 1
        assert stats['archive_failures'] == 0
    
    def test_archiving_failure_graceful_handling(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test graceful handling of archiving failures."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create test Excel file
        excel_file = input_dir / "test_data.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)
        
        # Create config with archiving enabled
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1,
            archive_config=ArchiveConfig(enabled=True)
        )
        
        # Process file with mocked archiving failure
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Mock archive manager to simulate failure
        with patch.object(converter.archive_manager, 'archive_file') as mock_archive:
            from excel_to_csv.models.data_models import ArchiveResult
            mock_archive.return_value = ArchiveResult(
                success=False,
                source_path=excel_file,
                error_message="Simulated archive failure",
                operation_time=0.1
            )
            
            result = converter._process_file_pipeline(excel_file)
        
        # Processing should succeed despite archiving failure
        assert result is True
        
        # Verify CSV was still generated
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0
        
        # Verify original file remains (archiving failed)
        assert excel_file.exists()
        
        # Verify failure statistics
        stats = converter.get_statistics()
        assert stats['files_archived'] == 0
        assert stats['archive_failures'] == 1
        assert stats['csv_files_generated'] >= 1
    
    def test_multiple_files_archiving(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test archiving multiple files."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create multiple test Excel files
        file_names = ["file1.xlsx", "file2.xlsx", "file3.xlsx"]
        excel_files = []
        
        for name in file_names:
            excel_file = input_dir / name
            sample_excel_data.to_excel(excel_file, index=False)
            excel_files.append(excel_file)
        
        # Create config with archiving enabled
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=1,
            archive_config=ArchiveConfig(enabled=True)
        )
        
        # Process all files
        converter = ExcelToCSVConverter()
        converter.config = config
        
        results = []
        for excel_file in excel_files:
            result = converter._process_file_pipeline(excel_file)
            results.append(result)
        
        # All should succeed
        assert all(results)
        
        # Verify all files were archived
        archive_dir = input_dir / "archive"
        assert archive_dir.exists()
        
        for name in file_names:
            archived_file = archive_dir / name
            assert archived_file.exists()
        
        # Verify no original files remain
        for excel_file in excel_files:
            assert not excel_file.exists()
        
        # Verify statistics
        stats = converter.get_statistics()
        assert stats['files_archived'] == 3
        assert stats['archive_failures'] == 0
        assert stats['csv_files_generated'] >= 3
    
    def test_archiving_with_service_mode_simulation(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test archiving in a service-like scenario with multiple files arriving over time."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create config with archiving enabled
        config = Config(
            monitored_folders=[input_dir],
            output_folder=output_dir,
            confidence_threshold=0.5,
            max_concurrent=2,
            archive_config=ArchiveConfig(enabled=True)
        )
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        # Simulate files arriving over time
        processed_files = []
        for i in range(5):
            # Create file
            excel_file = input_dir / f"batch_{i}.xlsx"
            sample_excel_data.to_excel(excel_file, index=False)
            
            # Process immediately
            result = converter._process_file_pipeline(excel_file)
            processed_files.append((excel_file, result))
            
            # Small delay to simulate real-world timing
            time.sleep(0.1)
        
        # Verify all processing succeeded
        assert all(result for _, result in processed_files)
        
        # Verify all files were archived
        archive_dir = input_dir / "archive"
        assert archive_dir.exists()
        
        archived_files = list(archive_dir.glob("*.xlsx"))
        assert len(archived_files) == 5
        
        # Verify no original files remain in input
        remaining_files = list(input_dir.glob("*.xlsx"))
        assert len(remaining_files) == 0
        
        # Verify statistics
        stats = converter.get_statistics()
        assert stats['files_archived'] == 5
        assert stats['archive_failures'] == 0
    
    def test_archiving_configuration_from_yaml(self, temp_dir: Path, sample_excel_data: pd.DataFrame):
        """Test archiving with configuration loaded from YAML file."""
        # Setup directories
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create Excel file
        excel_file = input_dir / "test_data.xlsx"
        sample_excel_data.to_excel(excel_file, index=False)
        
        # Create YAML config file with archiving settings
        config_dict = {
            "monitoring": {
                "folders": [str(input_dir)],
                "file_patterns": ["*.xlsx", "*.xls"],
            },
            "output": {
                "folder": str(output_dir),
            },
            "confidence": {
                "threshold": 0.5,
            },
            "archiving": {
                "enabled": True,
                "archive_folder_name": "processed",
                "handle_conflicts": True,
                "timestamp_format": "%Y-%m-%d_%H%M%S"
            },
            "processing": {
                "max_concurrent": 1,
            },
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load config and process
        config_manager = ConfigManager()
        config = config_manager.load_config(config_file)
        
        converter = ExcelToCSVConverter()
        converter.config = config
        
        result = converter._process_file_pipeline(excel_file)
        
        assert result is True
        
        # Verify archiving used custom settings
        archive_dir = input_dir / "processed"  # Custom archive folder name
        assert archive_dir.exists()
        
        archived_file = archive_dir / "test_data.xlsx"
        assert archived_file.exists()
        assert not excel_file.exists()
        
        # Verify config was loaded correctly
        assert config.archive_config.enabled is True
        assert config.archive_config.archive_folder_name == "processed"
        assert config.archive_config.timestamp_format == "%Y-%m-%d_%H%M%S"