"""Comprehensive tests for ArchiveManager class."""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.models.data_models import ArchiveConfig, ArchiveResult, ArchiveError, RetryConfig


class TestArchiveManager:
    """Test cases for ArchiveManager class."""
    
    def test_init_default(self):
        """Test ArchiveManager initialization with defaults."""
        manager = ArchiveManager()
        
        assert manager.retry_config is not None
        assert manager.retry_config.max_attempts == 3
        assert manager.retry_config.delay == 1.0
        assert manager.retry_config.backoff_factor == 2.0
        assert manager.retry_config.max_delay == 10.0
        assert manager.logger is not None
    
    def test_init_custom_retry_config(self):
        """Test ArchiveManager initialization with custom retry config."""
        custom_retry = RetryConfig(
            max_attempts=5,
            delay=2.0,
            backoff_factor=1.5,
            max_delay=30.0
        )
        manager = ArchiveManager(retry_config=custom_retry)
        
        assert manager.retry_config == custom_retry
        assert manager.retry_config.max_attempts == 5
        assert manager.retry_config.delay == 2.0
    
    def test_create_archive_folder_new(self):
        """Test creating a new archive folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_folder = Path(temp_dir)
            manager = ArchiveManager()
            
            archive_path = manager.create_archive_folder(base_folder, "archive")
            
            assert archive_path.exists()
            assert archive_path.is_dir()
            assert archive_path.name == "archive"
            assert archive_path.parent == base_folder
    
    def test_create_archive_folder_existing(self):
        """Test creating archive folder when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_folder = Path(temp_dir)
            archive_folder_name = "existing_archive"
            
            # Create the archive folder first
            archive_path = base_folder / archive_folder_name
            archive_path.mkdir()
            
            manager = ArchiveManager()
            result_path = manager.create_archive_folder(base_folder, archive_folder_name)
            
            assert result_path == archive_path
            assert result_path.exists()
            assert result_path.is_dir()
    
    def test_validate_archive_path_valid(self):
        """Test archive path validation with valid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir)
            manager = ArchiveManager()
            
            assert manager.validate_archive_path(valid_path) is True
    
    def test_validate_archive_path_nonexistent(self):
        """Test archive path validation with nonexistent path."""
        nonexistent_path = Path("/nonexistent/path/that/should/not/exist")
        manager = ArchiveManager()
        
        assert manager.validate_archive_path(nonexistent_path) is False
    
    def test_validate_archive_path_file(self):
        """Test archive path validation with file instead of directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file instead of directory
            file_path = Path(temp_dir) / "file.txt"
            file_path.write_text("test")
            
            manager = ArchiveManager()
            assert manager.validate_archive_path(file_path) is False
    
    def test_resolve_naming_conflicts_no_conflict(self):
        """Test conflict resolution when no conflict exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_folder = Path(temp_dir)
            original_name = "test_file.xlsx"
            manager = ArchiveManager()
            
            resolved_path = manager.resolve_naming_conflicts(
                archive_folder, original_name, "%Y%m%d_%H%M%S"
            )
            
            expected_path = archive_folder / original_name
            assert resolved_path == expected_path
    
    def test_resolve_naming_conflicts_with_conflict(self):
        """Test conflict resolution when file already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_folder = Path(temp_dir)
            original_name = "test_file.xlsx"
            
            # Create conflicting file
            conflicting_file = archive_folder / original_name
            conflicting_file.write_text("existing content")
            
            manager = ArchiveManager()
            
            with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime(2023, 12, 25, 14, 30, 45)
                mock_datetime.strftime = datetime.strftime  # Keep strftime working
                
                resolved_path = manager.resolve_naming_conflicts(
                    archive_folder, original_name, "%Y%m%d_%H%M%S"
                )
                
                # Should have timestamp added
                assert resolved_path.name != original_name
                assert "20231225_143045" in resolved_path.name
                assert resolved_path.suffix == ".xlsx"
    
    def test_archive_file_success(self):
        """Test successful file archiving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            source_dir.mkdir()
            
            # Create source file
            source_file = source_dir / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(
                enabled=True,
                archive_folder_name="archive",
                preserve_structure=True,
                handle_conflicts=True
            )
            
            manager = ArchiveManager()
            result = manager.archive_file(source_file, archive_config)
            
            assert isinstance(result, ArchiveResult)
            assert result.success is True
            assert result.source_path == source_file
            assert result.archive_path is not None
            assert result.archive_path.exists()
            assert result.error_message is None
            assert not source_file.exists()  # Original file should be moved
    
    def test_archive_file_disabled(self):
        """Test archiving when disabled in config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(enabled=False)
            
            manager = ArchiveManager()
            result = manager.archive_file(source_file, archive_config)
            
            assert isinstance(result, ArchiveResult)
            assert result.success is True
            assert result.source_path == source_file
            assert result.archive_path is None
            assert result.error_message is None
            assert source_file.exists()  # Original file should remain
    
    def test_archive_file_source_not_exists(self):
        """Test archiving when source file doesn't exist."""
        nonexistent_file = Path("/nonexistent/file.xlsx")
        archive_config = ArchiveConfig(enabled=True)
        
        manager = ArchiveManager()
        result = manager.archive_file(nonexistent_file, archive_config)
        
        assert isinstance(result, ArchiveResult)
        assert result.success is False
        assert result.source_path == nonexistent_file
        assert result.archive_path is None
        assert "does not exist" in result.error_message
    
    def test_archive_file_with_structure_preservation(self):
        """Test archiving with structure preservation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested source structure
            source_dir = Path(temp_dir) / "source"
            nested_dir = source_dir / "subdir" / "nested"
            nested_dir.mkdir(parents=True)
            
            source_file = nested_dir / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(
                enabled=True,
                archive_folder_name="archive",
                preserve_structure=True
            )
            
            manager = ArchiveManager()
            result = manager.archive_file(source_file, archive_config)
            
            assert result.success is True
            assert result.archive_path is not None
            
            # Check structure preservation
            # Archive path should maintain relative structure
            relative_to_root = source_file.relative_to(source_dir)
            expected_archive_path = source_dir / "archive" / relative_to_root
            assert result.archive_path == expected_archive_path
    
    def test_archive_file_without_structure_preservation(self):
        """Test archiving without structure preservation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested source structure
            source_dir = Path(temp_dir) / "source" 
            nested_dir = source_dir / "subdir" / "nested"
            nested_dir.mkdir(parents=True)
            
            source_file = nested_dir / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(
                enabled=True,
                archive_folder_name="archive",
                preserve_structure=False
            )
            
            manager = ArchiveManager()
            result = manager.archive_file(source_file, archive_config)
            
            assert result.success is True
            assert result.archive_path is not None
            
            # File should be directly in archive folder, not preserving structure
            expected_archive_path = source_dir / "archive" / "test.xlsx"
            assert result.archive_path == expected_archive_path
    
    @patch('excel_to_csv.archiving.archive_manager.shutil.move')
    def test_archive_file_move_error(self, mock_move):
        """Test archiving with file move error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(enabled=True)
            
            # Mock shutil.move to raise an error
            mock_move.side_effect = OSError("Permission denied")
            
            manager = ArchiveManager()
            result = manager.archive_file(source_file, archive_config)
            
            assert result.success is False
            assert "Permission denied" in result.error_message
    
    def test_move_file_atomic_success(self):
        """Test atomic file move operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.txt"
            target = Path(temp_dir) / "target.txt"
            
            source.write_text("test content")
            
            manager = ArchiveManager()
            manager._move_file_atomic(source, target)
            
            assert target.exists()
            assert not source.exists()
            assert target.read_text() == "test content"
    
    @patch('excel_to_csv.archiving.archive_manager.shutil.move')
    def test_move_file_with_retry_success_first_attempt(self, mock_move):
        """Test move with retry succeeding on first attempt."""
        manager = ArchiveManager()
        source = Path("source.txt")
        target = Path("target.txt")
        
        manager._move_file_with_retry(source, target)
        
        mock_move.assert_called_once_with(source, target)
    
    @patch('excel_to_csv.archiving.archive_manager.shutil.move')
    @patch('time.sleep')
    def test_move_file_with_retry_success_after_retries(self, mock_sleep, mock_move):
        """Test move with retry succeeding after failures."""
        manager = ArchiveManager()
        source = Path("source.txt")
        target = Path("target.txt")
        
        # First two calls fail, third succeeds
        mock_move.side_effect = [OSError("Temporary error"), OSError("Temporary error"), None]
        
        manager._move_file_with_retry(source, target)
        
        assert mock_move.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch('excel_to_csv.archiving.archive_manager.shutil.move')
    @patch('time.sleep')
    def test_move_file_with_retry_max_attempts_exceeded(self, mock_sleep, mock_move):
        """Test move with retry failing after max attempts."""
        manager = ArchiveManager()
        source = Path("source.txt")
        target = Path("target.txt")
        
        # All attempts fail
        mock_move.side_effect = OSError("Persistent error")
        
        with pytest.raises(OSError, match="Persistent error"):
            manager._move_file_with_retry(source, target)
        
        assert mock_move.call_count == 3  # Default max attempts
    
    def test_is_transient_error_true_cases(self):
        """Test transient error detection for retryable errors."""
        manager = ArchiveManager()
        
        # Test various transient error cases
        assert manager._is_transient_error(OSError("Resource temporarily unavailable"))
        assert manager._is_transient_error(OSError("Device or resource busy"))
        assert manager._is_transient_error(PermissionError("Permission denied"))
        assert manager._is_transient_error(OSError("No space left on device"))
    
    def test_is_transient_error_false_cases(self):
        """Test transient error detection for non-retryable errors."""
        manager = ArchiveManager()
        
        # Test non-transient errors
        assert manager._is_transient_error(FileNotFoundError("File not found")) is False
        assert manager._is_transient_error(IsADirectoryError("Is a directory")) is False
        assert manager._is_transient_error(ValueError("Invalid value")) is False
    
    def test_archive_file_creates_timestamp(self):
        """Test that archive operation records timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(enabled=True)
            
            start_time = datetime.now()
            manager = ArchiveManager()
            result = manager.archive_file(source_file, archive_config)
            end_time = datetime.now()
            
            assert result.success is True
            assert result.operation_time is not None
            assert start_time <= result.operation_time <= end_time
    
    def test_archive_multiple_files_different_names(self):
        """Test archiving multiple files with different names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            archive_config = ArchiveConfig(enabled=True, archive_folder_name="archive")
            manager = ArchiveManager()
            
            # Create multiple source files
            files = ["file1.xlsx", "file2.xlsx", "file3.xlsx"]
            for filename in files:
                source_file = source_dir / filename
                source_file.write_text(f"content of {filename}")
                
                result = manager.archive_file(source_file, archive_config)
                
                assert result.success is True
                assert result.archive_path.name == filename
                assert result.archive_path.exists()
                assert not source_file.exists()
    
    def test_archive_file_performance(self):
        """Test archive performance with timing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "test.xlsx"
            source_file.write_text("test content")
            
            archive_config = ArchiveConfig(enabled=True)
            
            manager = ArchiveManager()
            start_time = datetime.now()
            result = manager.archive_file(source_file, archive_config)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            assert result.success is True
            assert elapsed < 1.0  # Should complete within 1 second for small file