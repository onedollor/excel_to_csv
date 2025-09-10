"""Unit tests for ArchiveManager class."""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, call
import pytest

from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.models.data_models import (
    ArchiveConfig,
    ArchiveError,
    ArchiveResult,
    RetryConfig,
)


class TestArchiveManager:
    """Test cases for ArchiveManager class."""
    
    def test_init_default(self):
        """Test ArchiveManager initialization with default settings."""
        manager = ArchiveManager()
        
        assert manager.retry_config is not None
        assert manager.retry_config.max_attempts == 3
        assert manager.retry_config.delay == 1.0
        assert manager.retry_config.max_delay == 10.0
        assert hasattr(manager, 'logger')
    
    def test_init_custom_retry_config(self):
        """Test ArchiveManager initialization with custom retry config."""
        retry_config = RetryConfig(max_attempts=5, delay=2.0)
        manager = ArchiveManager(retry_config)
        
        assert manager.retry_config.max_attempts == 5
        assert manager.retry_config.delay == 2.0
    
    def test_archive_file_disabled(self, temp_dir: Path):
        """Test archiving when disabled in configuration."""
        manager = ArchiveManager()
        test_file = temp_dir / "test.xlsx"
        test_file.write_text("test content")
        
        archive_config = ArchiveConfig(enabled=False)
        result = manager.archive_file(test_file, archive_config)
        
        assert result.success is True
        assert result.source_path == test_file
        assert result.archive_path is None
        assert test_file.exists()  # File should not be moved
    
    def test_archive_file_success(self, temp_dir: Path):
        """Test successful file archiving."""
        manager = ArchiveManager()
        test_file = temp_dir / "test.xlsx"
        test_file.write_text("test content")
        
        archive_config = ArchiveConfig(enabled=True, archive_folder_name="archive")
        result = manager.archive_file(test_file, archive_config)
        
        assert result.success is True
        assert result.source_path == test_file
        assert result.archive_path == temp_dir / "archive" / "test.xlsx"
        assert not test_file.exists()  # Original file should be moved
        assert result.archive_path.exists()  # File should exist in archive
        assert result.archive_path.read_text() == "test content"
        assert result.operation_time > 0
    
    def test_archive_file_nonexistent_source(self, temp_dir: Path):
        """Test archiving non-existent file."""
        manager = ArchiveManager()
        nonexistent_file = temp_dir / "nonexistent.xlsx"
        
        archive_config = ArchiveConfig(enabled=True)
        result = manager.archive_file(nonexistent_file, archive_config)
        
        assert result.success is False
        assert result.source_path == nonexistent_file
        assert "does not exist" in result.error_message
        assert result.operation_time > 0
    
    def test_archive_file_source_is_directory(self, temp_dir: Path):
        """Test archiving when source is a directory."""
        manager = ArchiveManager()
        test_dir = temp_dir / "test_directory"
        test_dir.mkdir()
        
        archive_config = ArchiveConfig(enabled=True)
        result = manager.archive_file(test_dir, archive_config)
        
        assert result.success is False
        assert result.source_path == test_dir
        assert "not a file" in result.error_message
    
    def test_create_archive_folder_new(self, temp_dir: Path):
        """Test creating new archive folder."""
        manager = ArchiveManager()
        
        archive_folder = manager.create_archive_folder(temp_dir, "archive")
        
        assert archive_folder == temp_dir / "archive"
        assert archive_folder.exists()
        assert archive_folder.is_dir()
    
    def test_create_archive_folder_existing(self, temp_dir: Path):
        """Test using existing archive folder."""
        manager = ArchiveManager()
        existing_archive = temp_dir / "archive"
        existing_archive.mkdir()
        
        archive_folder = manager.create_archive_folder(temp_dir, "archive")
        
        assert archive_folder == existing_archive
        assert archive_folder.exists()
    
    def test_create_archive_folder_permission_error(self, temp_dir: Path):
        """Test handling permission error when creating archive folder."""
        manager = ArchiveManager()
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(ArchiveError) as exc_info:
                manager.create_archive_folder(temp_dir, "archive")
            
            assert exc_info.value.error_type == "permission"
            assert "Permission denied" in str(exc_info.value)
    
    def test_create_archive_folder_existing_file_conflict(self, temp_dir: Path):
        """Test handling conflict when archive name exists as file."""
        manager = ArchiveManager()
        
        # Create a file with the archive folder name
        conflicting_file = temp_dir / "archive"
        conflicting_file.write_text("content")
        
        with pytest.raises(ArchiveError) as exc_info:
            manager.create_archive_folder(temp_dir, "archive")
        
        assert exc_info.value.error_type == "filesystem"
        assert "not a directory" in str(exc_info.value)
    
    def test_resolve_naming_conflicts_single(self, temp_dir: Path):
        """Test resolving single naming conflict."""
        manager = ArchiveManager()
        
        # Create conflicting file
        conflicting_file = temp_dir / "test.xlsx"
        conflicting_file.write_text("existing content")
        
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            result = manager.resolve_naming_conflicts(conflicting_file, "%Y%m%d_%H%M%S")
            
            expected = temp_dir / "test_20240101_120000.xlsx"
            assert result == expected
    
    def test_resolve_naming_conflicts_multiple(self, temp_dir: Path):
        """Test resolving multiple naming conflicts."""
        manager = ArchiveManager()
        
        # Create conflicting files
        (temp_dir / "test.xlsx").write_text("content1")
        (temp_dir / "test_20240101_120000.xlsx").write_text("content2")
        (temp_dir / "test_20240101_120000_001.xlsx").write_text("content3")
        
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            result = manager.resolve_naming_conflicts(temp_dir / "test.xlsx", "%Y%m%d_%H%M%S")
            
            expected = temp_dir / "test_20240101_120000_002.xlsx"
            assert result == expected
    
    def test_resolve_naming_conflicts_too_many(self, temp_dir: Path):
        """Test handling too many naming conflicts."""
        manager = ArchiveManager()
        
        # Create many conflicting files to trigger the limit
        base_name = "test.xlsx"
        (temp_dir / base_name).write_text("content")
        
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            # Mock Path.exists to always return True to simulate infinite conflicts
            with patch.object(Path, 'exists', return_value=True):
                with pytest.raises(ArchiveError) as exc_info:
                    manager.resolve_naming_conflicts(temp_dir / base_name, "%Y%m%d_%H%M%S")
                
                assert exc_info.value.error_type == "filesystem"
                assert "Too many naming conflicts" in str(exc_info.value)
    
    def test_move_file_atomic_success(self, temp_dir: Path):
        """Test successful atomic file move."""
        manager = ArchiveManager()
        
        source = temp_dir / "source.txt"
        source.write_text("test content")
        target = temp_dir / "target.txt"
        
        manager._move_file_atomic(source, target)
        
        assert not source.exists()
        assert target.exists()
        assert target.read_text() == "test content"
    
    def test_move_file_atomic_permission_error(self, temp_dir: Path):
        """Test handling permission error in atomic move."""
        manager = ArchiveManager()
        
        source = temp_dir / "source.txt"
        source.write_text("content")
        target = temp_dir / "target.txt"
        
        with patch.object(Path, 'replace') as mock_replace:
            mock_replace.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(ArchiveError) as exc_info:
                manager._move_file_atomic(source, target)
            
            assert exc_info.value.error_type == "permission"
            assert "Permission denied" in str(exc_info.value)
    
    def test_move_file_atomic_os_error(self, temp_dir: Path):
        """Test handling OS error in atomic move."""
        manager = ArchiveManager()
        
        source = temp_dir / "source.txt" 
        source.write_text("content")
        target = temp_dir / "target.txt"
        
        with patch.object(Path, 'replace') as mock_replace:
            mock_replace.side_effect = OSError("Disk full")
            
            with pytest.raises(ArchiveError) as exc_info:
                manager._move_file_atomic(source, target)
            
            assert exc_info.value.error_type == "filesystem"
            assert "Disk full" in str(exc_info.value)
    
    def test_move_file_with_retry_success_first_attempt(self, temp_dir: Path):
        """Test successful file move on first attempt."""
        retry_config = RetryConfig(max_attempts=3, delay=0.1)
        manager = ArchiveManager(retry_config)
        
        source = temp_dir / "source.txt"
        source.write_text("content")
        target = temp_dir / "target.txt"
        
        manager._move_file_with_retry(source, target)
        
        assert not source.exists()
        assert target.exists()
    
    def test_move_file_with_retry_success_second_attempt(self, temp_dir: Path):
        """Test successful file move on second attempt after retry."""
        retry_config = RetryConfig(max_attempts=3, delay=0.01)
        manager = ArchiveManager(retry_config)
        
        source = temp_dir / "source.txt" 
        source.write_text("content")
        target = temp_dir / "target.txt"
        
        call_count = 0
        original_move = manager._move_file_atomic
        
        def mock_move(src, tgt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ArchiveError("Temporary failure", error_type="filesystem")
            original_move(src, tgt)
        
        manager._move_file_atomic = mock_move
        
        manager._move_file_with_retry(source, target)
        
        assert not source.exists()
        assert target.exists()
        assert call_count == 2
    
    def test_move_file_with_retry_permission_error_no_retry(self, temp_dir: Path):
        """Test that permission errors are not retried."""
        retry_config = RetryConfig(max_attempts=3, delay=0.01)
        manager = ArchiveManager(retry_config)
        
        source = temp_dir / "source.txt"
        source.write_text("content")
        target = temp_dir / "target.txt"
        
        with patch.object(manager, '_move_file_atomic') as mock_move:
            mock_move.side_effect = ArchiveError("Permission denied", error_type="permission")
            
            with pytest.raises(ArchiveError) as exc_info:
                manager._move_file_with_retry(source, target)
            
            # Should only be called once (no retries for permission errors)
            assert mock_move.call_count == 1
            assert exc_info.value.error_type == "permission"
    
    def test_move_file_with_retry_all_attempts_fail(self, temp_dir: Path):
        """Test failure after all retry attempts are exhausted."""
        retry_config = RetryConfig(max_attempts=3, delay=0.01)
        manager = ArchiveManager(retry_config)
        
        source = temp_dir / "source.txt"
        source.write_text("content")
        target = temp_dir / "target.txt"
        
        with patch.object(manager, '_move_file_atomic') as mock_move:
            mock_move.side_effect = ArchiveError("Persistent failure", error_type="filesystem")
            
            with pytest.raises(ArchiveError) as exc_info:
                manager._move_file_with_retry(source, target)
            
            assert mock_move.call_count == 3
            assert exc_info.value.error_type == "filesystem"
    
    def test_is_transient_error_permission(self):
        """Test that permission errors are not considered transient."""
        manager = ArchiveManager()
        
        error = ArchiveError("Permission denied", error_type="permission")
        assert manager._is_transient_error(error) is False
    
    def test_is_transient_error_filesystem(self):
        """Test that filesystem errors are considered transient."""
        manager = ArchiveManager()
        
        error = ArchiveError("Temporary lock", error_type="filesystem")
        assert manager._is_transient_error(error) is True
    
    def test_validate_archive_path_valid(self, temp_dir: Path):
        """Test validation of valid archive path."""
        manager = ArchiveManager()
        
        assert manager.validate_archive_path(temp_dir) is True
    
    def test_validate_archive_path_nonexistent(self, temp_dir: Path):
        """Test validation of non-existent path."""
        manager = ArchiveManager()
        nonexistent = temp_dir / "nonexistent"
        
        assert manager.validate_archive_path(nonexistent) is False
    
    def test_validate_archive_path_not_directory(self, temp_dir: Path):
        """Test validation when path is not a directory."""
        manager = ArchiveManager()
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")
        
        assert manager.validate_archive_path(test_file) is False
    
    @patch('pathlib.Path.touch')
    @patch('pathlib.Path.unlink')
    def test_validate_archive_path_permission_error(self, mock_unlink, mock_touch, temp_dir: Path):
        """Test validation when write permission is denied."""
        manager = ArchiveManager()
        
        mock_touch.side_effect = PermissionError("Permission denied")
        
        assert manager.validate_archive_path(temp_dir) is False
    
    def test_archive_file_with_conflicts(self, temp_dir: Path):
        """Test complete archiving workflow with naming conflicts."""
        manager = ArchiveManager()
        
        # Create source file
        source_file = temp_dir / "test.xlsx"
        source_file.write_text("test content")
        
        # Create archive folder and conflicting file
        archive_dir = temp_dir / "archive"
        archive_dir.mkdir()
        conflicting_file = archive_dir / "test.xlsx"
        conflicting_file.write_text("existing content")
        
        archive_config = ArchiveConfig(
            enabled=True, 
            archive_folder_name="archive",
            handle_conflicts=True,
            timestamp_format="%Y%m%d_%H%M%S"
        )
        
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            result = manager.archive_file(source_file, archive_config)
        
        assert result.success is True
        assert not source_file.exists()  # Original moved
        assert conflicting_file.exists()  # Original conflict file unchanged
        
        # New file should have timestamp suffix
        archived_file = archive_dir / "test_20240101_120000.xlsx"
        assert archived_file.exists()
        assert archived_file.read_text() == "test content"
        assert result.archive_path == archived_file
    
    def test_archive_file_unexpected_error(self, temp_dir: Path):
        """Test handling of unexpected errors during archiving."""
        manager = ArchiveManager()
        
        source_file = temp_dir / "test.xlsx"
        source_file.write_text("content")
        
        archive_config = ArchiveConfig(enabled=True)
        
        # Mock an unexpected error in create_archive_folder
        with patch.object(manager, 'create_archive_folder') as mock_create:
            mock_create.side_effect = RuntimeError("Unexpected error")
            
            result = manager.archive_file(source_file, archive_config)
        
        assert result.success is False
        assert result.source_path == source_file
        assert "Unexpected error" in result.error_message
        assert result.operation_time > 0