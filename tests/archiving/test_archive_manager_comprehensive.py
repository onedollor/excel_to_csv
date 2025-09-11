"""Comprehensive tests for Archive Manager with high coverage."""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import logging
import os

from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.models.data_models import (
    ArchiveConfig, 
    ArchiveResult, 
    ArchiveError, 
    RetryConfig
)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_file(temp_workspace):
    """Create a sample file for archiving."""
    file_path = temp_workspace / "test_file.xlsx"
    file_path.write_text("Sample Excel content", encoding='utf-8')
    return file_path


@pytest.fixture
def archive_config():
    """Create default archive configuration."""
    return ArchiveConfig(
        enabled=True,
        archive_folder_name="archive",
        timestamp_format="%Y%m%d_%H%M%S",
        handle_conflicts=True,
        preserve_structure=True
    )


@pytest.fixture
def retry_config():
    """Create retry configuration."""
    return RetryConfig(
        max_attempts=3,
        delay=0.1,  # Short delay for tests
        backoff_factor=2.0,
        max_delay=1.0
    )


class TestArchiveManagerInitialization:
    """Test Archive Manager initialization."""
    
    def test_initialization_with_defaults(self):
        """Test ArchiveManager initialization with default retry config."""
        manager = ArchiveManager()
        
        assert manager is not None
        assert hasattr(manager, 'logger')
        assert hasattr(manager, 'retry_config')
        assert manager.retry_config.max_attempts == 3
        assert manager.retry_config.delay == 1.0
        assert manager.retry_config.backoff_factor == 2.0
        assert manager.retry_config.max_delay == 10.0
    
    def test_initialization_with_custom_retry_config(self, retry_config):
        """Test ArchiveManager initialization with custom retry config."""
        manager = ArchiveManager(retry_config=retry_config)
        
        assert manager.retry_config == retry_config
        assert manager.retry_config.max_attempts == 3
        assert manager.retry_config.delay == 0.1


class TestArchiveFileOperation:
    """Test main archive file operation."""
    
    def test_archive_disabled_returns_success(self, sample_file):
        """Test that archiving disabled returns success without moving file."""
        manager = ArchiveManager()
        config = ArchiveConfig(enabled=False)
        
        result = manager.archive_file(sample_file, config)
        
        assert result.success is True
        assert result.source_path == sample_file
        assert result.archive_path is None
        assert result.error_message is None
        assert result.operation_time >= 0
        assert sample_file.exists()  # File should still exist
    
    def test_archive_file_success_basic(self, sample_file, archive_config):
        """Test successful basic file archiving."""
        manager = ArchiveManager()
        
        result = manager.archive_file(sample_file, archive_config)
        
        assert result.success is True
        assert result.source_path == sample_file
        assert result.archive_path is not None
        assert result.error_message is None
        assert result.operation_time >= 0
        
        # Check file was moved correctly
        assert not sample_file.exists()
        assert result.archive_path.exists()
        assert result.archive_path.name == "test_file.xlsx"
        assert result.archive_path.parent.name == "archive"
    
    def test_archive_file_success_with_content_verification(self, temp_workspace, archive_config):
        """Test archiving with content verification."""
        # Create file with specific content
        original_content = "This is test content for verification"
        test_file = temp_workspace / "content_test.xlsx"
        test_file.write_text(original_content, encoding='utf-8')
        
        manager = ArchiveManager()
        result = manager.archive_file(test_file, archive_config)
        
        assert result.success is True
        assert result.archive_path.read_text(encoding='utf-8') == original_content
    
    def test_archive_nonexistent_file_fails(self, temp_workspace, archive_config):
        """Test archiving nonexistent file fails with proper error."""
        nonexistent_file = temp_workspace / "does_not_exist.xlsx"
        manager = ArchiveManager()
        
        result = manager.archive_file(nonexistent_file, archive_config)
        
        assert result.success is False
        assert result.source_path == nonexistent_file
        assert result.archive_path is None
        assert "does not exist" in result.error_message
        assert result.operation_time >= 0
    
    def test_archive_directory_instead_of_file_fails(self, temp_workspace, archive_config):
        """Test archiving directory instead of file fails."""
        directory_path = temp_workspace / "test_directory"
        directory_path.mkdir()
        
        manager = ArchiveManager()
        result = manager.archive_file(directory_path, archive_config)
        
        assert result.success is False
        assert "not a file" in result.error_message
    
    def test_archive_file_with_conflict_resolution(self, temp_workspace, archive_config):
        """Test archive file with naming conflict resolution."""
        # Create original file
        original_file = temp_workspace / "conflict_test.xlsx"
        original_file.write_text("Original content", encoding='utf-8')
        
        # Create archive folder and existing file
        archive_folder = temp_workspace / "archive"
        archive_folder.mkdir()
        existing_file = archive_folder / "conflict_test.xlsx"
        existing_file.write_text("Existing content", encoding='utf-8')
        
        manager = ArchiveManager()
        result = manager.archive_file(original_file, archive_config)
        
        assert result.success is True
        assert result.archive_path != existing_file  # Should be different
        assert result.archive_path.name != "conflict_test.xlsx"  # Should have timestamp
        assert "conflict_test" in result.archive_path.name
        assert result.archive_path.exists()
        assert result.archive_path.read_text(encoding='utf-8') == "Original content"
    
    def test_archive_file_conflicts_disabled(self, temp_workspace):
        """Test archiving with conflict handling disabled."""
        # Create original file
        original_file = temp_workspace / "no_conflict_test.xlsx"
        original_file.write_text("Original content", encoding='utf-8')
        
        # Create archive folder and existing file
        archive_folder = temp_workspace / "archive"
        archive_folder.mkdir()
        existing_file = archive_folder / "no_conflict_test.xlsx"
        existing_file.write_text("Existing content", encoding='utf-8')
        
        config = ArchiveConfig(
            enabled=True,
            archive_folder_name="archive",
            handle_conflicts=False
        )
        
        manager = ArchiveManager()
        result = manager.archive_file(original_file, config)
        
        assert result.success is True
        # File should be overwritten
        assert result.archive_path == existing_file
        assert result.archive_path.read_text(encoding='utf-8') == "Original content"


class TestArchiveFolderCreation:
    """Test archive folder creation functionality."""
    
    def test_create_archive_folder_new(self, temp_workspace):
        """Test creating new archive folder."""
        manager = ArchiveManager()
        
        archive_folder = manager.create_archive_folder(temp_workspace, "new_archive")
        
        assert archive_folder.exists()
        assert archive_folder.is_dir()
        assert archive_folder.name == "new_archive"
        assert archive_folder.parent == temp_workspace
    
    def test_create_archive_folder_existing(self, temp_workspace):
        """Test creating archive folder that already exists."""
        # Pre-create folder
        existing_folder = temp_workspace / "existing_archive"
        existing_folder.mkdir()
        
        manager = ArchiveManager()
        archive_folder = manager.create_archive_folder(temp_workspace, "existing_archive")
        
        assert archive_folder == existing_folder
        assert archive_folder.exists()
        assert archive_folder.is_dir()
    
    def test_create_archive_folder_file_exists_with_same_name(self, temp_workspace):
        """Test error when file exists with same name as archive folder."""
        # Create file with same name as desired folder
        conflicting_file = temp_workspace / "conflicting_archive"
        conflicting_file.write_text("I am a file, not a folder!")
        
        manager = ArchiveManager()
        
        with pytest.raises(ArchiveError, match="not a directory"):
            manager.create_archive_folder(temp_workspace, "conflicting_archive")
    
    def test_create_archive_folder_permission_error(self, temp_workspace):
        """Test archive folder creation with permission error."""
        manager = ArchiveManager()
        
        # Mock mkdir to raise permission error
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            with pytest.raises(ArchiveError, match="Permission denied"):
                manager.create_archive_folder(temp_workspace, "no_permission")
    
    def test_create_archive_folder_os_error(self, temp_workspace):
        """Test archive folder creation with OS error."""
        manager = ArchiveManager()
        
        # Mock mkdir to raise OS error
        with patch('pathlib.Path.mkdir', side_effect=OSError("Disk full")):
            with pytest.raises(ArchiveError, match="Failed to create"):
                manager.create_archive_folder(temp_workspace, "disk_full")
    
    def test_create_archive_folder_nested_path(self, temp_workspace):
        """Test creating archive folder with nested path."""
        manager = ArchiveManager()
        
        # Create nested structure
        nested_base = temp_workspace / "level1" / "level2"
        nested_base.mkdir(parents=True)
        
        archive_folder = manager.create_archive_folder(nested_base, "deep_archive")
        
        assert archive_folder.exists()
        assert archive_folder == nested_base / "deep_archive"


class TestNamingConflictResolution:
    """Test naming conflict resolution functionality."""
    
    def test_resolve_naming_conflicts_basic(self, temp_workspace):
        """Test basic naming conflict resolution."""
        # Create existing file
        existing_file = temp_workspace / "existing.xlsx"
        existing_file.write_text("Existing content")
        
        manager = ArchiveManager()
        resolved_path = manager.resolve_naming_conflicts(existing_file, "%Y%m%d_%H%M%S")
        
        assert resolved_path != existing_file
        assert resolved_path.parent == existing_file.parent
        assert resolved_path.suffix == existing_file.suffix
        assert "existing_" in resolved_path.name  # Should contain timestamp
        assert not resolved_path.exists()  # Resolved path should not exist
    
    def test_resolve_naming_conflicts_multiple_conflicts(self, temp_workspace):
        """Test resolving multiple naming conflicts."""
        # Create multiple conflicting files
        base_name = "multi_conflict.xlsx"
        original_file = temp_workspace / base_name
        original_file.write_text("Original")
        
        # Mock datetime to return consistent timestamp
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            # Create first conflict
            conflict1 = temp_workspace / "multi_conflict_20240101_120000.xlsx"
            conflict1.write_text("Conflict 1")
            
            # Create second conflict  
            conflict2 = temp_workspace / "multi_conflict_20240101_120000_001.xlsx"
            conflict2.write_text("Conflict 2")
            
            manager = ArchiveManager()
            resolved_path = manager.resolve_naming_conflicts(original_file, "%Y%m%d_%H%M%S")
            
            assert resolved_path.name == "multi_conflict_20240101_120000_002.xlsx"
            assert not resolved_path.exists()
    
    def test_resolve_naming_conflicts_excessive_conflicts(self, temp_workspace):
        """Test handling excessive naming conflicts."""
        original_file = temp_workspace / "excessive.xlsx"
        original_file.write_text("Original")
        
        # Mock datetime and file existence to simulate 1000+ conflicts
        with patch('excel_to_csv.archiving.archive_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            with patch('pathlib.Path.exists', return_value=True):  # Always conflicts
                manager = ArchiveManager()
                
                with pytest.raises(ArchiveError, match="Too many naming conflicts"):
                    manager.resolve_naming_conflicts(original_file, "%Y%m%d_%H%M%S")
    
    def test_resolve_naming_conflicts_custom_timestamp_format(self, temp_workspace):
        """Test conflict resolution with custom timestamp format."""
        existing_file = temp_workspace / "custom_format.xlsx"
        existing_file.write_text("Existing")
        
        manager = ArchiveManager()
        resolved_path = manager.resolve_naming_conflicts(existing_file, "%Y-%m-%d_%H-%M-%S")
        
        assert resolved_path != existing_file
        # Should contain custom timestamp format pattern
        assert "-" in resolved_path.name  # Custom format uses hyphens


class TestFileMovingOperations:
    """Test file moving operations."""
    
    def test_move_file_atomic_success(self, temp_workspace):
        """Test successful atomic file move."""
        source_file = temp_workspace / "source.xlsx"
        source_file.write_text("Source content")
        
        target_file = temp_workspace / "target.xlsx"
        
        manager = ArchiveManager()
        manager._move_file_atomic(source_file, target_file)
        
        assert not source_file.exists()
        assert target_file.exists()
        assert target_file.read_text() == "Source content"
    
    def test_move_file_atomic_permission_error(self, temp_workspace):
        """Test atomic file move with permission error."""
        source_file = temp_workspace / "source.xlsx"
        source_file.write_text("Source content")
        
        target_file = temp_workspace / "target.xlsx"
        
        manager = ArchiveManager()
        
        # Mock Path.replace to raise permission error
        with patch('pathlib.Path.replace', side_effect=PermissionError("Access denied")):
            with pytest.raises(ArchiveError, match="Permission denied"):
                manager._move_file_atomic(source_file, target_file)
    
    def test_move_file_atomic_os_error(self, temp_workspace):
        """Test atomic file move with OS error."""
        source_file = temp_workspace / "source.xlsx"
        source_file.write_text("Source content")
        
        target_file = temp_workspace / "target.xlsx"
        
        manager = ArchiveManager()
        
        # Mock Path.replace to raise OS error
        with patch('pathlib.Path.replace', side_effect=OSError("Disk full")):
            with pytest.raises(ArchiveError, match="Failed to move"):
                manager._move_file_atomic(source_file, target_file)
    
    def test_move_file_with_retry_success_first_attempt(self, temp_workspace):
        """Test file move with retry succeeding on first attempt."""
        source_file = temp_workspace / "retry_source.xlsx"
        source_file.write_text("Retry content")
        
        target_file = temp_workspace / "retry_target.xlsx"
        
        manager = ArchiveManager()
        manager._move_file_with_retry(source_file, target_file)
        
        assert not source_file.exists()
        assert target_file.exists()
        assert target_file.read_text() == "Retry content"
    
    def test_move_file_with_retry_success_after_failures(self, temp_workspace, retry_config):
        """Test file move with retry succeeding after initial failures."""
        source_file = temp_workspace / "retry_after_fail.xlsx"
        source_file.write_text("Retry after fail content")
        
        target_file = temp_workspace / "retry_target.xlsx"
        
        manager = ArchiveManager(retry_config)
        
        # Mock _move_file_atomic to fail twice, then succeed
        call_count = 0
        def mock_move_side_effect(source, target):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ArchiveError("Temporary failure", error_type="filesystem")
            # Success on third call
        
        with patch.object(manager, '_move_file_atomic', side_effect=mock_move_side_effect):
            manager._move_file_with_retry(source_file, target_file)
        
        assert call_count == 3  # Should have tried 3 times
    
    def test_move_file_with_retry_permission_error_no_retry(self, temp_workspace, retry_config):
        """Test that permission errors are not retried."""
        source_file = temp_workspace / "no_retry.xlsx"
        source_file.write_text("No retry content")
        
        target_file = temp_workspace / "no_retry_target.xlsx"
        
        manager = ArchiveManager(retry_config)
        
        # Mock to always raise permission error
        def mock_move_side_effect(source, target):
            raise ArchiveError("Permission denied", error_type="permission")
        
        with patch.object(manager, '_move_file_atomic', side_effect=mock_move_side_effect):
            with pytest.raises(ArchiveError, match="Permission denied"):
                manager._move_file_with_retry(source_file, target_file)
    
    def test_move_file_with_retry_max_attempts_exceeded(self, temp_workspace, retry_config):
        """Test file move failing after max retry attempts."""
        source_file = temp_workspace / "max_fail.xlsx"
        source_file.write_text("Max fail content")
        
        target_file = temp_workspace / "max_fail_target.xlsx"
        
        manager = ArchiveManager(retry_config)
        
        # Mock to always fail with filesystem error
        def mock_move_side_effect(source, target):
            raise ArchiveError("Persistent failure", error_type="filesystem")
        
        with patch.object(manager, '_move_file_atomic', side_effect=mock_move_side_effect):
            with pytest.raises(ArchiveError, match="Persistent failure"):
                manager._move_file_with_retry(source_file, target_file)


class TestTransientErrorDetection:
    """Test transient error detection logic."""
    
    def test_is_transient_error_permission_false(self):
        """Test that permission errors are not considered transient."""
        manager = ArchiveManager()
        
        perm_error = ArchiveError("Permission denied", error_type="permission")
        assert manager._is_transient_error(perm_error) is False
    
    def test_is_transient_error_filesystem_true(self):
        """Test that filesystem errors are considered transient."""
        manager = ArchiveManager()
        
        fs_error = ArchiveError("Disk busy", error_type="filesystem")
        assert manager._is_transient_error(fs_error) is True
    
    def test_is_transient_error_unknown_archive_error(self):
        """Test handling of ArchiveError without error_type."""
        manager = ArchiveManager()
        
        unknown_error = ArchiveError("Unknown error")
        # Should not crash, should return False for unknown type
        assert manager._is_transient_error(unknown_error) is False
    
    def test_is_transient_error_windows_sharing_violation(self):
        """Test Windows sharing violation detection."""
        manager = ArchiveManager()
        
        # Mock Windows error with sharing violation
        win_error = OSError("Sharing violation")
        win_error.winerror = 32  # Sharing violation
        
        assert manager._is_transient_error(win_error) is True
    
    def test_is_transient_error_unix_busy_error(self):
        """Test UNIX busy error detection.""" 
        manager = ArchiveManager()
        
        # Mock UNIX error with EBUSY
        unix_error = OSError("Resource busy")
        unix_error.errno = 16  # EBUSY
        
        assert manager._is_transient_error(unix_error) is True
    
    def test_is_transient_error_non_transient_os_error(self):
        """Test non-transient OS error detection."""
        manager = ArchiveManager()
        
        non_transient = OSError("No such file")
        non_transient.errno = 2  # ENOENT - not transient
        
        assert manager._is_transient_error(non_transient) is False
    
    def test_is_transient_error_generic_exception(self):
        """Test that generic exceptions are not considered transient."""
        manager = ArchiveManager()
        
        generic_error = ValueError("Invalid value")
        assert manager._is_transient_error(generic_error) is False


class TestArchivePathValidation:
    """Test archive path validation functionality."""
    
    def test_validate_archive_path_valid_directory(self, temp_workspace):
        """Test validation of valid directory."""
        manager = ArchiveManager()
        
        assert manager.validate_archive_path(temp_workspace) is True
    
    def test_validate_archive_path_nonexistent(self, temp_workspace):
        """Test validation of nonexistent path."""
        manager = ArchiveManager()
        
        nonexistent = temp_workspace / "does_not_exist"
        assert manager.validate_archive_path(nonexistent) is False
    
    def test_validate_archive_path_file_not_directory(self, temp_workspace):
        """Test validation of file instead of directory."""
        file_path = temp_workspace / "is_file.txt"
        file_path.write_text("I am a file")
        
        manager = ArchiveManager()
        
        assert manager.validate_archive_path(file_path) is False
    
    def test_validate_archive_path_write_permission_test(self, temp_workspace):
        """Test validation includes write permission test."""
        manager = ArchiveManager()
        
        # This should succeed in temp directory
        assert manager.validate_archive_path(temp_workspace) is True
    
    def test_validate_archive_path_no_write_permission(self, temp_workspace):
        """Test validation fails when no write permission."""
        manager = ArchiveManager()
        
        # Mock Path.touch to raise permission error
        with patch('pathlib.Path.touch', side_effect=PermissionError("No write access")):
            assert manager.validate_archive_path(temp_workspace) is False
    
    def test_validate_archive_path_os_error_during_test(self, temp_workspace):
        """Test validation handles OS error during write test."""
        manager = ArchiveManager()
        
        # Mock Path.touch to raise OS error
        with patch('pathlib.Path.touch', side_effect=OSError("I/O error")):
            assert manager.validate_archive_path(temp_workspace) is False
    
    def test_validate_archive_path_exception_handling(self, temp_workspace):
        """Test validation handles general exceptions."""
        manager = ArchiveManager()
        
        # Mock Path.exists to raise unexpected exception
        with patch('pathlib.Path.exists', side_effect=RuntimeError("Unexpected error")):
            assert manager.validate_archive_path(temp_workspace) is False


class TestArchiveResultStructure:
    """Test ArchiveResult structure and behavior."""
    
    def test_archive_result_success_structure(self, sample_file, archive_config):
        """Test structure of successful ArchiveResult."""
        manager = ArchiveManager()
        result = manager.archive_file(sample_file, archive_config)
        
        # Check all expected fields are present
        assert hasattr(result, 'success')
        assert hasattr(result, 'source_path')
        assert hasattr(result, 'archive_path')
        assert hasattr(result, 'error_message')
        assert hasattr(result, 'operation_time')
        
        # Check success case values
        assert result.success is True
        assert result.source_path == sample_file
        assert result.archive_path is not None
        assert result.error_message is None
        assert isinstance(result.operation_time, (int, float))
        assert result.operation_time >= 0
    
    def test_archive_result_failure_structure(self, temp_workspace, archive_config):
        """Test structure of failed ArchiveResult."""
        nonexistent_file = temp_workspace / "does_not_exist.xlsx"
        
        manager = ArchiveManager()
        result = manager.archive_file(nonexistent_file, archive_config)
        
        # Check failure case values
        assert result.success is False
        assert result.source_path == nonexistent_file
        assert result.archive_path is None
        assert result.error_message is not None
        assert isinstance(result.operation_time, (int, float))
        assert result.operation_time >= 0


class TestArchiveManagerLogging:
    """Test Archive Manager logging behavior."""
    
    def test_initialization_logging(self, caplog):
        """Test that initialization is properly logged."""
        with caplog.at_level(logging.INFO):
            ArchiveManager()
        
        # Check that initialization success is logged
        log_messages = [record.message for record in caplog.records]
        assert any("Archive Manager initialized successfully" in msg for msg in log_messages)
    
    def test_archive_disabled_logging(self, sample_file, caplog):
        """Test logging when archiving is disabled."""
        config = ArchiveConfig(enabled=False)
        manager = ArchiveManager()
        
        with caplog.at_level(logging.DEBUG):
            manager.archive_file(sample_file, config)
        
        log_messages = [record.message for record in caplog.records]
        assert any("Archiving disabled" in msg for msg in log_messages)
    
    def test_successful_archive_logging(self, sample_file, archive_config, caplog):
        """Test logging for successful archive operations."""
        manager = ArchiveManager()
        
        with caplog.at_level(logging.INFO):
            result = manager.archive_file(sample_file, archive_config)
        
        log_messages = [record.message for record in caplog.records]
        success_logged = any("Successfully archived file" in msg for msg in log_messages)
        assert success_logged
    
    def test_error_logging(self, temp_workspace, archive_config, caplog):
        """Test error logging for failed operations."""
        nonexistent_file = temp_workspace / "does_not_exist.xlsx"
        manager = ArchiveManager()
        
        with caplog.at_level(logging.ERROR):
            manager.archive_file(nonexistent_file, archive_config)
        
        log_messages = [record.message for record in caplog.records]
        error_logged = any("Archive operation failed" in msg for msg in log_messages)
        assert error_logged


if __name__ == "__main__":
    pytest.main([__file__])