"""Comprehensive tests for File Monitor with high coverage."""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import logging
import os

from excel_to_csv.monitoring.file_monitor import FileMonitor, ExcelFileHandler, FileMonitorError
from watchdog.events import FileCreatedEvent, FileModifiedEvent, DirCreatedEvent


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_excel_files(temp_workspace):
    """Create sample Excel files for testing."""
    files = []
    
    # Create Excel files
    xlsx_file = temp_workspace / "test1.xlsx"
    xlsx_file.write_text("Excel content")
    files.append(xlsx_file)
    
    xls_file = temp_workspace / "test2.xls"
    xls_file.write_text("Excel content")
    files.append(xls_file)
    
    # Create non-Excel file (should be ignored)
    txt_file = temp_workspace / "ignore.txt"
    txt_file.write_text("Text content")
    
    return files


@pytest.fixture
def mock_callback():
    """Create mock callback function."""
    return MagicMock()


class TestExcelFileHandler:
    """Test ExcelFileHandler class."""
    
    def test_handler_initialization(self, mock_callback):
        """Test ExcelFileHandler initialization."""
        file_patterns = ["*.xlsx", "*.xls"]
        debounce_seconds = 1.0
        
        handler = ExcelFileHandler(file_patterns, mock_callback, debounce_seconds)
        
        assert handler.file_patterns == file_patterns
        assert handler.callback == mock_callback
        assert handler.debounce_seconds == debounce_seconds
        assert hasattr(handler, 'logger')
        assert hasattr(handler, '_pending_files')
        assert hasattr(handler, '_processed_files')
        assert hasattr(handler, '_debounce_thread')
        assert handler._debounce_thread.daemon is True
    
    def test_handler_default_debounce(self, mock_callback):
        """Test ExcelFileHandler with default debounce time."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        assert handler.debounce_seconds == 2.0
    
    def test_on_created_excel_file(self, mock_callback, temp_workspace):
        """Test handling file creation events for Excel files."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback, debounce_seconds=0.1)
        
        # Create file event
        excel_file = temp_workspace / "new.xlsx"
        event = FileCreatedEvent(str(excel_file))
        
        handler.on_created(event)
        
        # Wait for debounce processing
        time.sleep(0.2)
        
        # Should be added to pending files
        assert excel_file in handler._pending_files
    
    def test_on_created_directory_ignored(self, mock_callback, temp_workspace):
        """Test that directory creation events are ignored."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        # Create directory event
        new_dir = temp_workspace / "newdir"
        event = DirCreatedEvent(str(new_dir))
        
        handler.on_created(event)
        
        # Should not be added to pending files
        assert new_dir not in handler._pending_files
    
    def test_on_created_non_excel_ignored(self, mock_callback, temp_workspace):
        """Test that non-Excel file creation is ignored."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        # Create non-Excel file event
        txt_file = temp_workspace / "document.txt"
        event = FileCreatedEvent(str(txt_file))
        
        handler.on_created(event)
        
        # Should not be added to pending files
        assert txt_file not in handler._pending_files
    
    def test_on_modified_excel_file(self, mock_callback, temp_workspace):
        """Test handling file modification events for Excel files."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback, debounce_seconds=0.1)
        
        # Create modification event
        excel_file = temp_workspace / "modified.xlsx"
        event = FileModifiedEvent(str(excel_file))
        
        handler.on_modified(event)
        
        # Wait for debounce processing
        time.sleep(0.2)
        
        # Should be added to pending files
        assert excel_file in handler._pending_files
    
    def test_on_modified_directory_ignored(self, mock_callback, temp_workspace):
        """Test that directory modification events are ignored."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        # Create directory modification event
        event = FileModifiedEvent(str(temp_workspace))
        event.is_directory = True
        
        handler.on_modified(event)
        
        # Should not be added to pending files
        assert len(handler._pending_files) == 0
    
    def test_matches_patterns_xlsx(self, mock_callback):
        """Test pattern matching for .xlsx files."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        assert handler._matches_patterns(Path("test.xlsx")) is True
        assert handler._matches_patterns(Path("TEST.XLSX")) is True  # Case insensitive
        assert handler._matches_patterns(Path("test.xls")) is False
        assert handler._matches_patterns(Path("test.txt")) is False
    
    def test_matches_patterns_multiple(self, mock_callback):
        """Test pattern matching with multiple patterns."""
        handler = ExcelFileHandler(["*.xlsx", "*.xls"], mock_callback)
        
        assert handler._matches_patterns(Path("test.xlsx")) is True
        assert handler._matches_patterns(Path("test.xls")) is True
        assert handler._matches_patterns(Path("test.txt")) is False
    
    def test_matches_patterns_complex(self, mock_callback):
        """Test pattern matching with complex patterns."""
        handler = ExcelFileHandler(["data_*.xlsx", "report*.xls"], mock_callback)
        
        assert handler._matches_patterns(Path("data_file.xlsx")) is True
        assert handler._matches_patterns(Path("report123.xls")) is True
        assert handler._matches_patterns(Path("test.xlsx")) is False
        assert handler._matches_patterns(Path("data_file.xls")) is False
    
    def test_add_pending_file(self, mock_callback):
        """Test adding files to pending list."""
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        file_path = Path("test.xlsx")
        handler._add_pending_file(file_path)
        
        assert file_path in handler._pending_files
        assert isinstance(handler._pending_files[file_path], float)
    
    def test_process_pending_files_timeout(self, mock_callback, temp_workspace):
        """Test processing of pending files after timeout."""
        # Create actual file for processing
        excel_file = temp_workspace / "pending.xlsx"
        excel_file.write_text("Excel content")
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback, debounce_seconds=0.1)
        
        # Add file to pending
        handler._add_pending_file(excel_file)
        
        # Wait for debounce processing
        time.sleep(0.2)
        
        # Callback should have been called
        mock_callback.assert_called_with(excel_file)
    
    def test_process_file_accessible(self, mock_callback, temp_workspace):
        """Test processing accessible file."""
        excel_file = temp_workspace / "accessible.xlsx"
        excel_file.write_text("Excel content")
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        # Process file directly
        handler._process_file(excel_file)
        
        # Should call callback and mark as processed
        mock_callback.assert_called_once_with(excel_file)
        assert excel_file in handler._processed_files
    
    def test_process_file_not_accessible(self, mock_callback, temp_workspace, caplog):
        """Test processing file that's not accessible."""
        excel_file = temp_workspace / "nonexistent.xlsx"
        # Don't create the file
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        with caplog.at_level(logging.DEBUG):
            handler._process_file(excel_file)
        
        # Should not call callback and should log warning
        mock_callback.assert_not_called()
        assert "not accessible" in caplog.text
    
    def test_is_file_accessible_existing_file(self, mock_callback, temp_workspace):
        """Test accessibility check for existing file."""
        excel_file = temp_workspace / "exists.xlsx"
        excel_file.write_text("Excel content")
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        assert handler._is_file_accessible(excel_file) is True
    
    def test_is_file_accessible_nonexistent_file(self, mock_callback, temp_workspace):
        """Test accessibility check for non-existent file."""
        excel_file = temp_workspace / "nonexistent.xlsx"
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        assert handler._is_file_accessible(excel_file) is False
    
    def test_is_file_accessible_permission_error(self, mock_callback, temp_workspace):
        """Test accessibility check with permission error."""
        excel_file = temp_workspace / "permission.xlsx"
        excel_file.write_text("Excel content")
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        # Mock stat to raise permission error
        with patch('pathlib.Path.stat', side_effect=PermissionError("Access denied")):
            assert handler._is_file_accessible(excel_file) is False
    
    def test_process_existing_file(self, mock_callback, temp_workspace):
        """Test processing existing file without duplication."""
        excel_file = temp_workspace / "existing.xlsx"
        excel_file.write_text("Excel content")
        
        handler = ExcelFileHandler(["*.xlsx"], mock_callback)
        
        # Process file twice
        handler.process_existing_file(excel_file)
        handler.process_existing_file(excel_file)
        
        # Should only be called once
        mock_callback.assert_called_once_with(excel_file)


class TestFileMonitor:
    """Test FileMonitor class."""
    
    def test_monitor_initialization(self, temp_workspace, mock_callback):
        """Test FileMonitor initialization."""
        folders = [temp_workspace]
        file_patterns = ["*.xlsx"]
        
        monitor = FileMonitor(folders, file_patterns, mock_callback)
        
        assert monitor.folders == folders
        assert monitor.file_patterns == file_patterns
        assert monitor.callback == mock_callback
        assert monitor.process_existing is True
        assert monitor.debounce_seconds == 2.0
        assert monitor.is_monitoring() is False
        assert hasattr(monitor, 'logger')
    
    def test_monitor_initialization_custom_params(self, temp_workspace, mock_callback):
        """Test FileMonitor initialization with custom parameters."""
        folders = [temp_workspace]
        file_patterns = ["*.xlsx"]
        
        monitor = FileMonitor(
            folders, 
            file_patterns, 
            mock_callback, 
            process_existing=False,
            debounce_seconds=1.0
        )
        
        assert monitor.process_existing is False
        assert monitor.debounce_seconds == 1.0
    
    def test_monitor_start_monitoring(self, temp_workspace, mock_callback):
        """Test starting file monitoring."""
        folders = [temp_workspace]
        
        monitor = FileMonitor(folders, ["*.xlsx"], mock_callback)
        
        with patch('excel_to_csv.monitoring.file_monitor.Observer') as mock_observer:
            mock_observer_instance = MagicMock()
            mock_observer.return_value = mock_observer_instance
            
            monitor.start_monitoring()
        
        assert monitor.is_monitoring() is True
        mock_observer_instance.start.assert_called_once()
    
    def test_monitor_start_already_monitoring(self, temp_workspace, mock_callback):
        """Test starting monitoring when already monitoring."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        monitor._monitoring = True
        
        with pytest.raises(FileMonitorError, match="already monitoring"):
            monitor.start_monitoring()
    
    def test_monitor_stop_monitoring(self, temp_workspace, mock_callback):
        """Test stopping file monitoring."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        # Mock observer
        mock_observer = MagicMock()
        monitor._observer = mock_observer
        monitor._monitoring = True
        
        monitor.stop_monitoring()
        
        assert monitor.is_monitoring() is False
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()
    
    def test_monitor_stop_not_monitoring(self, temp_workspace, mock_callback):
        """Test stopping monitoring when not monitoring."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        with pytest.raises(FileMonitorError, match="not currently monitoring"):
            monitor.stop_monitoring()
    
    def test_cleanup_monitoring(self, temp_workspace, mock_callback):
        """Test cleanup monitoring operation."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        # Mock observer
        mock_observer = MagicMock()
        monitor._observer = mock_observer
        monitor._monitoring = True
        
        monitor._cleanup_monitoring()
        
        assert monitor._observer is None
        assert monitor.is_monitoring() is False
    
    def test_validate_folders_existing(self, temp_workspace, mock_callback):
        """Test folder validation with existing folders."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        # Should not raise exception
        monitor._validate_folders()
    
    def test_validate_folders_nonexistent(self, temp_workspace, mock_callback):
        """Test folder validation with non-existent folders."""
        nonexistent = temp_workspace / "nonexistent"
        monitor = FileMonitor([nonexistent], ["*.xlsx"], mock_callback)
        
        with pytest.raises(FileMonitorError, match="does not exist"):
            monitor._validate_folders()
    
    def test_validate_folders_not_directory(self, temp_workspace, mock_callback):
        """Test folder validation with file instead of directory."""
        file_path = temp_workspace / "file.txt"
        file_path.write_text("content")
        
        monitor = FileMonitor([file_path], ["*.xlsx"], mock_callback)
        
        with pytest.raises(FileMonitorError, match="not a directory"):
            monitor._validate_folders()
    
    def test_scan_existing_files(self, temp_workspace, mock_callback, sample_excel_files):
        """Test scanning for existing files."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx", "*.xls"], mock_callback)
        
        monitor._scan_existing_files()
        
        # Should have called callback for each Excel file
        assert mock_callback.call_count == len(sample_excel_files)
        
        # Check that all Excel files were processed
        called_files = [call[0][0] for call in mock_callback.call_args_list]
        for excel_file in sample_excel_files:
            assert excel_file in called_files
    
    def test_scan_existing_files_disabled(self, temp_workspace, mock_callback, sample_excel_files):
        """Test that existing file scan can be disabled."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback, process_existing=False)
        
        monitor._scan_existing_files()
        
        # Should not call callback when process_existing is False
        mock_callback.assert_not_called()
    
    def test_add_folder_success(self, temp_workspace, mock_callback):
        """Test adding a new folder to monitor."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        new_folder = temp_workspace / "new_folder"
        new_folder.mkdir()
        
        with patch.object(monitor, '_observer') as mock_observer:
            monitor.add_folder(new_folder)
        
        assert new_folder in monitor.folders
        # Should schedule handler if monitoring
        if monitor.is_monitoring():
            mock_observer.schedule.assert_called()
    
    def test_add_folder_already_monitored(self, temp_workspace, mock_callback):
        """Test adding folder that's already being monitored."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        with pytest.raises(FileMonitorError, match="already being monitored"):
            monitor.add_folder(temp_workspace)
    
    def test_add_folder_nonexistent(self, temp_workspace, mock_callback):
        """Test adding non-existent folder."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        nonexistent = temp_workspace / "nonexistent"
        
        with pytest.raises(FileMonitorError, match="does not exist"):
            monitor.add_folder(nonexistent)
    
    def test_remove_folder_success(self, temp_workspace, mock_callback):
        """Test removing a monitored folder."""
        folder1 = temp_workspace / "folder1"
        folder2 = temp_workspace / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        monitor = FileMonitor([folder1, folder2], ["*.xlsx"], mock_callback)
        
        result = monitor.remove_folder(folder1)
        
        assert result is True
        assert folder1 not in monitor.folders
        assert folder2 in monitor.folders
    
    def test_remove_folder_not_monitored(self, temp_workspace, mock_callback):
        """Test removing folder that's not being monitored."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        other_folder = temp_workspace / "other"
        other_folder.mkdir()
        
        result = monitor.remove_folder(other_folder)
        
        assert result is False
        assert temp_workspace in monitor.folders
    
    def test_remove_folder_last_folder(self, temp_workspace, mock_callback):
        """Test removing the last monitored folder."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        with pytest.raises(FileMonitorError, match="Cannot remove the last monitored folder"):
            monitor.remove_folder(temp_workspace)
    
    def test_get_monitored_folders(self, temp_workspace, mock_callback):
        """Test getting list of monitored folders."""
        folder1 = temp_workspace / "folder1"
        folder2 = temp_workspace / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        monitor = FileMonitor([folder1, folder2], ["*.xlsx"], mock_callback)
        
        folders = monitor.get_monitored_folders()
        
        assert isinstance(folders, list)
        assert folder1 in folders
        assert folder2 in folders
        assert len(folders) == 2
    
    def test_get_statistics(self, temp_workspace, mock_callback):
        """Test getting monitoring statistics."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        stats = monitor.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'folders_count' in stats
        assert 'file_patterns' in stats
        assert 'is_monitoring' in stats
        assert 'process_existing' in stats
        assert 'debounce_seconds' in stats
        
        assert stats['folders_count'] == 1
        assert stats['file_patterns'] == ["*.xlsx"]
        assert stats['is_monitoring'] is False
        assert stats['process_existing'] is True
        assert stats['debounce_seconds'] == 2.0
    
    def test_context_manager_enter_exit(self, temp_workspace, mock_callback):
        """Test FileMonitor as context manager."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        with patch.object(monitor, 'start_monitoring') as mock_start:
            with patch.object(monitor, 'stop_monitoring') as mock_stop:
                with monitor as context_monitor:
                    assert context_monitor is monitor
                
                mock_start.assert_called_once()
                mock_stop.assert_called_once()
    
    def test_context_manager_exception_handling(self, temp_workspace, mock_callback):
        """Test context manager with exception."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        with patch.object(monitor, 'start_monitoring'):
            with patch.object(monitor, 'stop_monitoring') as mock_stop:
                try:
                    with monitor:
                        raise ValueError("Test exception")
                except ValueError:
                    pass
                
                # Should still call stop_monitoring
                mock_stop.assert_called_once()


class TestFileMonitorIntegration:
    """Test FileMonitor integration scenarios."""
    
    def test_monitor_with_real_files(self, temp_workspace, mock_callback):
        """Test monitoring with real file operations."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback, debounce_seconds=0.1)
        
        try:
            with patch('excel_to_csv.monitoring.file_monitor.Observer') as mock_observer:
                mock_observer_instance = MagicMock()
                mock_observer.return_value = mock_observer_instance
                
                monitor.start_monitoring()
                
                # Create a file
                excel_file = temp_workspace / "new_file.xlsx"
                excel_file.write_text("Excel content")
                
                # Process existing files manually (since we're mocking the observer)
                monitor._scan_existing_files()
                
                # Should have processed the file
                mock_callback.assert_called_with(excel_file)
        
        finally:
            if monitor.is_monitoring():
                monitor.stop_monitoring()
    
    def test_monitor_multiple_folders(self, temp_workspace, mock_callback):
        """Test monitoring multiple folders."""
        folder1 = temp_workspace / "folder1"
        folder2 = temp_workspace / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create files in both folders
        file1 = folder1 / "file1.xlsx"
        file2 = folder2 / "file2.xlsx"
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        monitor = FileMonitor([folder1, folder2], ["*.xlsx"], mock_callback)
        
        monitor._scan_existing_files()
        
        # Should have processed files from both folders
        assert mock_callback.call_count == 2
        called_files = [call[0][0] for call in mock_callback.call_args_list]
        assert file1 in called_files
        assert file2 in called_files
    
    def test_monitor_error_handling(self, temp_workspace, mock_callback):
        """Test error handling in file monitoring."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        # Test with observer start failure
        with patch('excel_to_csv.monitoring.file_monitor.Observer') as mock_observer:
            mock_observer_instance = MagicMock()
            mock_observer_instance.start.side_effect = RuntimeError("Observer error")
            mock_observer.return_value = mock_observer_instance
            
            with pytest.raises(FileMonitorError, match="Failed to start file monitoring"):
                monitor.start_monitoring()
    
    def test_monitor_logging(self, temp_workspace, mock_callback, caplog):
        """Test that monitoring operations are logged."""
        monitor = FileMonitor([temp_workspace], ["*.xlsx"], mock_callback)
        
        with caplog.at_level(logging.INFO):
            with patch('excel_to_csv.monitoring.file_monitor.Observer'):
                monitor.start_monitoring()
        
        log_messages = [record.message for record in caplog.records]
        assert any("Started monitoring" in msg for msg in log_messages)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_file_monitor_error_exception(self):
        """Test FileMonitorError exception."""
        error = FileMonitorError("Test error message")
        
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_handler_with_callback_exception(self, temp_workspace):
        """Test handler behavior when callback raises exception."""
        def failing_callback(file_path):
            raise RuntimeError("Callback error")
        
        excel_file = temp_workspace / "test.xlsx"
        excel_file.write_text("Content")
        
        handler = ExcelFileHandler(["*.xlsx"], failing_callback, debounce_seconds=0.1)
        
        # Should handle callback exception gracefully
        handler._process_file(excel_file)
        
        # File should still be marked as processed
        assert excel_file in handler._processed_files


if __name__ == "__main__":
    pytest.main([__file__])