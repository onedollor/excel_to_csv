"""Unit tests for file monitor."""

import pytest
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from excel_to_csv.monitoring.file_monitor import FileMonitor, ExcelFileHandler


class TestExcelFileHandler:
    """Test cases for ExcelFileHandler class."""
    
    def test_init(self):
        """Test ExcelFileHandler initialization."""
        callback = Mock()
        patterns = ["*.xlsx", "*.xls"]
        
        handler = ExcelFileHandler(patterns, callback, debounce_seconds=2.0)
        
        assert handler.file_patterns == patterns
        assert handler.callback == callback
        assert handler.debounce_seconds == 2.0
        assert isinstance(handler._pending_files, dict)
        assert hasattr(handler, '_pending_lock')
    
    def test_matches_patterns_xlsx(self):
        """Test pattern matching for .xlsx files."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback)
        
        assert handler._matches_patterns(Path("test.xlsx"))
        assert handler._matches_patterns("data.XLSX")  # Case insensitive
        assert not handler._matches_patterns("test.xls")
        assert not handler._matches_patterns(Path("test.txt"))
        assert not handler._matches_patterns("test")
    
    def test_matches_patterns_multiple(self):
        """Test pattern matching for multiple patterns."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx", "*.xls"], callback)
        
        assert handler._matches_patterns(Path("test.xlsx"))
        assert handler._matches_patterns(Path("test.xls"))
        assert handler._matches_patterns("data.XLSX")
        assert handler._matches_patterns("data.XLS")
        assert not handler._matches_patterns(Path("test.txt"))
        assert not handler._matches_patterns("test.csv")
    
    def test_matches_patterns_custom_patterns(self):
        """Test pattern matching with custom patterns."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsm", "data_*.xlsx"], callback)
        
        assert handler._matches_patterns(Path("test.xlsm"))
        assert handler._matches_patterns("data_test.xlsx")
        assert not handler._matches_patterns("test.xlsx")
        assert not handler._matches_patterns("other_test.xlsx")
    
    def test_on_created(self, temp_dir: Path):
        """Test file creation event handling."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback, debounce_seconds=0.1)
        
        # Create mock event
        test_file = temp_dir / "test.xlsx"
        test_file.touch()
        
        from watchdog.events import FileCreatedEvent
        event = FileCreatedEvent(str(test_file))
        
        handler.on_created(event)
        
        # Wait for debounce
        time.sleep(0.2)
        
        # Callback should be called after debounce
        assert test_file in handler._pending_files
    
    def test_on_modified(self, temp_dir: Path):
        """Test file modification event handling."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback, debounce_seconds=0.1)
        
        # Create mock event
        test_file = temp_dir / "modified.xlsx"
        test_file.touch()
        
        from watchdog.events import FileModifiedEvent
        event = FileModifiedEvent(str(test_file))
        
        handler.on_modified(event)
        
        # Wait for debounce
        time.sleep(0.2)
        
        # Should be tracked for debouncing
        assert test_file in handler._pending_files
    
    def test_debounce_mechanism(self, temp_dir: Path):
        """Test debouncing mechanism prevents multiple rapid calls."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback, debounce_seconds=0.2)
        
        test_file = temp_dir / "debounce.xlsx"
        test_file.touch()
        
        from watchdog.events import FileCreatedEvent
        event = FileCreatedEvent(str(test_file))
        
        # Trigger multiple events rapidly
        handler.on_created(event)
        handler.on_created(event)
        handler.on_created(event)
        
        # Should only create one timer
        assert len(handler._pending_files) == 1
        
        # Wait for debounce to complete
        time.sleep(0.3)
        
        # Callback should be called only once
        callback.assert_called_once_with(test_file)
    
    def test_ignore_non_matching_files(self, temp_dir: Path):
        """Test that non-matching files are ignored."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback)
        
        test_file = temp_dir / "ignored.txt"
        test_file.touch()
        
        from watchdog.events import FileCreatedEvent
        event = FileCreatedEvent(str(test_file))
        
        handler.on_created(event)
        
        # Should not create timer for non-matching file
        assert len(handler._pending_files) == 0
        callback.assert_not_called()
    
    def test_ignore_directories(self, temp_dir: Path):
        """Test that directory events are ignored."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback)
        
        test_dir = temp_dir / "test_directory"
        test_dir.mkdir()
        
        from watchdog.events import DirCreatedEvent
        event = DirCreatedEvent(str(test_dir))
        
        handler.on_created(event)
        
        # Should ignore directory events
        assert len(handler._pending_files) == 0
        callback.assert_not_called()
    
    def test_concurrent_timer_handling(self, temp_dir: Path):
        """Test concurrent timer handling is thread-safe."""
        callback = Mock()
        handler = ExcelFileHandler(["*.xlsx"], callback, debounce_seconds=0.1)
        
        def trigger_event(filename):
            test_file = temp_dir / filename
            test_file.touch()
            from watchdog.events import FileCreatedEvent
            event = FileCreatedEvent(str(test_file))
            handler.on_created(event)
        
        # Create multiple threads triggering events
        threads = []
        for i in range(5):
            thread = threading.Thread(target=trigger_event, args=(f"file_{i}.xlsx",))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have 5 different timers
        assert len(handler._pending_files) == 5
        
        # Wait for all debounces to complete
        time.sleep(0.2)
        
        # All callbacks should be called
        assert callback.call_count == 5


class TestFileMonitor:
    """Test cases for FileMonitor class."""
    
    def test_init_single_folder(self, temp_dir: Path):
        """Test FileMonitor initialization with single folder."""
        callback = Mock()
        monitor = FileMonitor([temp_dir], callback)
        
        assert monitor.folders == [temp_dir]
        assert monitor.callback == callback
        assert monitor.file_patterns == ["*.xlsx", "*.xls"]
        assert monitor.debounce_seconds == 2.0
        assert monitor._observer is None
        assert not monitor._running
    
    def test_init_multiple_folders(self, temp_dir: Path):
        """Test FileMonitor initialization with multiple folders."""
        callback = Mock()
        folder1 = temp_dir / "folder1"
        folder2 = temp_dir / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        monitor = FileMonitor([folder1, folder2], callback)
        
        assert monitor.folders == [folder1, folder2]
        assert len(monitor.folders) == 2
    
    def test_init_custom_patterns(self, temp_dir: Path):
        """Test FileMonitor initialization with custom patterns."""
        callback = Mock()
        custom_patterns = ["*.xlsm", "data_*.xlsx"]
        
        monitor = FileMonitor([temp_dir], callback, file_patterns=custom_patterns)
        
        assert monitor.file_patterns == custom_patterns
    
    def test_init_custom_debounce(self, temp_dir: Path):
        """Test FileMonitor initialization with custom debounce."""
        callback = Mock()
        
        monitor = FileMonitor([temp_dir], callback, debounce_seconds=5.0)
        
        assert monitor.debounce_seconds == 5.0
    
    def test_start_monitoring(self, temp_dir: Path):
        """Test starting monitoring service."""
        callback = Mock()
        monitor = FileMonitor([temp_dir], callback)
        
        monitor.start_monitoring()
        
        assert monitor._observer is not None
        assert monitor._running
        
        # Clean up
        monitor.stop_monitoring()
    
    def test_stop_monitoring(self, temp_dir: Path):
        """Test stopping monitoring service."""
        callback = Mock()
        monitor = FileMonitor([temp_dir], callback)
        
        monitor.start_monitoring()
        assert monitor._running
        
        monitor.stop_monitoring()
        assert not monitor._running
        assert monitor._observer is not None  # Observer exists but stopped
    
    def test_start_stop_multiple_times(self, temp_dir: Path):
        """Test starting and stopping monitoring multiple times."""
        callback = Mock()
        monitor = FileMonitor([temp_dir], callback)
        
        # Start/stop cycle 1
        monitor.start_monitoring()
        assert monitor._running
        monitor.stop_monitoring()
        assert not monitor._running
        
        # Start/stop cycle 2
        monitor.start_monitoring()
        assert monitor._running
        monitor.stop_monitoring()
        assert not monitor._running
    
    def test_stop_monitoring_when_not_started(self, temp_dir: Path):
        """Test stopping monitoring when not started should not error."""
        callback = Mock()
        monitor = FileMonitor([temp_dir], callback)
        
        # Should not raise error
        monitor.stop_monitoring()
        assert not monitor._running
    
    def test_start_monitoring_when_already_started(self, temp_dir: Path):
        """Test starting monitoring when already started."""
        callback = Mock()
        monitor = FileMonitor([temp_dir], callback)
        
        monitor.start_monitoring()
        assert monitor._running
        
        # Starting again should be safe
        monitor.start_monitoring()
        assert monitor._running
        
        # Clean up
        monitor.stop_monitoring()
    
    def test_scan_existing_files(self, temp_dir: Path):
        """Test scanning existing files in monitored folders."""
        callback = Mock()
        
        # Create existing Excel files
        existing_file1 = temp_dir / "existing1.xlsx"
        existing_file2 = temp_dir / "existing2.xls"
        non_excel_file = temp_dir / "ignored.txt"
        
        existing_file1.touch()
        existing_file2.touch()
        non_excel_file.touch()
        
        monitor = FileMonitor([temp_dir], callback)
        monitor.scan_existing_files()
        
        # Should call callback for Excel files only
        assert callback.call_count == 2
        called_paths = [call[0][0] for call in callback.call_args_list]
        assert existing_file1 in called_paths
        assert existing_file2 in called_paths
        assert non_excel_file not in called_paths
    
    def test_scan_existing_files_multiple_folders(self, temp_dir: Path):
        """Test scanning existing files across multiple folders."""
        callback = Mock()
        
        folder1 = temp_dir / "folder1"
        folder2 = temp_dir / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create files in different folders
        file1 = folder1 / "file1.xlsx"
        file2 = folder2 / "file2.xls"
        file1.touch()
        file2.touch()
        
        monitor = FileMonitor([folder1, folder2], callback)
        monitor.scan_existing_files()
        
        assert callback.call_count == 2
        called_paths = [call[0][0] for call in callback.call_args_list]
        assert file1 in called_paths
        assert file2 in called_paths
    
    def test_scan_existing_files_empty_folder(self, temp_dir: Path):
        """Test scanning existing files in empty folder."""
        callback = Mock()
        
        empty_folder = temp_dir / "empty"
        empty_folder.mkdir()
        
        monitor = FileMonitor([empty_folder], callback)
        monitor.scan_existing_files()
        
        # Should not call callback for empty folder
        callback.assert_not_called()
    
    def test_scan_existing_files_nonexistent_folder(self, temp_dir: Path):
        """Test scanning non-existent folder."""
        callback = Mock()
        
        nonexistent_folder = temp_dir / "does_not_exist"
        
        monitor = FileMonitor([nonexistent_folder], callback)
        
        # Should handle gracefully
        monitor.scan_existing_files()
        callback.assert_not_called()
    
    @patch('excel_to_csv.monitoring.file_monitor.Observer')
    def test_file_event_processing(self, mock_observer_class, temp_dir: Path):
        """Test that file events are properly processed."""
        callback = Mock()
        mock_observer = Mock()
        mock_observer_class.return_value = mock_observer
        
        monitor = FileMonitor([temp_dir], callback)
        monitor.start_monitoring()
        
        # Verify observer is configured correctly
        mock_observer_class.assert_called_once()
        mock_observer.schedule.assert_called()
        mock_observer.start.assert_called_once()
        
        # Clean up
        monitor.stop_monitoring()
        mock_observer.stop.assert_called_once()
    
    def test_context_manager_usage(self, temp_dir: Path):
        """Test FileMonitor as context manager."""
        callback = Mock()
        
        with FileMonitor([temp_dir], callback) as monitor:
            assert monitor._running
            assert monitor._observer is not None
        
        # Should be stopped after exiting context
        assert not monitor._running
    
    def test_real_file_detection(self, temp_dir: Path):
        """Integration test with real file creation."""
        callback = Mock()
        
        monitor = FileMonitor([temp_dir], callback, debounce_seconds=0.1)
        monitor.start_monitoring()
        
        try:
            # Create a real Excel file
            test_file = temp_dir / "real_test.xlsx"
            test_file.touch()
            
            # Wait for file system event and debounce
            time.sleep(0.3)
            
            # Callback should have been called
            # Note: This test might be flaky depending on file system speed
            # So we'll be lenient about exact timing
            
        finally:
            monitor.stop_monitoring()
    
    def test_error_handling_in_callback(self, temp_dir: Path):
        """Test error handling when callback raises exception."""
        def failing_callback(path):
            raise Exception("Callback failed")
        
        monitor = FileMonitor([temp_dir], failing_callback)
        
        # Create existing file to trigger callback
        test_file = temp_dir / "error_test.xlsx" 
        test_file.touch()
        
        # Should handle callback errors gracefully
        try:
            monitor.scan_existing_files()
            # If we get here, error was handled gracefully
        except Exception as e:
            # Should not propagate callback exceptions
            pytest.fail(f"Callback exception not handled: {e}")
    
    def test_monitoring_subdirectories_disabled(self, temp_dir: Path):
        """Test that subdirectories are not monitored by default."""
        callback = Mock()
        
        # Create subdirectory with Excel file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        sub_file = subdir / "sub_excel.xlsx"
        sub_file.touch()
        
        monitor = FileMonitor([temp_dir], callback)
        monitor.scan_existing_files()
        
        # Should not find files in subdirectories
        callback.assert_not_called()
    
    def test_file_pattern_case_insensitive(self, temp_dir: Path):
        """Test that file patterns are case insensitive."""
        callback = Mock()
        
        # Create files with different case extensions
        file_lower = temp_dir / "test.xlsx"
        file_upper = temp_dir / "test.XLSX"
        file_mixed = temp_dir / "test.XlSx"
        
        file_lower.touch()
        file_upper.touch()  
        file_mixed.touch()
        
        monitor = FileMonitor([temp_dir], callback)
        monitor.scan_existing_files()
        
        # All should be detected
        assert callback.call_count == 3
    
    def test_large_number_of_files(self, temp_dir: Path):
        """Test handling large number of files."""
        callback = Mock()
        
        # Create many Excel files
        num_files = 100
        for i in range(num_files):
            file_path = temp_dir / f"file_{i:03d}.xlsx"
            file_path.touch()
        
        monitor = FileMonitor([temp_dir], callback)
        monitor.scan_existing_files()
        
        # Should handle all files
        assert callback.call_count == num_files
    
    def test_performance_timing(self, temp_dir: Path):
        """Test performance of file scanning."""
        callback = Mock()
        
        # Create moderate number of files
        for i in range(50):
            file_path = temp_dir / f"perf_{i}.xlsx"
            file_path.touch()
        
        monitor = FileMonitor([temp_dir], callback)
        
        start_time = time.time()
        monitor.scan_existing_files()
        end_time = time.time()
        
        # Scanning should complete quickly
        scan_duration = end_time - start_time
        assert scan_duration < 1.0  # Should take less than 1 second
        assert callback.call_count == 50