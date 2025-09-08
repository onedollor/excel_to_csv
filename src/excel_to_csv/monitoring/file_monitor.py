"""File system monitoring for Excel-to-CSV converter.

This module provides real-time file system monitoring capabilities including:
- Cross-platform directory watching using watchdog
- Excel file pattern filtering
- Multiple folder monitoring simultaneously
- Initial folder scanning for existing files
- Event debouncing and file stability checking
"""

import fnmatch
import time
import threading
from pathlib import Path
from typing import Callable, List, Optional, Set, Union
from queue import Queue, Empty

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent, FileModifiedEvent, FileCreatedEvent

from excel_to_csv.utils.logger import get_processing_logger


class FileMonitorError(Exception):
    """Raised when file monitoring fails."""
    pass


class ExcelFileHandler(FileSystemEventHandler):
    """File system event handler for Excel files.
    
    Handles file system events and filters for Excel files matching
    specified patterns. Provides debouncing to avoid processing
    files that are still being written.
    """
    
    def __init__(
        self, 
        file_patterns: List[str], 
        callback: Callable[[Path], None],
        debounce_seconds: float = 2.0
    ):
        """Initialize Excel file handler.
        
        Args:
            file_patterns: List of file patterns to match (e.g., ['*.xlsx', '*.xls'])
            callback: Callback function to call when Excel file is detected
            debounce_seconds: Seconds to wait after last modification before processing
        """
        super().__init__()
        self.file_patterns = file_patterns
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.logger = get_processing_logger(__name__)
        
        # Track pending files for debouncing
        self._pending_files: dict[Path, float] = {}
        self._pending_lock = threading.Lock()
        
        # Start debounce processing thread
        self._debounce_thread = threading.Thread(target=self._process_pending_files, daemon=True)
        self._debounce_thread.start()
        
        # Track processed files with their modification times to avoid duplicates
        # Dict mapping file path to last processed modification time
        self._processed_files: dict[Path, float] = {}
        self._processed_lock = threading.Lock()
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self._matches_patterns(file_path):
            self.logger.debug(f"Excel file created: {file_path}")
            self._add_pending_file(file_path)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if self._matches_patterns(file_path):
            self.logger.debug(f"Excel file modified: {file_path}")
            self._add_pending_file(file_path)
    
    def _matches_patterns(self, file_path: Path) -> bool:
        """Check if file matches any of the specified patterns.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file matches patterns
        """
        filename = file_path.name
        return any(fnmatch.fnmatch(filename.lower(), pattern.lower()) 
                  for pattern in self.file_patterns)
    
    def _add_pending_file(self, file_path: Path) -> None:
        """Add file to pending processing queue.
        
        Args:
            file_path: Path to add to pending queue
        """
        with self._pending_lock:
            self._pending_files[file_path] = time.time()
    
    def _process_pending_files(self) -> None:
        """Process pending files after debounce period."""
        while True:
            try:
                current_time = time.time()
                files_to_process = []
                
                with self._pending_lock:
                    # Find files that have been stable for debounce period
                    for file_path, last_modified in list(self._pending_files.items()):
                        if current_time - last_modified >= self.debounce_seconds:
                            files_to_process.append(file_path)
                            del self._pending_files[file_path]
                
                # Process stable files
                for file_path in files_to_process:
                    self._process_file(file_path)
                
                # Sleep before next check
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in debounce processing: {e}", exc_info=True)
                time.sleep(5.0)  # Wait longer on error
    
    def _process_file(self, file_path: Path) -> None:
        """Process a stable file.
        
        Args:
            file_path: Path to process
        """
        try:
            # Check if file still exists and is readable
            if not file_path.exists():
                self.logger.debug(f"File no longer exists: {file_path}")
                return
            
            if not file_path.is_file():
                self.logger.debug(f"Path is not a file: {file_path}")
                return
            
            # Check file modification time to avoid processing unchanged files
            try:
                current_mtime = file_path.stat().st_mtime
            except OSError:
                self.logger.warning(f"Cannot get file stats: {file_path}")
                return
            
            # Avoid processing same file multiple times (unless modified)
            with self._processed_lock:
                if file_path in self._processed_files:
                    last_processed_mtime = self._processed_files[file_path]
                    if current_mtime <= last_processed_mtime:
                        self.logger.debug(f"File already processed and unchanged: {file_path} (mtime: {current_mtime})")
                        return
                    else:
                        self.logger.info(f"File modified since last processing: {file_path} (old: {last_processed_mtime}, new: {current_mtime})")
                
                self._processed_files[file_path] = current_mtime
            
            # Check if file is accessible (not locked)
            if not self._is_file_accessible(file_path):
                self.logger.warning(f"File is locked or inaccessible: {file_path}")
                # Remove from processed dict so we can retry later
                with self._processed_lock:
                    self._processed_files.pop(file_path, None)
                return
            
            self.logger.info(f"Processing Excel file: {file_path}")
            self.callback(file_path)
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            # Remove from processed dict so we can retry later
            with self._processed_lock:
                self._processed_files.pop(file_path, None)
    
    def _is_file_accessible(self, file_path: Path) -> bool:
        """Check if file is accessible for reading.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file is accessible
        """
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
            return True
        except (OSError, IOError, PermissionError):
            return False
    
    def process_existing_file(self, file_path: Path) -> None:
        """Process an existing file (for initial scan).
        
        Args:
            file_path: Path to existing file
        """
        if self._matches_patterns(file_path):
            self.logger.debug(f"Processing existing file: {file_path}")
            self._process_file(file_path)


class FileMonitor:
    """Monitors directories for Excel files and triggers processing.
    
    The FileMonitor provides:
    - Real-time monitoring of multiple directories
    - Pattern-based file filtering
    - Debounced file processing
    - Initial scanning of existing files
    - Graceful start/stop capabilities
    
    Example:
        >>> monitor = FileMonitor(
        ...     folders=[Path("./data")], 
        ...     patterns=["*.xlsx"],
        ...     callback=process_excel_file
        ... )
        >>> monitor.start_monitoring()
        >>> # ... monitoring runs in background ...
        >>> monitor.stop_monitoring()
    """
    
    def __init__(
        self, 
        folders: List[Path], 
        file_patterns: List[str],
        callback: Callable[[Path], None],
        debounce_seconds: float = 2.0,
        process_existing: bool = True
    ):
        """Initialize file monitor.
        
        Args:
            folders: List of directories to monitor
            file_patterns: List of file patterns to match
            callback: Callback function for detected files
            debounce_seconds: Debounce period for file stability
            process_existing: Whether to process existing files on startup
        """
        self.folders = folders
        self.file_patterns = file_patterns
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.process_existing = process_existing
        self.logger = get_processing_logger(__name__)
        
        # Monitoring state
        self._observer: Optional[Observer] = None
        self._handlers: List[ExcelFileHandler] = []
        self._is_monitoring = False
        self._monitoring_lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """Start monitoring configured directories.
        
        Raises:
            FileMonitorError: If monitoring cannot be started
        """
        with self._monitoring_lock:
            if self._is_monitoring:
                self.logger.warning("File monitoring is already running")
                return
            
            try:
                self._validate_folders()
                
                # Create observer
                self._observer = Observer()
                self._handlers = []
                
                # Set up monitoring for each folder
                for folder in self.folders:
                    handler = ExcelFileHandler(
                        self.file_patterns, 
                        self.callback, 
                        self.debounce_seconds
                    )
                    
                    self._observer.schedule(handler, str(folder), recursive=False)
                    self._handlers.append(handler)
                    
                    self.logger.info(f"Monitoring folder: {folder}")
                
                # Start observer
                self._observer.start()
                self._is_monitoring = True
                
                self.logger.info(
                    f"Started monitoring {len(self.folders)} folders "
                    f"for patterns: {self.file_patterns}"
                )
                
                # Process existing files if requested
                if self.process_existing:
                    self._scan_existing_files()
                
            except Exception as e:
                self._cleanup_monitoring()
                raise FileMonitorError(f"Failed to start monitoring: {e}") from e
    
    def stop_monitoring(self) -> None:
        """Stop monitoring directories."""
        with self._monitoring_lock:
            if not self._is_monitoring:
                self.logger.debug("File monitoring is not running")
                return
            
            self._cleanup_monitoring()
            self.logger.info("Stopped file monitoring")
    
    def _cleanup_monitoring(self) -> None:
        """Clean up monitoring resources."""
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5.0)
            except Exception as e:
                self.logger.warning(f"Error stopping observer: {e}")
            finally:
                self._observer = None
        
        self._handlers.clear()
        self._is_monitoring = False
    
    def _validate_folders(self) -> None:
        """Validate that all folders exist and are accessible.
        
        Raises:
            FileMonitorError: If validation fails
        """
        for folder in self.folders:
            if not folder.exists():
                raise FileMonitorError(f"Folder does not exist: {folder}")
            
            if not folder.is_dir():
                raise FileMonitorError(f"Path is not a directory: {folder}")
            
            # Test read access
            try:
                list(folder.iterdir())
            except (OSError, PermissionError) as e:
                raise FileMonitorError(f"Cannot access folder {folder}: {e}")
    
    def _scan_existing_files(self) -> None:
        """Scan folders for existing Excel files."""
        self.logger.info("Scanning for existing Excel files...")
        
        total_files = 0
        for folder in self.folders:
            try:
                folder_files = 0
                for file_path in folder.iterdir():
                    if file_path.is_file():
                        # Process through first handler (they all have same patterns/callback)
                        if self._handlers:
                            self._handlers[0].process_existing_file(file_path)
                            folder_files += 1
                
                self.logger.info(f"Found {folder_files} files in {folder}")
                total_files += folder_files
                
            except Exception as e:
                self.logger.error(f"Error scanning folder {folder}: {e}")
        
        self.logger.info(f"Completed initial scan: {total_files} total files processed")
    
    def add_folder(self, folder: Path) -> None:
        """Add a new folder to monitor.
        
        Args:
            folder: Path to folder to add
            
        Raises:
            FileMonitorError: If folder cannot be added
        """
        with self._monitoring_lock:
            if folder in self.folders:
                self.logger.warning(f"Folder already being monitored: {folder}")
                return
            
            # Validate folder
            if not folder.exists() or not folder.is_dir():
                raise FileMonitorError(f"Invalid folder: {folder}")
            
            self.folders.append(folder)
            
            # If currently monitoring, add to observer
            if self._is_monitoring and self._observer:
                try:
                    handler = ExcelFileHandler(
                        self.file_patterns, 
                        self.callback, 
                        self.debounce_seconds
                    )
                    
                    self._observer.schedule(handler, str(folder), recursive=False)
                    self._handlers.append(handler)
                    
                    self.logger.info(f"Added folder to monitoring: {folder}")
                    
                    # Scan new folder for existing files
                    if self.process_existing:
                        for file_path in folder.iterdir():
                            if file_path.is_file():
                                handler.process_existing_file(file_path)
                
                except Exception as e:
                    # Remove from folder list if monitoring setup failed
                    self.folders.remove(folder)
                    raise FileMonitorError(f"Failed to add folder to monitoring: {e}")
    
    def remove_folder(self, folder: Path) -> bool:
        """Remove a folder from monitoring.
        
        Args:
            folder: Path to folder to remove
            
        Returns:
            True if folder was removed, False if not found
        """
        with self._monitoring_lock:
            try:
                self.folders.remove(folder)
                
                # If currently monitoring, restart to remove folder
                # (watchdog doesn't support removing individual watches easily)
                if self._is_monitoring:
                    self.logger.info(f"Restarting monitoring to remove folder: {folder}")
                    self.stop_monitoring()
                    self.start_monitoring()
                
                return True
                
            except ValueError:
                return False
    
    def get_monitored_folders(self) -> List[Path]:
        """Get list of currently monitored folders.
        
        Returns:
            List of monitored folder paths
        """
        return self.folders.copy()
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active.
        
        Returns:
            True if monitoring is active
        """
        return self._is_monitoring
    
    def get_statistics(self) -> dict:
        """Get monitoring statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        stats = {
            'is_monitoring': self._is_monitoring,
            'folders_count': len(self.folders),
            'folders': [str(folder) for folder in self.folders],
            'file_patterns': self.file_patterns,
            'debounce_seconds': self.debounce_seconds,
        }
        
        # Add handler-specific statistics
        if self._handlers:
            total_processed = sum(
                len(handler._processed_files) for handler in self._handlers
            )
            total_pending = sum(
                len(handler._pending_files) for handler in self._handlers
            )
            
            stats.update({
                'processed_files': total_processed,
                'pending_files': total_pending,
            })
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()