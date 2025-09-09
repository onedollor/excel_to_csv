"""Archive manager for handling file archiving operations.

This module provides the ArchiveManager class that handles the complete
file archiving workflow including folder creation, conflict resolution,
and atomic file movement operations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from excel_to_csv.models.data_models import (
    ArchiveConfig,
    ArchiveError,
    ArchiveResult,
    RetryConfig,
)
from excel_to_csv.utils.logging_decorators import log_operation, log_method, operation_context
from excel_to_csv.utils.correlation import CorrelationContext


class ArchiveManager:
    """Manages file archiving operations for processed Excel files.
    
    The ArchiveManager handles:
    - Creating archive folders within monitored directories
    - Resolving filename conflicts with timestamps
    - Atomic file movement operations
    - Comprehensive error handling and logging
    - Path validation and permission checking
    
    Example:
        >>> archive_config = ArchiveConfig(enabled=True, archive_folder_name="archive")
        >>> manager = ArchiveManager()
        >>> result = manager.archive_file(Path("input/data.xlsx"), archive_config)
        >>> if result.success:
        ...     print(f"File archived to: {result.archive_path}")
    """
    
    @log_operation("initialize_archive_manager", log_args=False)
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize the archive manager.
        
        Args:
            retry_config: Retry configuration for failed operations
        """
        with operation_context(
            "archive_manager_initialization",
            logger=None  # Will use default logger
        ) as metrics:
            
            self.logger = logging.getLogger(__name__)
            self.retry_config = retry_config or RetryConfig(
                max_attempts=3,
                delay=1.0,  # Shorter delay for archiving operations
                backoff_factor=2.0,
                max_delay=10.0  # Max 10 seconds for archiving retries
            )
            
            # Log initialization details
            metrics.add_metadata("max_attempts", self.retry_config.max_attempts)
            metrics.add_metadata("initial_delay", self.retry_config.delay)
            metrics.add_metadata("backoff_factor", self.retry_config.backoff_factor)
            metrics.add_metadata("max_delay", self.retry_config.max_delay)
            
            self.logger.info(
                "Archive Manager initialized successfully",
                extra={
                    "structured": {
                        "operation": "archive_manager_init_success",
                        "retry_config": {
                            "max_attempts": self.retry_config.max_attempts,
                            "initial_delay": self.retry_config.delay,
                            "backoff_factor": self.retry_config.backoff_factor,
                            "max_delay": self.retry_config.max_delay
                        }
                    }
                }
            )
    
    @log_operation("archive_file_operation", log_args=False)
    def archive_file(
        self, 
        file_path: Path, 
        archive_config: ArchiveConfig
    ) -> ArchiveResult:
        """Archive a file to the designated archive folder.
        
        Args:
            file_path: Path to the file to archive
            archive_config: Archive configuration settings
            
        Returns:
            ArchiveResult containing operation details
        """
        with operation_context(
            "file_archiving",
            self.logger,
            file_path=str(file_path),
            archive_enabled=archive_config.enabled,
            archive_folder=archive_config.archive_folder_name,
            handle_conflicts=archive_config.handle_conflicts
        ) as metrics:
            start_time = time.time()
            
            # Quick return if archiving is disabled
            if not archive_config.enabled:
                operation_time = time.time() - start_time
                result = ArchiveResult(
                    success=True,
                    source_path=file_path,
                    operation_time=operation_time
                )
                
                metrics.add_metadata("archiving_disabled", True)
                metrics.add_metadata("operation_time", operation_time)
                
                self.logger.debug(
                    f"Archiving disabled for file: {file_path}",
                    extra={
                        "structured": {
                            "operation": "archiving_skipped",
                            "file_path": str(file_path),
                            "reason": "archiving_disabled",
                            "operation_time": operation_time
                        }
                    }
                )
                
                return result
            
            try:
                # Validate input
                if not file_path.exists():
                    raise ArchiveError(
                        f"Source file does not exist: {file_path}",
                        file_path=file_path,
                        error_type="filesystem"
                    )
                
                if not file_path.is_file():
                    raise ArchiveError(
                        f"Source path is not a file: {file_path}",
                        file_path=file_path,
                        error_type="filesystem"
                    )
                
                # Create archive folder
                archive_folder = self.create_archive_folder(
                    file_path.parent, 
                    archive_config.archive_folder_name
                )
                
                # Determine target path and resolve conflicts
                target_path = archive_folder / file_path.name
                if target_path.exists() and archive_config.handle_conflicts:
                    target_path = self.resolve_naming_conflicts(
                        target_path, 
                        archive_config.timestamp_format
                    )
                
                # Perform atomic file move with retry logic
                self._move_file_with_retry(file_path, target_path)
                
                # Log success
                operation_time = time.time() - start_time
                file_size = target_path.stat().st_size
                
                metrics.add_metadata("archive_success", True)
                metrics.add_metadata("operation_time", operation_time)
                metrics.add_metadata("source_path", str(file_path))
                metrics.add_metadata("archive_path", str(target_path))
                metrics.add_metadata("file_size", file_size)
                metrics.add_metadata("conflict_resolved", str(target_path) != str(archive_folder / file_path.name))
                
                self.logger.info(
                    f"Successfully archived file: {file_path.name} -> {target_path} ({file_size:,} bytes, took {operation_time:.3f}s)",
                    extra={
                        "structured": {
                            "operation": "file_archiving_success",
                            "source_path": str(file_path),
                            "archive_path": str(target_path),
                            "file_size_bytes": file_size,
                            "operation_time": operation_time,
                            "archive_folder": str(archive_folder),
                            "conflict_resolved": str(target_path) != str(archive_folder / file_path.name)
                        }
                    }
                )
                
                return ArchiveResult(
                    success=True,
                    source_path=file_path,
                    archive_path=target_path,
                    operation_time=operation_time
                )
            
            except ArchiveError as e:
                operation_time = time.time() - start_time
                
                metrics.add_metadata("archive_success", False)
                metrics.add_metadata("error_type", "ArchiveError")
                metrics.add_metadata("error_category", e.error_type if hasattr(e, 'error_type') else "unknown")
                metrics.add_metadata("operation_time", operation_time)
                
                self.logger.error(
                    f"Archive operation failed: {e}",
                    extra={
                        "structured": {
                            "operation": "file_archiving_failed",
                            "source_path": str(file_path),
                            "error_type": "ArchiveError",
                            "error_category": e.error_type if hasattr(e, 'error_type') else "unknown",
                            "error_message": str(e),
                            "operation_time": operation_time
                        }
                    }
                )
                
                return ArchiveResult(
                    success=False,
                    source_path=file_path,
                    error_message=str(e),
                    operation_time=operation_time
                )
            
            except Exception as e:
                operation_time = time.time() - start_time
                error_msg = f"Unexpected error during archiving: {e}"
                error_type = type(e).__name__
                
                metrics.add_metadata("archive_success", False)
                metrics.add_metadata("error_type", error_type)
                metrics.add_metadata("error_category", "unexpected")
                metrics.add_metadata("operation_time", operation_time)
                
                self.logger.error(
                    error_msg,
                    exc_info=True,
                    extra={
                        "structured": {
                            "operation": "file_archiving_unexpected_error",
                            "source_path": str(file_path),
                            "error_type": error_type,
                            "error_message": str(e),
                            "operation_time": operation_time
                        }
                    }
                )
                
                return ArchiveResult(
                    success=False,
                    source_path=file_path,
                    error_message=error_msg,
                    operation_time=operation_time
                )
    
    @log_operation("create_archive_folder", log_args=False)
    def create_archive_folder(self, base_folder: Path, archive_folder_name: str) -> Path:
        """Create archive folder within the base folder.
        
        Args:
            base_folder: Parent directory where archive folder should be created
            archive_folder_name: Name of the archive subfolder
            
        Returns:
            Path to the created (or existing) archive folder
            
        Raises:
            ArchiveError: If archive folder cannot be created
        """
        with operation_context(
            "archive_folder_creation",
            self.logger,
            base_folder=str(base_folder),
            archive_folder_name=archive_folder_name
        ) as metrics:
            try:
                archive_folder = base_folder / archive_folder_name
                
                archive_folder = base_folder / archive_folder_name
                folder_existed = archive_folder.exists()
                
                self.logger.debug(f"Checking archive folder: {archive_folder}")
                
                if archive_folder.exists():
                    if not archive_folder.is_dir():
                        error_msg = f"Archive path exists but is not a directory: {archive_folder}"
                        metrics.add_metadata("creation_result", "failed")
                        metrics.add_metadata("error_type", "not_directory")
                        
                        self.logger.error(
                            error_msg,
                            extra={
                                "structured": {
                                    "operation": "archive_folder_creation_failed",
                                    "archive_folder": str(archive_folder),
                                    "error_type": "not_directory",
                                    "folder_existed": folder_existed
                                }
                            }
                        )
                        
                        raise ArchiveError(
                            error_msg,
                            file_path=archive_folder,
                            error_type="filesystem"
                        )
                    else:
                        metrics.add_metadata("creation_result", "already_exists")
                        metrics.add_metadata("folder_existed", True)
                        
                        self.logger.debug(
                            f"Archive folder already exists: {archive_folder}",
                            extra={
                                "structured": {
                                    "operation": "archive_folder_exists",
                                    "archive_folder": str(archive_folder),
                                    "base_folder": str(base_folder)
                                }
                            }
                        )
                else:
                    # Create the directory with appropriate permissions
                    archive_folder.mkdir(parents=True, exist_ok=True)
                    
                    metrics.add_metadata("creation_result", "created")
                    metrics.add_metadata("folder_existed", False)
                    
                    self.logger.info(
                        f"Created archive folder: {archive_folder}",
                        extra={
                            "structured": {
                                "operation": "archive_folder_created",
                                "archive_folder": str(archive_folder),
                                "base_folder": str(base_folder),
                                "folder_name": archive_folder_name
                            }
                        }
                    )
                
                # Verify we can write to the directory
                if not archive_folder.is_dir():
                    error_msg = f"Archive folder is not accessible: {archive_folder}"
                    metrics.add_metadata("creation_result", "failed")
                    metrics.add_metadata("error_type", "not_accessible")
                    
                    self.logger.error(
                        error_msg,
                        extra={
                            "structured": {
                                "operation": "archive_folder_access_failed",
                                "archive_folder": str(archive_folder),
                                "error_type": "not_accessible"
                            }
                        }
                    )
                    
                    raise ArchiveError(
                        error_msg,
                        file_path=archive_folder,
                        error_type="permission"
                    )
                
                return archive_folder
                
            except PermissionError as e:
                raise ArchiveError(
                    f"Permission denied creating archive folder {base_folder / archive_folder_name}: {e}",
                    file_path=base_folder / archive_folder_name,
                    error_type="permission"
                ) from e
                
            except OSError as e:
                raise ArchiveError(
                    f"Failed to create archive folder {base_folder / archive_folder_name}: {e}",
                    file_path=base_folder / archive_folder_name,
                    error_type="filesystem"
                ) from e
    
    @log_operation("resolve_naming_conflicts", log_args=False)
    def resolve_naming_conflicts(
        self, 
        target_path: Path, 
        timestamp_format: str
    ) -> Path:
        """Resolve filename conflicts by appending timestamps.
        
        Args:
            target_path: Original target path that has a conflict
            timestamp_format: Format string for timestamp generation
            
        Returns:
            Path with timestamp suffix that doesn't conflict
        """
        with operation_context(
            "naming_conflict_resolution",
            self.logger,
            original_path=str(target_path),
            timestamp_format=timestamp_format
        ) as metrics:
            base_path = target_path.parent
            stem = target_path.stem
            suffix = target_path.suffix
            
            # Generate timestamp
            timestamp = datetime.now().strftime(timestamp_format)
            
            # Create new filename with timestamp
            new_name = f"{stem}_{timestamp}{suffix}"
            new_path = base_path / new_name
            
            # Handle multiple conflicts within the same timestamp
            counter = 1
            while new_path.exists():
                new_name = f"{stem}_{timestamp}_{counter:03d}{suffix}"
                new_path = base_path / new_name
                counter += 1
                
                # Prevent infinite loop
                if counter > 999:
                    raise ArchiveError(
                        f"Too many naming conflicts for file: {target_path}",
                        file_path=target_path,
                        error_type="filesystem"
                    )
            
            # Log conflict resolution
            metrics.add_metadata("original_name", target_path.name)
            metrics.add_metadata("resolved_name", new_path.name)
            metrics.add_metadata("timestamp", timestamp)
            metrics.add_metadata("counter_used", counter - 1)
            metrics.add_metadata("conflicts_resolved", counter)
            
            if counter > 1:
                self.logger.warning(
                    f"Resolved naming conflict: {target_path.name} -> {new_path.name} (conflict #{counter-1})",
                    extra={
                        "structured": {
                            "operation": "naming_conflict_resolved",
                            "original_name": target_path.name,
                            "resolved_name": new_path.name,
                            "timestamp": timestamp,
                            "conflicts_resolved": counter,
                            "counter_used": counter - 1
                        }
                    }
                )
            else:
                self.logger.warning(
                    f"Resolved naming conflict: {target_path.name} -> {new_path.name}",
                    extra={
                        "structured": {
                            "operation": "naming_conflict_resolved",
                            "original_name": target_path.name,
                            "resolved_name": new_path.name,
                            "timestamp": timestamp,
                            "conflicts_resolved": 1
                        }
                    }
                )
            
            return new_path
    
    def _move_file_atomic(self, source: Path, target: Path) -> None:
        """Perform atomic file move operation.
        
        Args:
            source: Source file path
            target: Target file path
            
        Raises:
            ArchiveError: If file move operation fails
        """
        try:
            # Use Path.replace() for atomic move operation on same filesystem
            source.replace(target)
            
        except PermissionError as e:
            raise ArchiveError(
                f"Permission denied moving file {source} to {target}: {e}",
                file_path=source,
                error_type="permission"
            ) from e
            
        except OSError as e:
            raise ArchiveError(
                f"Failed to move file {source} to {target}: {e}",
                file_path=source,
                error_type="filesystem"
            ) from e
    
    @log_operation("move_file_with_retry", log_args=False)
    def _move_file_with_retry(self, source: Path, target: Path) -> None:
        """Perform file move with retry logic for transient failures.
        
        Args:
            source: Source file path
            target: Target file path
            
        Raises:
            ArchiveError: If file move fails after all retries
        """
        with operation_context(
            "file_move_with_retry",
            self.logger,
            source_path=str(source),
            target_path=str(target),
            max_attempts=self.retry_config.max_attempts
        ) as metrics:
            last_exception = None
            
            for attempt in range(self.retry_config.max_attempts):
                try:
                    self._move_file_atomic(source, target)
                    
                    if attempt > 0:
                        metrics.add_metadata("success_attempt", attempt + 1)
                        self.logger.info(
                            f"File move succeeded on attempt {attempt + 1}: {source.name} -> {target.name}",
                            extra={
                                "structured": {
                                    "operation": "file_move_retry_success",
                                    "source_path": str(source),
                                    "target_path": str(target),
                                    "success_attempt": attempt + 1,
                                    "max_attempts": self.retry_config.max_attempts
                                }
                            }
                        )
                    else:
                        metrics.add_metadata("success_attempt", 1)
                        self.logger.debug(f"File move succeeded on first attempt: {source.name} -> {target.name}")
                    
                    return
                    
                except ArchiveError as e:
                    last_exception = e
                    
                    # Don't retry permission errors
                    if e.error_type == "permission":
                        raise
                    
                    # Log retry attempt
                    if attempt < self.retry_config.max_attempts - 1:
                        delay = self.retry_config.get_delay(attempt)
                        
                        self.logger.warning(
                            f"File move attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...",
                            extra={
                                "structured": {
                                    "operation": "file_move_retry_attempt",
                                    "source_path": str(source),
                                    "target_path": str(target),
                                    "attempt": attempt + 1,
                                    "max_attempts": self.retry_config.max_attempts,
                                    "error_type": type(e).__name__,
                                    "error_message": str(e),
                                    "retry_delay": delay
                                }
                            }
                        )
                        time.sleep(delay)
                    else:
                        metrics.add_metadata("final_failure", True)
                        metrics.add_metadata("attempts_made", self.retry_config.max_attempts)
                        
                        self.logger.error(
                            f"File move failed after {self.retry_config.max_attempts} attempts: {e}",
                            extra={
                                "structured": {
                                    "operation": "file_move_final_failure",
                                    "source_path": str(source),
                                    "target_path": str(target),
                                    "attempts_made": self.retry_config.max_attempts,
                                    "final_error_type": type(e).__name__,
                                    "final_error_message": str(e)
                                }
                            }
                        )
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
    
    def _is_transient_error(self, error: Exception) -> bool:
        """Check if an error is likely transient and worth retrying.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error might be transient
        """
        if isinstance(error, ArchiveError):
            # Don't retry permission errors
            if error.error_type == "permission":
                return False
            
            # Retry filesystem errors (might be temporary locks, network issues, etc.)
            if error.error_type == "filesystem":
                return True
        
        # Check for specific transient OS errors
        if isinstance(error, OSError):
            # Windows sharing violations, temporary locks, etc.
            if hasattr(error, 'winerror'):
                transient_winerrors = [32, 33]  # Sharing violation, lock violation
                return getattr(error, 'winerror') in transient_winerrors
            
            # UNIX temporary errors
            if hasattr(error, 'errno'):
                transient_errnos = [16, 26]  # EBUSY, ETXTBSY
                return error.errno in transient_errnos
        
        return False
    
    def validate_archive_path(self, path: Path) -> bool:
        """Validate that a path is suitable for archiving.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid for archiving
        """
        try:
            # Check if path exists and is accessible
            if not path.exists():
                return False
            
            # Check if it's a directory
            if not path.is_dir():
                return False
            
            # Try to create a temporary file to test write permissions
            test_file = path / ".archive_test"
            try:
                test_file.touch()
                test_file.unlink()
                return True
            except (PermissionError, OSError):
                return False
                
        except Exception:
            return False