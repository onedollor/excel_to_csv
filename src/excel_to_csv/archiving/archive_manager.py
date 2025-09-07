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
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize the archive manager.
        
        Args:
            retry_config: Retry configuration for failed operations
        """
        self.logger = logging.getLogger(__name__)
        self.retry_config = retry_config or RetryConfig(
            max_attempts=3,
            delay=1.0,  # Shorter delay for archiving operations
            backoff_factor=2.0,
            max_delay=10.0  # Max 10 seconds for archiving retries
        )
    
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
        start_time = time.time()
        
        # Quick return if archiving is disabled
        if not archive_config.enabled:
            return ArchiveResult(
                success=True,
                source_path=file_path,
                operation_time=time.time() - start_time
            )
        
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
            self.logger.info(
                f"Successfully archived file: {file_path} -> {target_path} "
                f"(took {operation_time:.3f}s)"
            )
            
            return ArchiveResult(
                success=True,
                source_path=file_path,
                archive_path=target_path,
                operation_time=operation_time
            )
            
        except ArchiveError as e:
            operation_time = time.time() - start_time
            self.logger.error(f"Archive operation failed: {e}")
            
            return ArchiveResult(
                success=False,
                source_path=file_path,
                error_message=str(e),
                operation_time=operation_time
            )
            
        except Exception as e:
            operation_time = time.time() - start_time
            error_msg = f"Unexpected error during archiving: {e}"
            self.logger.error(error_msg, exc_info=True)
            
            return ArchiveResult(
                success=False,
                source_path=file_path,
                error_message=error_msg,
                operation_time=operation_time
            )
    
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
        try:
            archive_folder = base_folder / archive_folder_name
            
            if archive_folder.exists():
                if not archive_folder.is_dir():
                    raise ArchiveError(
                        f"Archive path exists but is not a directory: {archive_folder}",
                        file_path=archive_folder,
                        error_type="filesystem"
                    )
            else:
                # Create the directory with appropriate permissions
                archive_folder.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created archive folder: {archive_folder}")
            
            # Verify we can write to the directory
            if not archive_folder.is_dir():
                raise ArchiveError(
                    f"Archive folder is not accessible: {archive_folder}",
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
        
        if counter > 1:
            self.logger.warning(
                f"Resolved naming conflict: {target_path.name} -> {new_path.name} "
                f"(conflict #{counter-1})"
            )
        else:
            self.logger.warning(
                f"Resolved naming conflict: {target_path.name} -> {new_path.name}"
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
    
    def _move_file_with_retry(self, source: Path, target: Path) -> None:
        """Perform file move with retry logic for transient failures.
        
        Args:
            source: Source file path
            target: Target file path
            
        Raises:
            ArchiveError: If file move fails after all retries
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                self._move_file_atomic(source, target)
                
                if attempt > 0:
                    self.logger.info(
                        f"File move succeeded on attempt {attempt + 1}: {source} -> {target}"
                    )
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
                        f"File move attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"File move failed after {self.retry_config.max_attempts} attempts: {e}"
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