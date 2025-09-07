"""Main application orchestrator for Excel-to-CSV converter.

This module provides the central coordination point that integrates all components:
- File monitoring for automatic processing
- Excel processing pipeline with confidence analysis
- CSV generation and output management
- Error handling with retry logic
- Service mode for continuous operation
- CLI mode for one-time processing
"""

import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, List
from queue import Queue, Empty
from dataclasses import dataclass

from excel_to_csv.config.config_manager import config_manager
from excel_to_csv.models.data_models import Config, WorksheetData
from excel_to_csv.processors.excel_processor import ExcelProcessor, ExcelProcessingError
from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
from excel_to_csv.generators.csv_generator import CSVGenerator, CSVGenerationError
from excel_to_csv.monitoring.file_monitor import FileMonitor, FileMonitorError
from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.utils.logger import setup_logging, get_processing_logger, shutdown_logging


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    files_processed: int = 0
    files_failed: int = 0
    worksheets_analyzed: int = 0
    worksheets_accepted: int = 0
    csv_files_generated: int = 0
    processing_errors: int = 0
    files_archived: int = 0
    archive_failures: int = 0


class ExcelToCSVConverter:
    """Main application class for Excel-to-CSV conversion.
    
    The ExcelToCSVConverter orchestrates all components to provide:
    - Continuous service mode monitoring directories
    - One-time CLI processing of specific files
    - Integrated pipeline with error handling and retry logic
    - Performance monitoring and statistics
    - Graceful shutdown handling
    
    Example:
        >>> # Service mode - continuous monitoring
        >>> converter = ExcelToCSVConverter("config.yaml")
        >>> converter.run_service()
        
        >>> # CLI mode - process single file  
        >>> converter.process_file("data.xlsx")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Excel-to-CSV converter.
        
        Args:
            config_path: Path to configuration file (None for defaults)
        """
        # Load configuration
        self.config = config_manager.load_config(config_path)
        
        # Set up logging
        setup_logging(self.config.logging)
        self.logger = get_processing_logger(__name__)
        self.logger.info(f"config_path: {config_path} initialized")
        
        # Initialize components
        self.excel_processor = ExcelProcessor(
            max_file_size_mb=self.config.max_file_size_mb
        )
        
        self.confidence_analyzer = ConfidenceAnalyzer(
            threshold=self.config.confidence_threshold
        )
        
        self.csv_generator = CSVGenerator()
        
        self.archive_manager = ArchiveManager(
            retry_config=self.config.retry_settings
        )
        
        # Processing state
        self.stats = ProcessingStats()
        self.failed_files: Dict[Path, int] = {}  # Track retry counts
        self.processing_queue: Queue[Path] = Queue()
        
        # Service mode components
        self.file_monitor: Optional[FileMonitor] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("Excel-to-CSV converter initialized")
        self.logger.info(f"Configuration: {len(self.config.monitored_folders)} folders, "
                        f"threshold: {self.config.confidence_threshold}, "
                        f"max concurrent: {self.config.max_concurrent}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            self.logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self.shutdown()
        
        # Handle common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM]:
            if hasattr(signal, sig.name):
                signal.signal(sig, signal_handler)
    
    def run_service(self) -> None:
        """Run in service mode - continuous monitoring and processing.
        
        This method starts the file monitor and processing threads,
        then runs until shutdown is requested.
        """
        if self.is_running:
            self.logger.warning("Service is already running")
            return
        
        try:
            self.logger.info("Starting Excel-to-CSV converter service...")
            self.is_running = True
            
            # Start thread pool for processing
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent,
                thread_name_prefix="ExcelProcessor"
            )
            
            # Start processing queue handler
            processing_thread = threading.Thread(
                target=self._process_queue_worker,
                name="QueueProcessor",
                daemon=True
            )
            processing_thread.start()
            
            # Start file monitor
            self.file_monitor = FileMonitor(
                folders=self.config.monitored_folders,
                file_patterns=self.config.file_patterns,
                callback=self._on_file_detected,
                process_existing=True
            )
            
            self.file_monitor.start_monitoring()
            
            # Start statistics reporting
            stats_thread = threading.Thread(
                target=self._stats_reporter,
                name="StatsReporter", 
                daemon=True
            )
            stats_thread.start()
            
            self.logger.info("Service started successfully")
            
            # Main service loop
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    self.shutdown_event.wait(1.0)  # Check every second
                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt received")
                    break
            
        except Exception as e:
            self.logger.error(f"Service error: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_service()
    
    def _on_file_detected(self, file_path: Path) -> None:
        """Callback for when file monitor detects a new file.
        
        Args:
            file_path: Path to detected Excel file
        """
        self.logger.info(f"File detected: {file_path}")
        self.processing_queue.put(file_path)
    
    def _process_queue_worker(self) -> None:
        """Worker thread for processing queued files."""
        while self.is_running:
            try:
                # Get file from queue with timeout
                try:
                    file_path = self.processing_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Submit processing job to thread pool
                if self.executor:
                    future = self.executor.submit(self._process_file_with_retry, file_path)
                    # Don't wait for completion - let it run asynchronously
                    
                self.processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Queue worker error: {e}", exc_info=True)
    
    def _stats_reporter(self) -> None:
        """Background thread for periodic statistics reporting."""
        last_report_time = time.time()
        report_interval = 300  # Report every 5 minutes
        
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    self._log_statistics()
                    last_report_time = current_time
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Stats reporter error: {e}", exc_info=True)
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single Excel file (CLI mode).
        
        Args:
            file_path: Path to Excel file to process
            
        Returns:
            True if processing was successful
        """
        self.logger.info(f"Processing single file: {file_path}")
        return self._process_file_with_retry(file_path)
    
    def _process_file_with_retry(self, file_path: Path) -> bool:
        """Process file with retry logic.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            True if processing was successful
        """
        retry_count = self.failed_files.get(file_path, 0)
        max_retries = self.config.retry_settings.max_attempts
        
        if retry_count >= max_retries:
            self.logger.error(f"Max retries exceeded for file: {file_path}")
            return False
        
        try:
            success = self._process_file_pipeline(file_path)
            
            if success:
                # Remove from failed files on success
                self.failed_files.pop(file_path, None)
                self.stats.files_processed += 1
            else:
                # Increment retry count
                self.failed_files[file_path] = retry_count + 1
                self.stats.files_failed += 1
                
                # Schedule retry if not at max
                if retry_count + 1 < max_retries:
                    retry_delay = self.config.retry_settings.get_delay(retry_count)
                    self.logger.info(f"Scheduling retry for {file_path} in {retry_delay}s")
                    
                    # Schedule retry (in service mode)
                    if self.is_running:
                        threading.Timer(retry_delay, lambda: self.processing_queue.put(file_path)).start()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
            self.failed_files[file_path] = retry_count + 1
            self.stats.processing_errors += 1
            return False
    
    def _process_file_pipeline(self, file_path: Path) -> bool:
        """Execute the full processing pipeline for a file.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            True if processing was successful
        """
        try:
            # Step 1: Process Excel file and extract worksheets
            worksheets = self.excel_processor.process_file(file_path)
            
            if not worksheets:
                self.logger.info(f"No worksheets found in file: {file_path}")
                return True  # Not an error, just empty file
            
            # Step 2: Analyze each worksheet for confidence
            qualified_worksheets = []
            
            for worksheet in worksheets:
                self.stats.worksheets_analyzed += 1
                
                confidence_score = self.confidence_analyzer.analyze_worksheet(worksheet)
                worksheet.confidence_score = confidence_score
                
                if confidence_score.is_confident:
                    qualified_worksheets.append(worksheet)
                    self.stats.worksheets_accepted += 1
                    
                    self.logger.info(
                        f"Worksheet '{worksheet.worksheet_name}' accepted "
                        f"(confidence: {confidence_score.overall_score:.3f})"
                    )
                else:
                    self.logger.info(
                        f"Worksheet '{worksheet.worksheet_name}' rejected "
                        f"(confidence: {confidence_score.overall_score:.3f})"
                    )
            
            # Step 3: Generate CSV files for qualified worksheets
            csv_files_created = 0
            
            for worksheet in qualified_worksheets:
                try:
                    output_path = self.csv_generator.generate_csv(
                        worksheet, 
                        self.config.output_config
                    )
                    
                    csv_files_created += 1
                    self.stats.csv_files_generated += 1
                    
                    self.logger.info(f"Generated CSV: {output_path}")
                    
                except CSVGenerationError as e:
                    self.logger.error(f"Failed to generate CSV for worksheet "
                                    f"'{worksheet.worksheet_name}': {e}")
            
            # Step 4: Archive source file if CSV generation was successful and archiving is enabled
            archive_success = True
            if (csv_files_created > 0 or len(qualified_worksheets) == 0) and self.config.archive_config.enabled:
                try:
                    archive_result = self.archive_manager.archive_file(
                        file_path, 
                        self.config.archive_config
                    )
                    
                    if archive_result.success:
                        self.stats.files_archived += 1
                        self.logger.info(
                            f"Successfully archived: {file_path} -> {archive_result.archive_path} "
                            f"(took {archive_result.operation_time:.3f}s)"
                        )
                        # Update worksheet data with archive information
                        for worksheet in worksheets:
                            worksheet.archive_result = archive_result
                            worksheet.archived_at = archive_result.timestamp_used
                    else:
                        archive_success = False
                        self.stats.archive_failures += 1
                        self.logger.warning(
                            f"Archiving failed for {file_path}: {archive_result.error_message} "
                            f"(attempted for {archive_result.operation_time:.3f}s)"
                        )
                        
                except Exception as e:
                    archive_success = False
                    self.stats.archive_failures += 1
                    self.logger.error(f"Unexpected error during archiving of {file_path}: {e}", exc_info=True)
            
            # Log completion summary
            archive_status = ""
            if self.config.archive_config.enabled:
                archive_status = " (archived)" if archive_success else " (archive failed)"
            
            self.logger.info(
                f"Completed processing {file_path}: "
                f"{csv_files_created}/{len(worksheets)} worksheets converted{archive_status}"
            )
            
            return csv_files_created > 0 or len(qualified_worksheets) == 0
            
        except ExcelProcessingError as e:
            self.logger.error(f"Excel processing failed for {file_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Pipeline error for {file_path}: {e}", exc_info=True)
            return False
    
    def _log_statistics(self) -> None:
        """Log current processing statistics."""
        self.logger.info("=== Processing Statistics ===")
        self.logger.info(f"Files processed: {self.stats.files_processed}")
        self.logger.info(f"Files failed: {self.stats.files_failed}")
        self.logger.info(f"Worksheets analyzed: {self.stats.worksheets_analyzed}")
        self.logger.info(f"Worksheets accepted: {self.stats.worksheets_accepted}")
        self.logger.info(f"CSV files generated: {self.stats.csv_files_generated}")
        self.logger.info(f"Processing errors: {self.stats.processing_errors}")
        
        # Archiving statistics (only show if archiving is enabled)
        if self.config.archive_config.enabled:
            self.logger.info(f"Files archived: {self.stats.files_archived}")
            self.logger.info(f"Archive failures: {self.stats.archive_failures}")
            
            total_archive_attempts = self.stats.files_archived + self.stats.archive_failures
            if total_archive_attempts > 0:
                archive_success_rate = (self.stats.files_archived / total_archive_attempts) * 100
                self.logger.info(f"Archive success rate: {archive_success_rate:.1f}%")
        
        if self.stats.worksheets_analyzed > 0:
            acceptance_rate = (self.stats.worksheets_accepted / self.stats.worksheets_analyzed) * 100
            self.logger.info(f"Acceptance rate: {acceptance_rate:.1f}%")
        
        if self.file_monitor:
            monitor_stats = self.file_monitor.get_statistics()
            self.logger.info(f"Monitored folders: {monitor_stats['folders_count']}")
            self.logger.info(f"Files in queue: {monitor_stats.get('pending_files', 0)}")
        
        self.logger.info("=============================")
    
    def get_statistics(self) -> Dict:
        """Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'files_processed': self.stats.files_processed,
            'files_failed': self.stats.files_failed,
            'worksheets_analyzed': self.stats.worksheets_analyzed,
            'worksheets_accepted': self.stats.worksheets_accepted,
            'csv_files_generated': self.stats.csv_files_generated,
            'processing_errors': self.stats.processing_errors,
            'files_archived': self.stats.files_archived,
            'archive_failures': self.stats.archive_failures,
            'failed_files': dict(self.failed_files),
            'is_running': self.is_running,
        }
        
        if self.stats.worksheets_analyzed > 0:
            stats['acceptance_rate'] = (self.stats.worksheets_accepted / self.stats.worksheets_analyzed) * 100
        
        # Add archive success rate if archiving is enabled
        total_archive_attempts = self.stats.files_archived + self.stats.archive_failures
        if self.config.archive_config.enabled and total_archive_attempts > 0:
            stats['archive_success_rate'] = (self.stats.files_archived / total_archive_attempts) * 100
        
        if self.file_monitor:
            stats['monitor'] = self.file_monitor.get_statistics()
        
        return stats
    
    def shutdown(self) -> None:
        """Initiate graceful shutdown of the service."""
        if not self.is_running:
            self.logger.debug("Service is not running")
            return
        
        self.logger.info("Initiating graceful shutdown...")
        self.is_running = False
        self.shutdown_event.set()
    
    def _cleanup_service(self) -> None:
        """Clean up service resources."""
        self.logger.info("Cleaning up service resources...")
        
        # Stop file monitor
        if self.file_monitor:
            try:
                self.file_monitor.stop_monitoring()
            except Exception as e:
                self.logger.warning(f"Error stopping file monitor: {e}")
        
        # Wait for processing queue to finish
        if hasattr(self.processing_queue, 'join'):
            try:
                # Wait a reasonable time for queue to finish
                self.logger.info("Waiting for processing queue to finish...")
                # Process remaining items quickly
                remaining_items = []
                while True:
                    try:
                        item = self.processing_queue.get_nowait()
                        remaining_items.append(item)
                        self.processing_queue.task_done()
                    except Empty:
                        break
                
                if remaining_items:
                    self.logger.info(f"Skipping {len(remaining_items)} remaining items in queue")
                        
            except Exception as e:
                self.logger.warning(f"Error cleaning up processing queue: {e}")
        
        # Shutdown thread pool
        if self.executor:
            try:
                self.logger.info("Shutting down thread pool...")
                self.executor.shutdown(wait=True, timeout=30.0)
            except Exception as e:
                self.logger.warning(f"Error shutting down thread pool: {e}")
        
        # Final statistics
        self._log_statistics()
        
        # Shutdown logging
        shutdown_logging()
        
        self.logger.info("Service shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_running:
            self.shutdown()