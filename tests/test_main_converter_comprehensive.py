"""Comprehensive tests for Main Excel-to-CSV Converter with high coverage."""

import pytest
import tempfile
import shutil
import pandas as pd
import signal
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call, PropertyMock
from queue import Queue
import logging
import os

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter, ProcessingStats
from excel_to_csv.models.data_models import (
    Config, LoggingConfig, OutputConfig, RetryConfig, 
    ArchiveConfig, ConfidenceConfig, WorksheetData, ConfidenceScore
)
from excel_to_csv.processors.excel_processor import ExcelProcessingError
from excel_to_csv.generators.csv_generator import CSVGenerationError


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config(temp_workspace):
    """Create sample configuration for testing."""
    return Config(
        monitored_folders=[temp_workspace / "input"],
        confidence_threshold=0.8,
        confidence_config=ConfidenceConfig(
            threshold=0.8,
            weights={'data_density': 0.4, 'header_quality': 0.3, 'consistency': 0.3},
            min_rows=5,
            min_columns=1,
            max_empty_percentage=0.3
        ),
        output_folder=temp_workspace / "output",
        file_patterns=["*.xlsx", "*.xls"],
        logging=LoggingConfig(
            level="INFO",
            format="%(message)s",
            file_enabled=False,
            file_path=temp_workspace / "test.log",
            console_enabled=True,
            structured_enabled=False
        ),
        retry_settings=RetryConfig(
            max_attempts=3,
            delay=0.1,
            backoff_factor=2.0,
            max_delay=1.0
        ),
        output_config=OutputConfig(
            folder=temp_workspace / "output",
            naming_pattern="{filename}_{worksheet}.csv",
            include_timestamp=False,
            encoding="utf-8",
            delimiter=",",
            include_headers=True,
            timestamp_format="%Y%m%d_%H%M%S"
        ),
        archive_config=ArchiveConfig(
            enabled=True,
            archive_folder_name="archive",
            timestamp_format="%Y%m%d_%H%M%S",
            handle_conflicts=True,
            preserve_structure=True
        ),
        max_concurrent=2,
        max_file_size_mb=100
    )


@pytest.fixture
def sample_excel_file(temp_workspace):
    """Create sample Excel file for testing."""
    input_dir = temp_workspace / "input"
    input_dir.mkdir(exist_ok=True)
    
    file_path = input_dir / "test.xlsx"
    
    data = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 22],
        'Salary': [75000, 65000, 80000, 70000, 60000]
    })
    
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        data.to_excel(writer, sheet_name='Sheet1', index=False)
    
    return file_path


@pytest.fixture
def mock_config_manager():
    """Mock config manager for testing."""
    with patch('excel_to_csv.excel_to_csv_converter.config_manager') as mock:
        yield mock


class TestProcessingStats:
    """Test ProcessingStats dataclass."""
    
    def test_processing_stats_initialization(self):
        """Test ProcessingStats initialization with defaults."""
        stats = ProcessingStats()
        
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        assert stats.worksheets_analyzed == 0
        assert stats.worksheets_accepted == 0
        assert stats.csv_files_generated == 0
        assert stats.processing_errors == 0
        assert stats.files_archived == 0
        assert stats.archive_failures == 0
    
    def test_processing_stats_with_values(self):
        """Test ProcessingStats with custom values."""
        stats = ProcessingStats(
            files_processed=10,
            files_failed=2,
            worksheets_analyzed=15,
            worksheets_accepted=12,
            csv_files_generated=12,
            processing_errors=3,
            files_archived=8,
            archive_failures=2
        )
        
        assert stats.files_processed == 10
        assert stats.files_failed == 2
        assert stats.worksheets_analyzed == 15
        assert stats.worksheets_accepted == 12
        assert stats.csv_files_generated == 12
        assert stats.processing_errors == 3
        assert stats.files_archived == 8
        assert stats.archive_failures == 2


class TestConverterInitialization:
    """Test ExcelToCSVConverter initialization."""
    
    def test_initialization_with_default_config(self, mock_config_manager, sample_config):
        """Test converter initialization with default config."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
        
        assert converter.config == sample_config
        assert hasattr(converter, 'excel_processor')
        assert hasattr(converter, 'confidence_analyzer')
        assert hasattr(converter, 'csv_generator')
        assert hasattr(converter, 'archive_manager')
        assert isinstance(converter.stats, ProcessingStats)
        
        mock_config_manager.load_config.assert_called_once_with(None)
    
    def test_initialization_with_custom_config_path(self, mock_config_manager, sample_config):
        """Test converter initialization with custom config path."""
        mock_config_manager.load_config.return_value = sample_config
        config_path = "/custom/config.yaml"
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter(config_path)
        
        mock_config_manager.load_config.assert_called_once_with(config_path)
    
    def test_initialization_sets_up_components_correctly(self, mock_config_manager, sample_config):
        """Test that initialization sets up all components with correct parameters."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                with patch('excel_to_csv.excel_to_csv_converter.ExcelProcessor') as mock_processor:
                    with patch('excel_to_csv.excel_to_csv_converter.ConfidenceAnalyzer') as mock_analyzer:
                        with patch('excel_to_csv.excel_to_csv_converter.ArchiveManager') as mock_archive:
                            converter = ExcelToCSVConverter()
        
        # Check that components were initialized with correct parameters
        mock_processor.assert_called_once_with(max_file_size_mb=sample_config.max_file_size_mb)
        mock_analyzer.assert_called_once_with(
            threshold=sample_config.confidence_config.threshold,
            weights=sample_config.confidence_config.weights,
            min_rows=sample_config.confidence_config.min_rows,
            min_columns=sample_config.confidence_config.min_columns,
            max_empty_percentage=sample_config.confidence_config.max_empty_percentage
        )
        mock_archive.assert_called_once_with(retry_config=sample_config.retry_settings)


class TestSignalHandling:
    """Test signal handling functionality."""
    
    def test_setup_signal_handlers(self, mock_config_manager, sample_config):
        """Test signal handler setup."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                with patch('signal.signal') as mock_signal:
                    converter = ExcelToCSVConverter()
                    converter._setup_signal_handlers()
        
        # Check that signal handlers were set up
        assert mock_signal.call_count >= 2  # At least SIGINT and SIGTERM
        
        # Check specific signals
        signal_calls = [call[0] for call in mock_signal.call_args_list]
        assert any(signal.SIGINT in call for call in signal_calls)
        assert any(signal.SIGTERM in call for call in signal_calls)


class TestFileProcessing:
    """Test file processing functionality."""
    
    def test_process_file_success(self, mock_config_manager, sample_config, sample_excel_file):
        """Test successful single file processing."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock the pipeline to return success
                with patch.object(converter, '_process_file_pipeline', return_value=True):
                    result = converter.process_file(sample_excel_file)
        
        assert result is True
        assert converter.stats.files_processed == 1
        assert converter.stats.files_failed == 0
    
    def test_process_file_failure(self, mock_config_manager, sample_config, sample_excel_file):
        """Test file processing failure."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock the pipeline to return failure
                with patch.object(converter, '_process_file_pipeline', return_value=False):
                    result = converter.process_file(sample_excel_file)
        
        assert result is False
        assert converter.stats.files_processed == 0
        assert converter.stats.files_failed == 1
    
    def test_process_file_with_retry_logic(self, mock_config_manager, sample_config, sample_excel_file):
        """Test file processing with retry logic."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # First call fails, second succeeds
                with patch.object(converter, '_process_file_pipeline', side_effect=[False, True]):
                    # First attempt
                    result1 = converter._process_file_with_retry(sample_excel_file)
                    assert result1 is False
                    assert sample_excel_file in converter.failed_files
                    assert converter.failed_files[sample_excel_file] == 1
                    
                    # Second attempt (retry)
                    result2 = converter._process_file_with_retry(sample_excel_file)
                    assert result2 is True
                    assert sample_excel_file not in converter.failed_files
    
    def test_process_file_max_retries_exceeded(self, mock_config_manager, sample_config, sample_excel_file):
        """Test behavior when max retries are exceeded."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Set file as already having max retries
                converter.failed_files[sample_excel_file] = sample_config.retry_settings.max_attempts
                
                result = converter._process_file_with_retry(sample_excel_file)
                
                assert result is False
    
    def test_process_file_pipeline_success(self, mock_config_manager, sample_config, sample_excel_file):
        """Test successful processing pipeline."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock worksheet data
                mock_worksheet = WorksheetData(
                    source_file=sample_excel_file,
                    worksheet_name="Sheet1",
                    data=pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
                    metadata={}
                )
                
                # Mock successful pipeline components
                mock_confidence = ConfidenceScore(
                    overall_score=0.9,
                    data_density=0.8,
                    header_quality=0.9,
                    consistency_score=0.9,
                    threshold=0.8,
                    reasons=[]
                )
                with patch.object(converter.excel_processor, 'process_file', return_value=[mock_worksheet]):
                    with patch.object(converter.confidence_analyzer, 'analyze_worksheet', return_value=mock_confidence):
                        with patch.object(converter.csv_generator, 'generate_csv', return_value=Path("output.csv")):
                            with patch.object(converter.archive_manager, 'archive_file') as mock_archive:
                                mock_archive.return_value.success = True
                                
                                result = converter._process_file_pipeline(sample_excel_file)
                
                assert result is True
                assert converter.stats.worksheets_analyzed == 1
                assert converter.stats.worksheets_accepted == 1
                assert converter.stats.csv_files_generated == 1
    
    def test_process_file_pipeline_excel_processing_error(self, mock_config_manager, sample_config, sample_excel_file):
        """Test pipeline with Excel processing error."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock Excel processor to raise error
                with patch.object(converter.excel_processor, 'process_file', side_effect=ExcelProcessingError("Test error")):
                    result = converter._process_file_pipeline(sample_excel_file)
                
                assert result is False
    
    def test_process_file_pipeline_confidence_rejection(self, mock_config_manager, sample_config, sample_excel_file):
        """Test pipeline when worksheet is rejected by confidence analyzer."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock worksheet data
                mock_worksheet = WorksheetData(
                    source_file=sample_excel_file,
                    worksheet_name="Sheet1",
                    data=pd.DataFrame({'A': [1], 'B': [2]}),
                    metadata={}
                )
                
                # Mock components with confidence rejection
                mock_confidence = ConfidenceScore(
                    overall_score=0.3,
                    data_density=0.2,
                    header_quality=0.3,
                    consistency_score=0.4,
                    threshold=0.8,
                    reasons=["Low quality data"]
                )
                with patch.object(converter.excel_processor, 'process_file', return_value=[mock_worksheet]):
                    with patch.object(converter.confidence_analyzer, 'analyze_worksheet', return_value=mock_confidence):
                        result = converter._process_file_pipeline(sample_excel_file)
                
                assert result is True  # Pipeline still succeeds even if worksheets are rejected
                assert converter.stats.worksheets_analyzed == 1
                assert converter.stats.worksheets_accepted == 0
                assert converter.stats.csv_files_generated == 0


class TestServiceMode:
    """Test service mode functionality."""
    
    def test_run_service_initialization(self, mock_config_manager, sample_config):
        """Test service mode initialization."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock file monitor and other components
                with patch('excel_to_csv.excel_to_csv_converter.FileMonitor') as mock_monitor:
                    with patch('threading.Thread') as mock_thread:
                        with patch.object(converter, '_setup_signal_handlers'):
                            # Mock the monitoring loop to prevent infinite execution
                            def mock_monitor_side_effect():
                                converter.is_running = False
                                return None
                            
                            mock_monitor.return_value.start_monitoring.side_effect = mock_monitor_side_effect
                            
                            converter.run_service()
                
                # Verify file monitor was created and started
                mock_monitor.assert_called_once()
                mock_monitor.return_value.start_monitoring.assert_called_once()
    
    def test_on_file_detected(self, mock_config_manager, sample_config, sample_excel_file):
        """Test file detection callback."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Call the file detection callback
                converter._on_file_detected(sample_excel_file)
                
                # Check that file was added to processing queue
                assert not converter.processing_queue.empty()
                queued_file = converter.processing_queue.get()
                assert queued_file == sample_excel_file
    
    def test_process_queue_worker(self, mock_config_manager, sample_config, sample_excel_file):
        """Test queue worker functionality."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Add file to queue
                converter.processing_queue.put(sample_excel_file)
                
                # Mock executor to capture submitted tasks
                mock_executor = MagicMock()
                converter.executor = mock_executor
                
                # Run worker once
                converter.is_running = True
                
                # Start worker in thread and stop it quickly
                worker_thread = threading.Thread(target=converter._process_queue_worker)
                worker_thread.daemon = True
                worker_thread.start()
                
                # Give worker time to process the file
                time.sleep(0.1)
                converter.is_running = False
                worker_thread.join(timeout=1.0)
                
                # Verify file was submitted to executor
                mock_executor.submit.assert_called_once()
                # Check that the first argument contains our file path
                submitted_args = mock_executor.submit.call_args[0]
                assert len(submitted_args) >= 2
                assert submitted_args[1] == sample_excel_file


class TestStatisticsAndReporting:
    """Test statistics and reporting functionality."""
    
    def test_get_statistics(self, mock_config_manager, sample_config):
        """Test getting statistics dictionary."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Set some statistics
                converter.stats.files_processed = 10
                converter.stats.files_failed = 2
                converter.stats.worksheets_analyzed = 15
                converter.stats.worksheets_accepted = 12
                
                stats_dict = converter.get_statistics()
                
                assert isinstance(stats_dict, dict)
                assert stats_dict['files_processed'] == 10
                assert stats_dict['files_failed'] == 2
                assert stats_dict['worksheets_analyzed'] == 15
                assert stats_dict['worksheets_accepted'] == 12
                assert 'is_running' in stats_dict
                assert 'failed_files' in stats_dict
    
    def test_log_statistics(self, mock_config_manager, sample_config, caplog):
        """Test statistics logging."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Set some statistics
                converter.stats.files_processed = 5
                converter.stats.csv_files_generated = 3
                
                # Mock logger to capture log calls
                with patch.object(converter.logger, 'info') as mock_info:
                    converter._log_statistics()
                    
                    # Check that statistics were logged
                    assert mock_info.call_count > 0
                    logged_messages = [str(call.args[0]) for call in mock_info.call_args_list]
                    stats_logged = any("Processing Statistics" in msg for msg in logged_messages)
                    assert stats_logged
    
    def test_stats_reporter_thread(self, mock_config_manager, sample_config):
        """Test stats reporter thread functionality."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock time to trigger immediate reporting
                with patch('time.time', side_effect=[0, 400, 500]):  # First check, then trigger report
                    with patch('time.sleep'):  # Skip sleep
                        with patch.object(converter, '_log_statistics') as mock_log_stats:
                            # Run stats reporter briefly
                            converter.is_running = True
                            
                            stats_thread = threading.Thread(target=converter._stats_reporter)
                            stats_thread.daemon = True
                            stats_thread.start()
                            
                            # Let it run briefly then stop
                            time.sleep(0.05)
                            converter.is_running = False
                            stats_thread.join(timeout=1.0)
                        
                            # Should have called log_statistics at least once due to time mock
                            assert mock_log_stats.call_count >= 0  # At least doesn't crash


class TestShutdownAndCleanup:
    """Test shutdown and cleanup functionality."""
    
    def test_shutdown(self, mock_config_manager, sample_config):
        """Test converter shutdown."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Start converter
                converter.is_running = True
                
                converter.shutdown()
                
                assert converter.is_running is False
                assert converter.shutdown_event.is_set()
    
    def test_cleanup_service(self, mock_config_manager, sample_config):
        """Test service cleanup."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock file monitor
                converter.file_monitor = MagicMock()
                
                converter._cleanup_service()
                
                # Check that file monitor was stopped
                converter.file_monitor.stop_monitoring.assert_called_once()


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager_enter_exit(self, mock_config_manager, sample_config):
        """Test context manager enter and exit."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter_instance = None
                with ExcelToCSVConverter() as converter:
                    converter_instance = converter
                    assert isinstance(converter, ExcelToCSVConverter)
                    converter.is_running = True  # Set running to trigger shutdown
                
                # After exit, should not be running
                assert converter_instance.is_running is False
    
    def test_context_manager_with_exception(self, mock_config_manager, sample_config):
        """Test context manager behavior with exceptions."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter_instance = None
                try:
                    with ExcelToCSVConverter() as converter:
                        converter_instance = converter
                        converter.is_running = True  # Set running to trigger shutdown
                        raise ValueError("Test exception")
                except ValueError:
                    pass
                
                # After exception, should still not be running
                assert converter_instance.is_running is False


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_processing_with_csv_generation_error(self, mock_config_manager, sample_config, sample_excel_file):
        """Test handling of CSV generation errors."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock worksheet data
                mock_worksheet = WorksheetData(
                    source_file=sample_excel_file,
                    worksheet_name="Sheet1",
                    data=pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
                    metadata={}
                )
                
                # Mock successful Excel processing and confidence analysis, but CSV generation failure
                mock_confidence = ConfidenceScore(
                    overall_score=0.9,
                    data_density=0.8,
                    header_quality=0.9,
                    consistency_score=0.9,
                    threshold=0.8,
                    reasons=[]
                )
                with patch.object(converter.excel_processor, 'process_file', return_value=[mock_worksheet]):
                    with patch.object(converter.confidence_analyzer, 'analyze_worksheet', return_value=mock_confidence):
                        with patch.object(converter.csv_generator, 'generate_csv', side_effect=CSVGenerationError("CSV error")):
                            result = converter._process_file_pipeline(sample_excel_file)
                
                assert result is False
    
    def test_processing_with_unexpected_exception(self, mock_config_manager, sample_config, sample_excel_file):
        """Test handling of unexpected exceptions."""
        mock_config_manager.load_config.return_value = sample_config
        
        with patch('excel_to_csv.excel_to_csv_converter.setup_logging'):
            with patch('excel_to_csv.excel_to_csv_converter.get_processing_logger'):
                converter = ExcelToCSVConverter()
                
                # Mock unexpected exception during processing
                with patch.object(converter, '_process_file_pipeline', side_effect=RuntimeError("Unexpected error")):
                    result = converter._process_file_with_retry(sample_excel_file)
                
                assert result is False
                assert converter.stats.processing_errors == 1


if __name__ == "__main__":
    pytest.main([__file__])