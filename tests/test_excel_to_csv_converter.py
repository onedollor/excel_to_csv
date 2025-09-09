"""Comprehensive tests for ExcelToCSVConverter class."""

import pytest
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter, ProcessingStats
from excel_to_csv.models.data_models import Config, WorksheetData, ConfidenceScore
from excel_to_csv.processors.excel_processor import ExcelProcessingError
from excel_to_csv.generators.csv_generator import CSVGenerationError
from excel_to_csv.monitoring.file_monitor import FileMonitorError


class TestProcessingStats:
    """Test ProcessingStats dataclass."""
    
    def test_processing_stats_initialization(self):
        """Test ProcessingStats default initialization."""
        stats = ProcessingStats()
        
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        assert stats.worksheets_analyzed == 0
        assert stats.worksheets_accepted == 0
        assert stats.csv_files_generated == 0
        assert stats.processing_errors == 0
        assert stats.files_archived == 0
        assert stats.archive_failures == 0


class TestExcelToCSVConverter:
    """Test cases for ExcelToCSVConverter class."""
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_init_default_config(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test converter initialization with default config."""
        mock_config = Mock()
        mock_config.max_file_size_mb = 100
        mock_config.confidence_threshold = 0.7
        mock_config.monitored_folders = [Path("input")]
        mock_config.max_concurrent = 5
        mock_config.logging = Mock()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Verify configuration loading
        mock_config_manager.load_config.assert_called_once_with(None)
        mock_setup_logging.assert_called_once_with(mock_config.logging)
        
        # Verify initialization
        assert converter.config == mock_config
        assert converter.excel_processor is not None
        assert converter.confidence_analyzer is not None
        assert converter.csv_generator is not None
        assert converter.archive_manager is not None
        assert isinstance(converter.stats, ProcessingStats)
        assert converter.failed_files == {}
        assert isinstance(converter.processing_queue, Queue)
        assert converter.file_monitor is None
        assert converter.executor is None
        assert converter.is_running is False
        assert converter.shutdown_event is not None
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_init_custom_config(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test converter initialization with custom config path."""
        mock_config = Mock()
        mock_config.max_file_size_mb = 200
        mock_config.confidence_threshold = 0.8
        mock_config.monitored_folders = [Path("custom")]
        mock_config.max_concurrent = 10
        mock_config.logging = Mock()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        config_path = "custom_config.yaml"
        converter = ExcelToCSVConverter(config_path)
        
        mock_config_manager.load_config.assert_called_once_with(config_path)
        assert converter.config == mock_config
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_setup_signal_handlers(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test signal handler setup."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_get_logger.return_value = Mock()
        
        with patch('excel_to_csv.excel_to_csv_converter.signal') as mock_signal:
            converter = ExcelToCSVConverter()
            
            # Verify signal handlers were set up
            assert mock_signal.signal.call_count >= 2  # SIGINT and SIGTERM
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_success(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test successful single file processing."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Mock the pipeline method
        with patch.object(converter, '_process_file_with_retry', return_value=True) as mock_pipeline:
            file_path = Path("test.xlsx")
            result = converter.process_file(file_path)
            
            assert result is True
            mock_pipeline.assert_called_once_with(file_path)
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_failure(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test single file processing failure."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Mock the pipeline method to fail
        with patch.object(converter, '_process_file_with_retry', return_value=False) as mock_pipeline:
            file_path = Path("test.xlsx")
            result = converter.process_file(file_path)
            
            assert result is False
            mock_pipeline.assert_called_once_with(file_path)
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_with_retry_success_first_attempt(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test file processing succeeds on first attempt."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        with patch.object(converter, '_process_file_pipeline', return_value=True) as mock_pipeline:
            file_path = Path("test.xlsx")
            result = converter._process_file_with_retry(file_path)
            
            assert result is True
            mock_pipeline.assert_called_once_with(file_path)
            assert file_path not in converter.failed_files
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_with_retry_max_attempts(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test file processing with maximum retry attempts."""
        mock_config = self._create_mock_config()
        mock_config.retry_settings.max_attempts = 3
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        with patch.object(converter, '_process_file_pipeline', return_value=False) as mock_pipeline:
            with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
                file_path = Path("test.xlsx")
                
                # Call multiple times to simulate retries
                result1 = converter._process_file_with_retry(file_path)  # First attempt
                result2 = converter._process_file_with_retry(file_path)  # Second attempt  
                result3 = converter._process_file_with_retry(file_path)  # Third attempt (should stop)
                
                assert result1 is False
                assert result2 is False
                assert result3 is False
                assert mock_pipeline.call_count == 3
                assert converter.failed_files[file_path] == 3
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_pipeline_success(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test successful file processing pipeline."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Mock dependencies
        mock_worksheet_data = Mock(spec=WorksheetData)
        mock_worksheet_data.worksheet_name = "Sheet1"
        mock_confidence_score = Mock(spec=ConfidenceScore)
        mock_confidence_score.overall_score = 0.8
        
        converter.excel_processor.process_file = Mock(return_value=[mock_worksheet_data])
        converter.confidence_analyzer.analyze_worksheet = Mock(return_value=mock_confidence_score)
        converter.csv_generator.generate_csv = Mock(return_value=Path("output.csv"))
        converter.archive_manager.archive_file = Mock(return_value=Mock(success=True))
        
        file_path = Path("test.xlsx")
        result = converter._process_file_pipeline(file_path)
        
        assert result is True
        converter.excel_processor.process_file.assert_called_once_with(file_path)
        converter.confidence_analyzer.analyze_worksheet.assert_called_once()
        converter.csv_generator.generate_csv.assert_called_once()
        assert converter.stats.files_processed == 1
        assert converter.stats.worksheets_analyzed == 1
        assert converter.stats.worksheets_accepted == 1
        assert converter.stats.csv_files_generated == 1
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_pipeline_low_confidence(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test file processing with low confidence worksheet."""
        mock_config = self._create_mock_config()
        mock_config.confidence_threshold = 0.7
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Mock low confidence score
        mock_worksheet_data = Mock(spec=WorksheetData)
        mock_worksheet_data.worksheet_name = "Sheet1"
        mock_confidence_score = Mock(spec=ConfidenceScore)
        mock_confidence_score.overall_score = 0.5  # Below threshold
        
        converter.excel_processor.process_file = Mock(return_value=[mock_worksheet_data])
        converter.confidence_analyzer.analyze_worksheet = Mock(return_value=mock_confidence_score)
        converter.csv_generator.generate_csv = Mock()
        
        file_path = Path("test.xlsx")
        result = converter._process_file_pipeline(file_path)
        
        assert result is True  # File processed successfully even if no CSV generated
        assert converter.stats.worksheets_analyzed == 1
        assert converter.stats.worksheets_accepted == 0  # Not accepted due to low confidence
        assert converter.stats.csv_files_generated == 0  # No CSV generated
        converter.csv_generator.generate_csv.assert_not_called()
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_process_file_pipeline_excel_processing_error(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test file processing with Excel processing error."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Mock Excel processing error
        converter.excel_processor.process_file = Mock(side_effect=ExcelProcessingError("Test error"))
        
        file_path = Path("test.xlsx")
        result = converter._process_file_pipeline(file_path)
        
        assert result is False
        assert converter.stats.processing_errors == 1
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_get_statistics(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test statistics retrieval."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Set some statistics
        converter.stats.files_processed = 10
        converter.stats.files_failed = 2
        converter.stats.csv_files_generated = 15
        
        stats = converter.get_statistics()
        
        assert isinstance(stats, dict)
        assert stats["files_processed"] == 10
        assert stats["files_failed"] == 2
        assert stats["csv_files_generated"] == 15
        assert "processing_rate" in stats
        assert "success_rate" in stats
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_shutdown(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test converter shutdown."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        with patch.object(converter, '_cleanup_service') as mock_cleanup:
            with patch('excel_to_csv.excel_to_csv_converter.shutdown_logging') as mock_shutdown_logging:
                converter.shutdown()
                
                assert converter.shutdown_event.is_set()
                mock_cleanup.assert_called_once()
                mock_shutdown_logging.assert_called_once()
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_context_manager(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test converter as context manager."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        with patch.object(ExcelToCSVConverter, 'shutdown') as mock_shutdown:
            with ExcelToCSVConverter() as converter:
                assert isinstance(converter, ExcelToCSVConverter)
            
            mock_shutdown.assert_called_once()
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_on_file_detected(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test file detection callback."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        file_path = Path("test.xlsx")
        converter._on_file_detected(file_path)
        
        # File should be added to processing queue
        assert not converter.processing_queue.empty()
        assert converter.processing_queue.get() == file_path
    
    def _create_mock_config(self) -> Mock:
        """Create a mock configuration object."""
        mock_config = Mock(spec=Config)
        mock_config.max_file_size_mb = 100
        mock_config.confidence_threshold = 0.7
        mock_config.monitored_folders = [Path("input")]
        mock_config.max_concurrent = 5
        mock_config.logging = Mock()
        mock_config.retry_settings = Mock()
        mock_config.retry_settings.max_attempts = 3
        mock_config.retry_settings.delay = 1
        mock_config.retry_settings.backoff_factor = 2
        mock_config.retry_settings.max_delay = 60
        mock_config.archive_config = Mock()
        mock_config.archive_config.enabled = True
        mock_config.output_config = Mock()
        return mock_config


class TestExcelToCSVConverterServiceMode:
    """Test service mode functionality separately due to complexity."""
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_run_service_initialization(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test service mode initialization."""
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Mock FileMonitor to avoid actual file system monitoring
        with patch('excel_to_csv.excel_to_csv_converter.FileMonitor') as mock_file_monitor_class:
            with patch.object(converter, 'shutdown_event') as mock_shutdown_event:
                mock_file_monitor = Mock()
                mock_file_monitor_class.return_value = mock_file_monitor
                mock_shutdown_event.wait.return_value = True  # Simulate immediate shutdown
                
                with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
                    mock_executor = Mock()
                    mock_executor_class.return_value = mock_executor
                    
                    converter.run_service()
                    
                    # Verify service initialization
                    mock_file_monitor_class.assert_called_once()
                    mock_file_monitor.start_monitoring.assert_called_once()
                    assert converter.is_running is False  # Should be False after shutdown
    
    def _create_mock_config(self) -> Mock:
        """Create a mock configuration object."""
        mock_config = Mock(spec=Config)
        mock_config.max_file_size_mb = 100
        mock_config.confidence_threshold = 0.7
        mock_config.monitored_folders = [Path("input")]
        mock_config.max_concurrent = 5
        mock_config.logging = Mock()
        mock_config.retry_settings = Mock()
        mock_config.retry_settings.max_attempts = 3
        mock_config.retry_settings.delay = 1
        mock_config.retry_settings.backoff_factor = 2
        mock_config.retry_settings.max_delay = 60
        mock_config.archive_config = Mock()
        mock_config.archive_config.enabled = True
        mock_config.output_config = Mock()
        mock_config.file_patterns = ["*.xlsx", "*.xls"]
        return mock_config