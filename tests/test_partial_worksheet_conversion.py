"""Test partial worksheet conversion behavior - when not all worksheets qualify for CSV conversion."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.models.data_models import WorksheetData, ConfidenceScore, Config


class TestPartialWorksheetConversion:
    """Test cases for partial worksheet conversion scenarios."""
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_partial_worksheet_conversion_success(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test Excel file with mixed worksheet confidence scores."""
        # Setup mocks
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Create mock worksheet data - 5 worksheets with different confidence levels
        worksheets = [
            self._create_mock_worksheet("Sheet1", 0.85),  # High confidence - should convert
            self._create_mock_worksheet("Sheet2", 0.45),  # Low confidence - should reject
            self._create_mock_worksheet("Sheet3", 0.92),  # High confidence - should convert
            self._create_mock_worksheet("Sheet4", 0.30),  # Low confidence - should reject
            self._create_mock_worksheet("Sheet5", 0.78),  # High confidence - should convert
        ]
        
        # Mock confidence scores
        confidence_scores = [
            self._create_mock_confidence_score(0.85, True),   # Sheet1 - accepted
            self._create_mock_confidence_score(0.45, False),  # Sheet2 - rejected
            self._create_mock_confidence_score(0.92, True),   # Sheet3 - accepted
            self._create_mock_confidence_score(0.30, False),  # Sheet4 - rejected
            self._create_mock_confidence_score(0.78, True),   # Sheet5 - accepted
        ]
        
        # Setup component mocks
        converter.excel_processor.process_file = Mock(return_value=worksheets)
        converter.confidence_analyzer.analyze_worksheet = Mock(side_effect=confidence_scores)
        converter.csv_generator.generate_csv = Mock(side_effect=[
            Path("output/Sheet1.csv"),
            Path("output/Sheet3.csv"), 
            Path("output/Sheet5.csv")
        ])
        converter.archive_manager.archive_file = Mock(return_value=Mock(success=True))
        
        # Execute test
        file_path = Path("test_file.xlsx")
        result = converter._process_file_pipeline(file_path)
        
        # Verify results
        assert result is True
        
        # Verify all worksheets were analyzed
        assert converter.confidence_analyzer.analyze_worksheet.call_count == 5
        assert converter.stats.worksheets_analyzed == 5
        
        # Verify only 3 worksheets were accepted (85%, 92%, 78% confidence)
        assert converter.stats.worksheets_accepted == 3
        
        # Verify only 3 CSV files were generated
        assert converter.csv_generator.generate_csv.call_count == 3
        assert converter.stats.csv_files_generated == 3
        
        # Verify file was archived despite partial conversion
        converter.archive_manager.archive_file.assert_called_once()
        
        # Verify logging captured rejections and acceptances
        log_calls = mock_logger.info.call_args_list
        acceptance_logs = [call for call in log_calls if "accepted" in str(call)]
        rejection_logs = [call for call in log_calls if "rejected" in str(call)]
        
        assert len(acceptance_logs) == 3  # 3 worksheets accepted
        assert len(rejection_logs) == 2   # 2 worksheets rejected
        
        # Verify completion summary log
        completion_logs = [call for call in log_calls if "Completed processing" in str(call)]
        assert len(completion_logs) == 1
        # Should show "3/5 worksheets converted"
        assert "3/5" in str(completion_logs[0])
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_no_worksheets_qualify_for_conversion(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test Excel file where no worksheets meet confidence threshold."""
        # Setup mocks
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Create mock worksheet data - all with low confidence
        worksheets = [
            self._create_mock_worksheet("Sheet1", 0.45),  # Low confidence
            self._create_mock_worksheet("Sheet2", 0.30),  # Low confidence
            self._create_mock_worksheet("Sheet3", 0.55),  # Low confidence
        ]
        
        # Mock low confidence scores for all worksheets
        confidence_scores = [
            self._create_mock_confidence_score(0.45, False),
            self._create_mock_confidence_score(0.30, False),
            self._create_mock_confidence_score(0.55, False),
        ]
        
        # Setup component mocks
        converter.excel_processor.process_file = Mock(return_value=worksheets)
        converter.confidence_analyzer.analyze_worksheet = Mock(side_effect=confidence_scores)
        converter.csv_generator.generate_csv = Mock()  # Should not be called
        converter.archive_manager.archive_file = Mock(return_value=Mock(success=True))
        
        # Execute test
        file_path = Path("test_file.xlsx")
        result = converter._process_file_pipeline(file_path)
        
        # Verify results
        assert result is True  # Still successful even with no conversions
        
        # Verify all worksheets were analyzed
        assert converter.confidence_analyzer.analyze_worksheet.call_count == 3
        assert converter.stats.worksheets_analyzed == 3
        
        # Verify no worksheets were accepted
        assert converter.stats.worksheets_accepted == 0
        
        # Verify no CSV files were generated
        converter.csv_generator.generate_csv.assert_not_called()
        assert converter.stats.csv_files_generated == 0
        
        # Verify file was still archived (archiving condition includes len(qualified_worksheets) == 0)
        converter.archive_manager.archive_file.assert_called_once()
        
        # Verify logging shows all rejections
        log_calls = mock_logger.info.call_args_list
        rejection_logs = [call for call in log_calls if "rejected" in str(call)]
        assert len(rejection_logs) == 3  # All 3 worksheets rejected
        
        # Verify completion summary shows 0/3 conversion
        completion_logs = [call for call in log_calls if "Completed processing" in str(call)]
        assert len(completion_logs) == 1
        assert "0/3" in str(completion_logs[0])
    
    @patch('excel_to_csv.excel_to_csv_converter.setup_logging')
    @patch('excel_to_csv.excel_to_csv_converter.get_processing_logger')
    @patch('excel_to_csv.excel_to_csv_converter.config_manager')
    def test_csv_generation_failure_partial_success(self, mock_config_manager, mock_get_logger, mock_setup_logging):
        """Test scenario where some CSV generation fails but others succeed."""
        # Setup mocks
        mock_config = self._create_mock_config()
        mock_config_manager.load_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        converter = ExcelToCSVConverter()
        
        # Create mock worksheet data - 3 high confidence worksheets
        worksheets = [
            self._create_mock_worksheet("Sheet1", 0.85),
            self._create_mock_worksheet("Sheet2", 0.90),
            self._create_mock_worksheet("Sheet3", 0.88),
        ]
        
        # Mock high confidence scores for all
        confidence_scores = [
            self._create_mock_confidence_score(0.85, True),
            self._create_mock_confidence_score(0.90, True),
            self._create_mock_confidence_score(0.88, True),
        ]
        
        # Setup component mocks - CSV generation fails for middle worksheet
        from excel_to_csv.generators.csv_generator import CSVGenerationError
        converter.excel_processor.process_file = Mock(return_value=worksheets)
        converter.confidence_analyzer.analyze_worksheet = Mock(side_effect=confidence_scores)
        converter.csv_generator.generate_csv = Mock(side_effect=[
            Path("output/Sheet1.csv"),           # Success
            CSVGenerationError("Disk full"),    # Failure
            Path("output/Sheet3.csv"),           # Success
        ])
        converter.archive_manager.archive_file = Mock(return_value=Mock(success=True))
        
        # Execute test
        file_path = Path("test_file.xlsx")
        result = converter._process_file_pipeline(file_path)
        
        # Verify results
        assert result is True  # Still successful overall
        
        # Verify all worksheets were analyzed and accepted
        assert converter.stats.worksheets_analyzed == 3
        assert converter.stats.worksheets_accepted == 3
        
        # Verify only 2 CSV files were successfully generated (1 failed)
        assert converter.csv_generator.generate_csv.call_count == 3
        assert converter.stats.csv_files_generated == 2  # Only successful ones counted
        
        # Verify file was archived (csv_files_created > 0)
        converter.archive_manager.archive_file.assert_called_once()
        
        # Verify error was logged for failed CSV generation
        error_logs = mock_logger.error.call_args_list
        csv_error_logs = [call for call in error_logs if "Failed to generate CSV" in str(call)]
        assert len(csv_error_logs) == 1
        assert "Sheet2" in str(csv_error_logs[0])
        assert "Disk full" in str(csv_error_logs[0])
    
    def _create_mock_config(self) -> Mock:
        """Create a mock configuration object."""
        mock_config = Mock(spec=Config)
        mock_config.confidence_threshold = 0.7  # 70% threshold
        mock_config.archive_config = Mock()
        mock_config.archive_config.enabled = True
        mock_config.output_config = Mock()
        mock_config.max_file_size_mb = 100
        mock_config.logging = Mock()
        mock_config.retry_settings = Mock()
        mock_config.retry_settings.max_attempts = 3
        mock_config.retry_settings.delay = 1
        mock_config.retry_settings.backoff_factor = 2
        mock_config.retry_settings.max_delay = 60
        mock_config.monitored_folders = [Path("input")]
        mock_config.max_concurrent = 5
        return mock_config
    
    def _create_mock_worksheet(self, name: str, confidence: float) -> Mock:
        """Create a mock WorksheetData object."""
        worksheet = Mock(spec=WorksheetData)
        worksheet.worksheet_name = name
        worksheet.data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        worksheet.source_file = Path("test.xlsx")
        return worksheet
    
    def _create_mock_confidence_score(self, score: float, is_confident: bool) -> Mock:
        """Create a mock ConfidenceScore object."""
        confidence = Mock(spec=ConfidenceScore)
        confidence.overall_score = score
        confidence.is_confident = is_confident
        return confidence


# Integration test with real file processing (if we want to add later)
class TestPartialWorksheetConversionIntegration:
    """Integration tests for partial worksheet conversion with real Excel files."""
    
    @pytest.mark.slow
    def test_real_excel_file_partial_conversion(self):
        """Test with a real Excel file containing mixed data quality."""
        # This would require creating actual Excel test files
        # with worksheets of varying data quality for integration testing
        pytest.skip("Integration test - requires real Excel test files")