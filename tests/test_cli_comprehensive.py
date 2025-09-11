"""Comprehensive tests for CLI module with high coverage."""

import pytest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner
import sys
import time

from excel_to_csv.cli import main, service, process, config_check, preview, stats, _display_stats
from excel_to_csv.models.data_models import (
    Config, LoggingConfig, OutputConfig, RetryConfig, 
    ArchiveConfig, ConfidenceConfig, WorksheetData
)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_excel_file(temp_workspace):
    """Create sample Excel file for testing."""
    file_path = temp_workspace / "test.xlsx"
    
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
def sample_config_file(temp_workspace):
    """Create sample config file for testing."""
    config_file = temp_workspace / "test_config.yaml"
    config_content = """
monitoring:
  folders:
    - "./input"
  file_patterns:
    - "*.xlsx"
confidence:
  threshold: 0.8
output:
  folder: "./output"
logging:
  level: "INFO"
processing:
  max_concurrent: 2
"""
    config_file.write_text(config_content, encoding='utf-8')
    return config_file


@pytest.fixture
def runner():
    """Create Click test runner."""
    return CliRunner()


class TestMainCommand:
    """Test main CLI command and group."""
    
    def test_main_without_args_shows_help(self, runner):
        """Test that main command without args shows help."""
        result = runner.invoke(main, [])
        
        assert result.exit_code == 0
        assert "Excel-to-CSV Converter" in result.output
        assert "Usage:" in result.output
    
    def test_main_with_version_flag(self, runner):
        """Test main command with --version flag."""
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "Excel-to-CSV Converter v1.0.0" in result.output
    
    def test_main_with_version_short_flag(self, runner):
        """Test main command with short version flag."""
        result = runner.invoke(main, ['-v'])
        
        assert result.exit_code == 0
        assert "Excel-to-CSV Converter v1.0.0" in result.output
    
    def test_main_with_config_option(self, runner, sample_config_file):
        """Test main command with config option."""
        result = runner.invoke(main, ['--config', str(sample_config_file)])
        
        assert result.exit_code == 0
        # Config is stored in context for subcommands
        assert "Usage:" in result.output
    
    def test_main_with_config_short_option(self, runner, sample_config_file):
        """Test main command with short config option."""
        result = runner.invoke(main, ['-c', str(sample_config_file)])
        
        assert result.exit_code == 0
        assert "Usage:" in result.output
    
    def test_main_with_nonexistent_config(self, runner):
        """Test main command with non-existent config file."""
        result = runner.invoke(main, ['--config', '/nonexistent/config.yaml'])
        
        # Click should handle the file existence check
        assert result.exit_code != 0
    
    def test_main_help_option(self, runner):
        """Test main command help option."""
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Show this message and exit" in result.output
        assert "Show version information" in result.output


class TestServiceCommand:
    """Test service command functionality."""
    
    def test_service_command_basic(self, runner):
        """Test basic service command invocation."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_converter.return_value.__exit__ = MagicMock(return_value=None)
            
            # Mock run_service to avoid infinite loop
            mock_instance.run_service.return_value = None
            
            result = runner.invoke(main, ['service'])
        
        assert result.exit_code == 0
        assert "Starting Excel-to-CSV Converter Service" in result.output
        assert "Press Ctrl+C to stop" in result.output
        mock_converter.assert_called_once_with(None)
        mock_instance.run_service.assert_called_once()
    
    def test_service_command_with_config(self, runner, sample_config_file):
        """Test service command with config file."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_converter.return_value.__exit__ = MagicMock(return_value=None)
            mock_instance.run_service.return_value = None
            
            result = runner.invoke(main, ['--config', str(sample_config_file), 'service'])
        
        assert result.exit_code == 0
        mock_converter.assert_called_once_with(str(sample_config_file))
    
    def test_service_command_keyboard_interrupt(self, runner):
        """Test service command handling KeyboardInterrupt."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_converter.return_value.__exit__ = MagicMock(return_value=None)
            
            # Mock KeyboardInterrupt
            mock_instance.run_service.side_effect = KeyboardInterrupt()
            
            result = runner.invoke(main, ['service'])
        
        assert result.exit_code == 0
        assert "Service stopped by user" in result.output
    
    def test_service_command_exception(self, runner):
        """Test service command handling general exceptions."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_converter.return_value.__exit__ = MagicMock(return_value=None)
            
            # Mock general exception
            mock_instance.run_service.side_effect = RuntimeError("Service error")
            
            result = runner.invoke(main, ['service'])
        
        assert result.exit_code == 1
        assert "Service error: Service error" in result.output


class TestProcessCommand:
    """Test process command functionality."""
    
    def test_process_command_success(self, runner, sample_excel_file):
        """Test successful process command."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.process_file.return_value = True
            mock_instance.get_statistics.return_value = {
                'worksheets_analyzed': 2,
                'worksheets_accepted': 1,
                'csv_files_generated': 1,
                'acceptance_rate': 50.0
            }
            
            result = runner.invoke(main, ['process', str(sample_excel_file)])
        
        assert result.exit_code == 0
        assert f"Processing file: {sample_excel_file}" in result.output
        assert "Processing completed successfully!" in result.output
        assert "Worksheets analyzed: 2" in result.output
        assert "Worksheets accepted: 1" in result.output
        assert "CSV files generated: 1" in result.output
        assert "Acceptance rate: 50.0%" in result.output
        
        mock_converter.assert_called_once_with(None)
        mock_instance.process_file.assert_called_once_with(sample_excel_file)
    
    def test_process_command_with_output_option(self, runner, sample_excel_file, temp_workspace):
        """Test process command with output directory option."""
        output_dir = temp_workspace / "custom_output"
        
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.process_file.return_value = True
            mock_instance.config.output_config = MagicMock()
            mock_instance.get_statistics.return_value = {
                'worksheets_analyzed': 1,
                'worksheets_accepted': 1,
                'csv_files_generated': 1
            }
            
            result = runner.invoke(main, ['process', str(sample_excel_file), '--output', str(output_dir)])
        
        assert result.exit_code == 0
        assert f"Output directory: {output_dir}" in result.output
        assert mock_instance.config.output_config.folder == output_dir
    
    def test_process_command_with_output_short_option(self, runner, sample_excel_file, temp_workspace):
        """Test process command with short output option."""
        output_dir = temp_workspace / "custom_output"
        
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.process_file.return_value = True
            mock_instance.config.output_config = MagicMock()
            mock_instance.get_statistics.return_value = {
                'worksheets_analyzed': 1,
                'worksheets_accepted': 1,
                'csv_files_generated': 1
            }
            
            result = runner.invoke(main, ['process', str(sample_excel_file), '-o', str(output_dir)])
        
        assert result.exit_code == 0
        assert f"Output directory: {output_dir}" in result.output
    
    def test_process_command_failure(self, runner, sample_excel_file):
        """Test process command when processing fails."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.process_file.return_value = False
            
            result = runner.invoke(main, ['process', str(sample_excel_file)])
        
        assert result.exit_code == 1
        assert "Processing failed!" in result.output
    
    def test_process_command_exception(self, runner, sample_excel_file):
        """Test process command handling exceptions."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_converter.side_effect = RuntimeError("Processing error")
            
            result = runner.invoke(main, ['process', str(sample_excel_file)])
        
        assert result.exit_code == 1
        assert "Processing error: Processing error" in result.output
    
    def test_process_command_stats_without_acceptance_rate(self, runner, sample_excel_file):
        """Test process command when stats don't include acceptance rate."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.process_file.return_value = True
            mock_instance.get_statistics.return_value = {
                'worksheets_analyzed': 1,
                'worksheets_accepted': 1,
                'csv_files_generated': 1
                # No acceptance_rate
            }
            
            result = runner.invoke(main, ['process', str(sample_excel_file)])
        
        assert result.exit_code == 0
        assert "Processing completed successfully!" in result.output
        # Should not crash when acceptance_rate is missing


class TestConfigCheckCommand:
    """Test config-check command functionality."""
    
    def test_config_check_success(self, runner):
        """Test successful config check."""
        mock_config = MagicMock()
        mock_config.monitored_folders = [Path("./input"), Path("./data")]
        mock_config.file_patterns = ["*.xlsx", "*.xls"]
        mock_config.confidence_threshold = 0.8
        mock_config.max_concurrent = 5
        mock_config.max_file_size_mb = 100
        mock_config.output_folder = Path("./output")
        mock_config.logging.level = "INFO"
        
        # Mock folder existence
        def mock_exists(self):
            return str(self) in ["./input", "./data"]
        
        with patch('excel_to_csv.cli.config_manager') as mock_config_manager:
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                mock_config_manager.load_config.return_value = mock_config
                
                result = runner.invoke(main, ['config-check'])
        
        assert result.exit_code == 0
        assert "Configuration loaded successfully" in result.output
        assert "Configuration Summary:" in result.output
        assert "Monitored folders: 2" in result.output
        assert "✓ ./input" in result.output
        assert "✓ ./data" in result.output
        assert "File patterns: *.xlsx, *.xls" in result.output
        assert "Confidence threshold: 0.8" in result.output
        assert "Max concurrent: 5" in result.output
        assert "Max file size: 100MB" in result.output
        assert "Output folder: ./output" in result.output
        assert "Logging level: INFO" in result.output
    
    def test_config_check_with_missing_folders(self, runner):
        """Test config check with some missing folders."""
        mock_config = MagicMock()
        mock_config.monitored_folders = [Path("./input"), Path("./missing")]
        mock_config.file_patterns = ["*.xlsx"]
        mock_config.confidence_threshold = 0.7
        mock_config.max_concurrent = 3
        mock_config.max_file_size_mb = 50
        mock_config.output_folder = None
        mock_config.logging.level = "DEBUG"
        
        # Mock folder existence - only ./input exists
        def mock_exists(self):
            return str(self) == "./input"
        
        with patch('excel_to_csv.cli.config_manager') as mock_config_manager:
            with patch('pathlib.Path.exists', side_effect=mock_exists):
                mock_config_manager.load_config.return_value = mock_config
                
                result = runner.invoke(main, ['config-check'])
        
        assert result.exit_code == 0
        assert "✓ ./input" in result.output
        assert "✗ ./missing" in result.output
        assert "Output folder: Same as source" in result.output
    
    def test_config_check_with_config_file(self, runner, sample_config_file):
        """Test config check with specific config file."""
        mock_config = MagicMock()
        mock_config.monitored_folders = [Path("./input")]
        mock_config.file_patterns = ["*.xlsx"]
        mock_config.confidence_threshold = 0.8
        mock_config.max_concurrent = 2
        mock_config.max_file_size_mb = 100
        mock_config.output_folder = Path("./output")
        mock_config.logging.level = "INFO"
        
        with patch('excel_to_csv.cli.config_manager') as mock_config_manager:
            with patch('pathlib.Path.exists', return_value=True):
                mock_config_manager.load_config.return_value = mock_config
                
                result = runner.invoke(main, ['--config', str(sample_config_file), 'config-check'])
        
        assert result.exit_code == 0
        mock_config_manager.load_config.assert_called_once_with(str(sample_config_file))
    
    def test_config_check_error(self, runner):
        """Test config check with configuration error."""
        with patch('excel_to_csv.cli.config_manager') as mock_config_manager:
            mock_config_manager.load_config.side_effect = RuntimeError("Config error")
            
            result = runner.invoke(main, ['config-check'])
        
        assert result.exit_code == 1
        assert "Configuration error: Config error" in result.output


class TestPreviewCommand:
    """Test preview command functionality."""
    
    def test_preview_command_success(self, runner, sample_excel_file):
        """Test successful preview command."""
        # Mock worksheet data
        mock_worksheet = MagicMock()
        mock_worksheet.worksheet_name = "Sheet1"
        mock_worksheet.row_count = 5
        mock_worksheet.column_count = 4
        mock_worksheet.data_density = 0.85
        mock_worksheet.data = pd.DataFrame({
            'ID': [1, 2, 3],
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35]
        })
        
        # Mock confidence score
        mock_confidence = MagicMock()
        mock_confidence.overall_score = 0.90
        mock_confidence.is_confident = True
        mock_confidence.reasons = ["High data density", "Clear headers", "Consistent data types"]
        
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.excel_processor.process_file.return_value = [mock_worksheet]
            mock_instance.confidence_analyzer.analyze_worksheet.return_value = mock_confidence
            
            result = runner.invoke(main, ['preview', str(sample_excel_file)])
        
        assert result.exit_code == 0
        assert f"Analyzing file: {sample_excel_file}" in result.output
        assert "Found 1 worksheets:" in result.output
        assert "1. Worksheet: 'Sheet1'" in result.output
        assert "Size: 5 rows × 4 columns" in result.output
        assert "Data density: 0.850" in result.output
        assert "Confidence: 0.900" in result.output
        assert "Decision: ACCEPT" in result.output
        assert "High data density" in result.output
        assert "Preview:" in result.output
        assert "Summary: 1/1 worksheets would be converted" in result.output
    
    def test_preview_command_with_max_rows(self, runner, sample_excel_file):
        """Test preview command with custom max rows."""
        mock_worksheet = MagicMock()
        mock_worksheet.worksheet_name = "Sheet1"
        mock_worksheet.row_count = 5
        mock_worksheet.column_count = 4
        mock_worksheet.data_density = 0.85
        mock_worksheet.data = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        
        mock_confidence = MagicMock()
        mock_confidence.overall_score = 0.90
        mock_confidence.is_confident = True
        mock_confidence.reasons = ["Good data"]
        
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.excel_processor.process_file.return_value = [mock_worksheet]
            mock_instance.confidence_analyzer.analyze_worksheet.return_value = mock_confidence
            
            result = runner.invoke(main, ['preview', str(sample_excel_file), '--max-rows', '3'])
        
        assert result.exit_code == 0
        # Should limit preview to 3 rows
    
    def test_preview_command_rejected_worksheet(self, runner, sample_excel_file):
        """Test preview command with rejected worksheet."""
        mock_worksheet = MagicMock()
        mock_worksheet.worksheet_name = "BadSheet"
        mock_worksheet.row_count = 2
        mock_worksheet.column_count = 1
        mock_worksheet.data_density = 0.30
        
        mock_confidence = MagicMock()
        mock_confidence.overall_score = 0.40
        mock_confidence.is_confident = False
        mock_confidence.reasons = ["Low data density", "Too few rows"]
        
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.excel_processor.process_file.return_value = [mock_worksheet]
            mock_instance.confidence_analyzer.analyze_worksheet.return_value = mock_confidence
            
            result = runner.invoke(main, ['preview', str(sample_excel_file)])
        
        assert result.exit_code == 0
        assert "Decision: REJECT" in result.output
        assert "Low data density" in result.output
        assert "Summary: 0/1 worksheets would be converted" in result.output
    
    def test_preview_command_no_worksheets(self, runner, sample_excel_file):
        """Test preview command when no worksheets found."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            mock_instance.excel_processor.process_file.return_value = []
            
            result = runner.invoke(main, ['preview', str(sample_excel_file)])
        
        assert result.exit_code == 0
        assert "No worksheets found in file" in result.output
    
    def test_preview_command_exception(self, runner, sample_excel_file):
        """Test preview command handling exceptions."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_converter.side_effect = RuntimeError("Preview error")
            
            result = runner.invoke(main, ['preview', str(sample_excel_file)])
        
        assert result.exit_code == 1
        assert "Preview error: Preview error" in result.output


class TestStatsCommand:
    """Test stats command functionality."""
    
    def test_stats_command_basic(self, runner):
        """Test basic stats command."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            
            with patch('excel_to_csv.cli._display_stats') as mock_display:
                result = runner.invoke(main, ['stats'])
        
        assert result.exit_code == 0
        mock_display.assert_called_once_with(mock_instance)
    
    def test_stats_command_watch_mode(self, runner):
        """Test stats command in watch mode."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            
            with patch('excel_to_csv.cli._display_stats') as mock_display:
                with patch('time.sleep') as mock_sleep:
                    with patch('click.clear') as mock_clear:
                        # Simulate KeyboardInterrupt after short delay
                        mock_sleep.side_effect = [None, KeyboardInterrupt()]
                        
                        result = runner.invoke(main, ['stats', '--watch'])
        
        assert result.exit_code == 0
        assert "Monitoring statistics" in result.output
        assert "Monitoring stopped" in result.output
        assert mock_display.call_count >= 1
    
    def test_stats_command_watch_with_interval(self, runner):
        """Test stats command in watch mode with custom interval."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            
            with patch('excel_to_csv.cli._display_stats'):
                with patch('time.sleep') as mock_sleep:
                    with patch('click.clear'):
                        mock_sleep.side_effect = KeyboardInterrupt()
                        
                        result = runner.invoke(main, ['stats', '--watch', '--interval', '60'])
        
        assert result.exit_code == 0
        # Should use 60 second interval
    
    def test_stats_command_exception(self, runner):
        """Test stats command handling exceptions."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_converter.side_effect = RuntimeError("Stats error")
            
            result = runner.invoke(main, ['stats'])
        
        assert result.exit_code == 1
        assert "Stats error: Stats error" in result.output


class TestDisplayStatsFunction:
    """Test _display_stats helper function."""
    
    def test_display_stats_complete(self):
        """Test _display_stats with complete statistics."""
        mock_converter = MagicMock()
        mock_converter.get_statistics.return_value = {
            'files_processed': 10,
            'files_failed': 2,
            'worksheets_analyzed': 15,
            'worksheets_accepted': 12,
            'csv_files_generated': 12,
            'processing_errors': 1,
            'acceptance_rate': 80.0,
            'is_running': True,
            'monitor': {
                'folders_count': 3,
                'pending_files': 5
            },
            'failed_files': {'file1.xlsx': 2, 'file2.xlsx': 1}
        }
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lambda: _display_stats(mock_converter))
        
        # Function should execute without errors
        mock_converter.get_statistics.assert_called_once()
    
    def test_display_stats_minimal(self):
        """Test _display_stats with minimal statistics."""
        mock_converter = MagicMock()
        mock_converter.get_statistics.return_value = {
            'files_processed': 0,
            'files_failed': 0,
            'worksheets_analyzed': 0,
            'worksheets_accepted': 0,
            'csv_files_generated': 0,
            'processing_errors': 0,
            'is_running': False,
            'failed_files': {}
        }
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lambda: _display_stats(mock_converter))
        
        mock_converter.get_statistics.assert_called_once()
    
    def test_display_stats_with_partial_data(self):
        """Test _display_stats with some missing optional fields."""
        mock_converter = MagicMock()
        mock_converter.get_statistics.return_value = {
            'files_processed': 5,
            'files_failed': 1,
            'worksheets_analyzed': 8,
            'worksheets_accepted': 6,
            'csv_files_generated': 6,
            'processing_errors': 0,
            'is_running': True,
            'failed_files': {}
            # Missing: acceptance_rate, monitor
        }
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lambda: _display_stats(mock_converter))
        
        mock_converter.get_statistics.assert_called_once()


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_help_for_all_commands(self, runner):
        """Test that help works for all commands."""
        commands = ['service', 'process', 'config-check', 'preview', 'stats']
        
        for command in commands:
            result = runner.invoke(main, [command, '--help'])
            assert result.exit_code == 0
            assert "Show this message and exit" in result.output
    
    def test_nonexistent_file_handling(self, runner):
        """Test handling of non-existent files in commands that require files."""
        nonexistent_file = "/path/to/nonexistent/file.xlsx"
        
        # Commands that require existing files should fail
        file_commands = ['process', 'preview']
        
        for command in file_commands:
            result = runner.invoke(main, [command, nonexistent_file])
            assert result.exit_code != 0
    
    def test_config_context_passing(self, runner, sample_config_file):
        """Test that config context is properly passed to subcommands."""
        with patch('excel_to_csv.cli.ExcelToCSVConverter') as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance
            
            with patch('excel_to_csv.cli._display_stats'):
                result = runner.invoke(main, ['--config', str(sample_config_file), 'stats'])
        
        assert result.exit_code == 0
        mock_converter.assert_called_once_with(str(sample_config_file))


if __name__ == "__main__":
    pytest.main([__file__])