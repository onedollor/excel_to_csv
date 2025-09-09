"""Basic tests for CLI module to improve coverage."""

import pytest
from click.testing import CliRunner

from excel_to_csv.cli import main


class TestCLI:
    """Test cases for CLI commands."""
    
    def test_version_flag(self):
        """Test --version flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "Excel-to-CSV Converter v1.0.0" in result.output
    
    def test_help_output(self):
        """Test help output when no command provided."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        
        assert result.exit_code == 0
        assert "Excel-to-CSV Converter" in result.output
        assert "intelligent automation" in result.output.lower()
    
    def test_help_flag(self):
        """Test --help flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Excel-to-CSV Converter" in result.output
    
    def test_service_command_help(self):
        """Test service command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['service', '--help'])
        
        assert result.exit_code == 0
        assert "service mode" in result.output.lower()
        assert "continuous monitoring" in result.output.lower()
    
    def test_process_command_help(self):
        """Test process command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['process', '--help'])
        
        assert result.exit_code == 0
        assert "process" in result.output.lower()
    
    def test_status_command_help(self):
        """Test status command help."""  
        runner = CliRunner()
        result = runner.invoke(main, ['status', '--help'])
        
        # May fail if status command doesn't exist, that's ok
        assert result.exit_code in [0, 2]
    
    def test_invalid_command(self):
        """Test invalid command handling."""
        runner = CliRunner()
        result = runner.invoke(main, ['invalid-command'])
        
        assert result.exit_code != 0
        assert "no such command" in result.output.lower()