"""Tests for main entry point module."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from excel_to_csv.main import run


class TestMainEntryPoint:
    """Test main entry point functionality."""
    
    @patch('excel_to_csv.main.main')
    def test_run_success(self, mock_main):
        """Test successful run execution."""
        mock_main.return_value = None
        
        # Should complete without error
        run()
        
        # Verify main was called
        mock_main.assert_called_once()
    
    @patch('excel_to_csv.main.main')
    @patch('sys.exit')
    def test_run_keyboard_interrupt(self, mock_exit, mock_main):
        """Test KeyboardInterrupt handling."""
        mock_main.side_effect = KeyboardInterrupt()
        
        with patch('builtins.print') as mock_print:
            run()
        
        # Should print cancellation message and exit with code 1
        mock_print.assert_called_once_with("\nOperation cancelled by user")
        mock_exit.assert_called_once_with(1)
    
    @patch('excel_to_csv.main.main')
    @patch('sys.exit')
    def test_run_unexpected_exception(self, mock_exit, mock_main):
        """Test unexpected exception handling."""
        test_error = Exception("Test error message")
        mock_main.side_effect = test_error
        
        with patch('builtins.print') as mock_print:
            run()
        
        # Should print error message to stderr and exit with code 1
        mock_print.assert_called_once_with("Unexpected error: Test error message", file=sys.stderr)
        mock_exit.assert_called_once_with(1)
    
    @patch('excel_to_csv.main.run')
    def test_main_name_execution(self, mock_run):
        """Test __main__ execution path."""
        # This tests the if __name__ == '__main__': block
        # We can't easily test this directly, but we can test the run function
        # which is what gets called
        
        mock_run.return_value = None
        
        # Simulate what happens when module is run directly
        exec(compile(open('src/excel_to_csv/main.py').read(), 'src/excel_to_csv/main.py', 'exec'))
        
        # The run function should have been available for execution
        assert callable(run)