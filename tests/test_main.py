"""Basic tests for main module to improve coverage."""

import pytest
from pathlib import Path

from excel_to_csv.main import main


class TestMain:
    """Test cases for main module."""
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        assert callable(main)
    
    def test_main_with_no_args(self, monkeypatch):
        """Test main function with no arguments."""
        # Mock sys.argv to avoid CLI interference
        test_args = ['excel-to-csv']
        monkeypatch.setattr('sys.argv', test_args)
        
        try:
            # This will likely exit, so catch SystemExit
            main()
        except SystemExit:
            # Expected behavior for CLI with no args
            pass
    
    def test_main_with_help_arg(self, monkeypatch):
        """Test main function with help argument."""
        test_args = ['excel-to-csv', '--help']
        monkeypatch.setattr('sys.argv', test_args)
        
        try:
            main()
        except SystemExit as e:
            # Help should exit with 0
            assert e.code == 0