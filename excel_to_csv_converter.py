#!/usr/bin/env python3
"""Excel-to-CSV Converter - Standalone Entry Point Script

This is the main entry point for the Excel-to-CSV converter application.
It provides a simple way to run the converter from the command line.

Usage:
    python excel_to_csv_converter.py [command] [options]
    
Commands:
    service     - Run in continuous monitoring mode
    process     - Process a single Excel file
    preview     - Preview worksheets without processing
    stats       - Display processing statistics
    config-check - Validate configuration files
    
For detailed help on any command:
    python excel_to_csv_converter.py [command] --help
    
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

try:
    from excel_to_csv.cli import main
    from excel_to_csv import __version__
except ImportError as e:
    print(f"Error importing Excel-to-CSV converter modules: {e}")
    print("\nThis script must be run from the project root directory.")
    print("Please ensure the 'src' directory exists and contains the Excel-to-CSV modules.")
    sys.exit(1)


def print_banner():
    """Print application banner with version information."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Excel-to-CSV Converter v{__version__}                          ║
║                                                                              ║
║  Intelligent automation tool for converting Excel worksheets to CSV files   ║
║  with confidence-based data table detection and real-time monitoring        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_environment():
    """Check if the environment is properly set up."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append(f"Python 3.9+ required, but found {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check if running from correct directory
    if not (script_dir / "src" / "excel_to_csv").exists():
        issues.append("Excel-to-CSV source code not found. Run from project root directory.")
    
    # Check if configuration directory exists (optional)
    config_dir = script_dir / "config"
    if not config_dir.exists():
        print("Note: No config directory found. Default settings will be used.")
    
    return issues


def show_quick_help():
    """Show quick help when no arguments provided."""
    help_text = """
Available commands:

  service         Run continuous file monitoring mode
  process FILE    Process a single Excel file
  preview FILE    Preview worksheets without processing  
  stats           Show processing statistics
  config-check    Validate configuration files

Examples:

  # Start monitoring service with default config
  python excel_to_csv_converter.py service

  # Process a single file
  python excel_to_csv_converter.py process data/input.xlsx

  # Preview worksheets in a file
  python excel_to_csv_converter.py preview data/sample.xlsx

  # Check configuration
  python excel_to_csv_converter.py config-check --config config/my_config.yaml

For detailed help on any command:
  python excel_to_csv_converter.py [command] --help

For full documentation, see: README.md
"""
    print(help_text)


def main_entry():
    """Main entry point with error handling and environment checks."""
    try:
        # Show banner for help commands or when no args
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--help', '-h', 'help']):
            print_banner()
            if len(sys.argv) == 1:
                show_quick_help()
                sys.exit(0)
        
        # Check environment
        issues = check_environment()
        if issues:
            print("Environment issues detected:")
            for issue in issues:
                print(f"  ❌ {issue}")
            sys.exit(1)
        
        # Run the CLI main function
        main()
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you're running from the project root directory")
        print("2. Check that all dependencies are installed: pip install -e .")
        print("3. Verify Python version is 3.9 or higher")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nIf this error persists, please check:")
        print("1. Configuration files for syntax errors")
        print("2. Input file permissions and formats") 
        print("3. Available disk space for output files")
        print("\nFor detailed logging, use: --log-level DEBUG")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()