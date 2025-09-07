"""Main entry point for Excel-to-CSV converter.

This module provides the main entry point that can be called
from the command line or imported as a module.
"""

import sys
from excel_to_csv.cli import main


def run() -> None:
    """Main entry point function."""
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    run()