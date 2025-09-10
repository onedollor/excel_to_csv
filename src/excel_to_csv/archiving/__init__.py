"""File archiving functionality for Excel-to-CSV converter.

This package provides file archiving capabilities to automatically organize
processed Excel files by moving them to designated archive folders after
successful CSV conversion.
"""

from .archive_manager import ArchiveManager

__all__ = ["ArchiveManager"]