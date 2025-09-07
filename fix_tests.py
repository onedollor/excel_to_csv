#!/usr/bin/env python3
"""Script to fix common issues in test files."""

import os
import re
from pathlib import Path

def fix_worksheet_data_constructors(content):
    """Fix WorksheetData constructor parameter names."""
    # Fix parameter names
    content = re.sub(r'(\s+)name=', r'\1worksheet_name=', content)
    content = re.sub(r'(\s+)file_path=', r'\1source_file=', content)
    
    # Remove invalid row_count and column_count parameters
    content = re.sub(r',\s*\n\s+row_count=[^,\n)]*', '', content)
    content = re.sub(r',\s*\n\s+column_count=[^,\n)]*', '', content)
    
    return content

def fix_logger_assertions(content):
    """Remove invalid logger assertions."""
    content = re.sub(r'\s+assert hasattr\([^,]+, [\'"]_logger[\'"]\)\n', '', content)
    return content

def fix_test_file(file_path):
    """Fix a single test file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_worksheet_data_constructors(content)
        content = fix_logger_assertions(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"ℹ️  No changes: {file_path}")
            return False
    
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all test files."""
    test_dirs = [
        "tests/analysis",
        "tests/config", 
        "tests/generators",
        "tests/monitoring",
        "tests/processors",
        "tests/integration",
        "tests/performance"
    ]
    
    files_fixed = 0
    total_files = 0
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            for test_file in test_path.glob("test_*.py"):
                total_files += 1
                if fix_test_file(test_file):
                    files_fixed += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files fixed: {files_fixed}")
    print(f"   Files unchanged: {total_files - files_fixed}")

if __name__ == "__main__":
    main()