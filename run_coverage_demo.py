#!/usr/bin/env python3
"""Demo script to show coverage testing configuration works."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run coverage demo showing the 90% threshold enforcement."""
    project_root = Path(__file__).parent
    
    print("Excel-to-CSV Converter - Coverage Testing Configuration Demo")
    print("=" * 60)
    print()
    
    print("1. Project structure verification:")
    print(f"   - Project root: {project_root}")
    print(f"   - Source code: {(project_root / 'src').exists()}")
    print(f"   - Tests directory: {(project_root / 'tests').exists()}")
    print(f"   - pyproject.toml: {(project_root / 'pyproject.toml').exists()}")
    print()
    
    print("2. Test discovery:")
    test_files = list((project_root / "tests").rglob("test_*.py"))
    print(f"   - Found {len(test_files)} test files")
    for test_file in test_files:
        relative_path = test_file.relative_to(project_root)
        print(f"     • {relative_path}")
    print()
    
    print("3. Source code files:")
    src_files = list((project_root / "src").rglob("*.py"))
    print(f"   - Found {len(src_files)} source files")
    for src_file in src_files:
        if "__pycache__" not in str(src_file):
            relative_path = src_file.relative_to(project_root)
            print(f"     • {relative_path}")
    print()
    
    print("4. Coverage configuration verification:")
    pyproject_file = project_root / "pyproject.toml"
    content = pyproject_file.read_text()
    
    # Check coverage configuration
    config_items = [
        ("90% threshold", "--cov-fail-under=90" in content),
        ("Branch coverage", "--cov-branch" in content or "branch = true" in content),
        ("Source directory", 'source = ["src"]' in content),
        ("HTML reports", "--cov-report=html" in content),
        ("XML reports", "--cov-report=xml" in content),
        ("Terminal reports", "term-missing" in content),
        ("Test exclusions", "*/tests/*" in content),
    ]
    
    for item, found in config_items:
        status = "✓" if found else "✗"
        print(f"   {status} {item}")
    print()
    
    print("5. Coverage system test:")
    print("   Running coverage configuration tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_coverage_config.py", 
            "-v", "--no-cov"
        ], cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✓ Coverage configuration tests passed")
            # Count passed tests
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "passed" in line and "failed" not in line:
                    print(f"   ✓ {line.strip()}")
                    break
        else:
            print("   ✗ Coverage configuration tests failed")
            print("   Error output:")
            for line in result.stderr.split('\n')[:5]:  # First 5 lines
                if line.strip():
                    print(f"     {line}")
    
    except Exception as e:
        print(f"   ✗ Error running tests: {e}")
    
    print()
    print("6. Ready for development!")
    print("   To run tests with coverage:")
    print("   $ python3 -m pytest --cov=excel_to_csv --cov-report=html")
    print()
    print("   To run specific test categories:")
    print("   $ python3 -m pytest tests/config/ -v")
    print("   $ python3 -m pytest tests/integration/ -v") 
    print("   $ python3 -m pytest -m \"not slow\" -v")
    print()
    print("   Coverage reports will be in:")
    print("   $ open coverage_html/index.html")
    print()


if __name__ == "__main__":
    main()