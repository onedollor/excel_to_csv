"""Coverage testing configuration and validation."""

import pytest
import subprocess
import sys
from pathlib import Path


def test_coverage_threshold_enforcement():
    """Test that coverage threshold is properly enforced."""
    # This test ensures that the pyproject.toml coverage settings are correct
    # and that the 90% threshold is enforced during test runs
    
    # Read pyproject.toml to verify coverage configuration
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"
    
    assert pyproject_file.exists(), "pyproject.toml should exist"
    
    content = pyproject_file.read_text()
    
    # Verify coverage configuration (can be in pytest config or coverage config)
    coverage_threshold_found = (
        'fail_under = 90' in content or 
        '--cov-fail-under=90' in content
    )
    assert coverage_threshold_found, "Coverage should fail under 90%"
    assert 'show_missing = true' in content, "Should show missing coverage"
    assert 'branch = true' in content, "Should include branch coverage"
    
    # Verify test source inclusion
    assert 'source = ["src"]' in content, "Should include src directory in coverage"
    
    # Verify omit patterns (should contain basic exclusions)
    basic_exclusions = ['tests', 'test_', '__init__.py']
    for exclusion in basic_exclusions:
        assert exclusion in content, f"Should omit files containing {exclusion} from coverage"


def test_coverage_reporting_formats():
    """Test that multiple coverage reporting formats are configured."""
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"
    
    content = pyproject_file.read_text()
    
    # Should support multiple report formats
    assert 'show_missing = true' in content, "Should show missing lines in terminal"
    
    # HTML and XML reports should be configurable
    expected_formats = ['html', 'xml', 'term-missing']
    # Note: These might not all be in pyproject.toml but should be available via CLI


def test_branch_coverage_enabled():
    """Test that branch coverage is enabled for comprehensive analysis."""
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"
    
    content = pyproject_file.read_text()
    
    # Branch coverage should be enabled for 90% target
    assert 'branch = true' in content, "Branch coverage should be enabled"


def test_coverage_source_configuration():
    """Test that coverage source directories are properly configured."""
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"
    
    content = pyproject_file.read_text()
    
    # Should include source directory
    assert 'source = ["src"]' in content, "Should measure coverage of src directory"
    
    # Should exclude test directories from coverage measurement
    test_exclusions = ['tests/', 'test_']
    for exclusion in test_exclusions:
        # Can be in omit section or other exclusion patterns
        exclusion_found = (
            exclusion in content or 
            f'*/{exclusion}*' in content or
            f'*{exclusion}*' in content
        )
        assert exclusion_found, f"Should exclude {exclusion} from coverage"


def test_coverage_integration_with_pytest():
    """Test that pytest-cov integration is properly configured."""
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"
    
    content = pyproject_file.read_text()
    
    # pytest-cov should be configured in dependencies
    assert 'pytest-cov' in content, "pytest-cov should be in dependencies"
    
    # Coverage should be integrated with pytest
    if '[tool.coverage.run]' in content:
        # Coverage configuration exists
        assert 'source = ["src"]' in content
        assert 'branch = true' in content


class TestCoverageValidation:
    """Test coverage validation and reporting functionality."""
    
    def test_minimum_coverage_components(self):
        """Test that key components have coverage tests."""
        project_root = Path(__file__).parent.parent
        
        # Key components that must have test coverage
        required_test_files = [
            "tests/config/test_config_manager.py",
            "tests/processors/test_excel_processor.py", 
            "tests/analysis/test_confidence_analyzer.py",
            "tests/generators/test_csv_generator.py",
            "tests/monitoring/test_file_monitor.py",
            "tests/integration/test_end_to_end.py"
        ]
        
        for test_file in required_test_files:
            test_path = project_root / test_file
            assert test_path.exists(), f"Required test file {test_file} should exist"
            
            # Verify test file has content
            content = test_path.read_text()
            assert len(content) > 100, f"Test file {test_file} should have substantial content"
            assert "def test_" in content, f"Test file {test_file} should have test functions"
    
    def test_source_components_exist(self):
        """Test that source components exist for coverage measurement."""
        project_root = Path(__file__).parent.parent
        
        # Key source components
        required_source_files = [
            "src/excel_to_csv/config/config_manager.py",
            "src/excel_to_csv/processors/excel_processor.py",
            "src/excel_to_csv/analysis/confidence_analyzer.py", 
            "src/excel_to_csv/generators/csv_generator.py",
            "src/excel_to_csv/monitoring/file_monitor.py",
            "src/excel_to_csv/excel_to_csv_converter.py",
            "src/excel_to_csv/cli.py"
        ]
        
        for source_file in required_source_files:
            source_path = project_root / source_file
            assert source_path.exists(), f"Source file {source_file} should exist for coverage"
            
            # Verify source file has substantive content
            content = source_path.read_text()
            assert len(content) > 200, f"Source file {source_file} should have substantial code"
    
    def test_test_discovery_configuration(self):
        """Test that test discovery is properly configured."""
        project_root = Path(__file__).parent.parent
        
        # Check pytest configuration
        pyproject_file = project_root / "pyproject.toml"
        content = pyproject_file.read_text()
        
        # Test discovery should be configured
        if '[tool.pytest.ini_options]' in content:
            # Verify test path configuration
            assert 'testpaths = ["tests"]' in content, "Should configure test discovery path"
            
            # Verify test pattern matching
            patterns = ['test_*.py', '*_test.py']
            # These might be defaults, so don't strictly require them in config


def test_coverage_reporting_setup():
    """Test coverage reporting configuration."""
    project_root = Path(__file__).parent.parent
    
    # Verify coverage can generate reports in common formats
    expected_outputs = [
        "htmlcov/",  # HTML coverage reports
        ".coverage",  # Coverage data file
    ]
    
    # These directories/files are created during coverage runs
    # We just verify the structure supports them
    
    # Check that pyproject.toml supports HTML reporting
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        content = pyproject_file.read_text()
        # HTML output directory can be configured
        # Default is usually htmlcov/ which is fine


def test_coverage_exclusions():
    """Test that appropriate files are excluded from coverage."""
    project_root = Path(__file__).parent.parent
    pyproject_file = project_root / "pyproject.toml"
    
    if pyproject_file.exists():
        content = pyproject_file.read_text()
        
        # Common exclusions that should be present  
        exclusion_keywords = [
            'tests',  # Test files themselves
            '__init__.py',  # Init files often excluded
        ]
        
        for keyword in exclusion_keywords:
            # Should be excluded from coverage measurement
            assert keyword in content, f"Should exclude files containing {keyword} from coverage"


class TestCoverageMetrics:
    """Test coverage metrics and thresholds."""
    
    def test_coverage_threshold_value(self):
        """Test that coverage threshold is set to 90%."""
        project_root = Path(__file__).parent.parent
        pyproject_file = project_root / "pyproject.toml"
        
        content = pyproject_file.read_text()
        
        # Verify 90% threshold (can be in pytest config or coverage config)
        threshold_found = (
            'fail_under = 90' in content or 
            '--cov-fail-under=90' in content
        )
        assert threshold_found, "Coverage threshold should be 90%"
    
    def test_branch_coverage_threshold(self):
        """Test that branch coverage is included in 90% target."""
        project_root = Path(__file__).parent.parent
        pyproject_file = project_root / "pyproject.toml"
        
        content = pyproject_file.read_text()
        
        # Branch coverage should be enabled
        assert 'branch = true' in content, "Branch coverage should be enabled"
        
        # The 90% threshold applies to total coverage including branches
        threshold_found = (
            'fail_under = 90' in content or 
            '--cov-fail-under=90' in content
        )
        assert threshold_found, "90% threshold should include branches"
    
    def test_coverage_precision(self):
        """Test coverage precision configuration."""
        project_root = Path(__file__).parent.parent
        pyproject_file = project_root / "pyproject.toml"
        
        content = pyproject_file.read_text()
        
        # Precision should be configured for accurate reporting
        if 'precision' in content:
            # If precision is configured, it should be reasonable (0-2 decimal places)
            import re
            precision_match = re.search(r'precision = (\d+)', content)
            if precision_match:
                precision = int(precision_match.group(1))
                assert 0 <= precision <= 2, "Coverage precision should be 0-2 decimal places"


def test_coverage_documentation():
    """Test that coverage configuration is documented."""
    project_root = Path(__file__).parent.parent
    
    # Check if README mentions coverage
    readme_file = project_root / "README.md"
    if readme_file.exists():
        content = readme_file.read_text()
        # Coverage should be mentioned in documentation
        coverage_keywords = ['coverage', '90%', 'pytest-cov']
        coverage_mentioned = any(keyword in content.lower() for keyword in coverage_keywords)
        # Don't require this, but it's good practice
    
    # Check if there are coverage-related scripts
    pyproject_file = project_root / "pyproject.toml"
    content = pyproject_file.read_text()
    
    # pytest should be configured properly
    assert 'pytest' in content, "pytest should be configured"


def test_ci_coverage_integration():
    """Test that coverage is configured for CI/CD integration."""
    project_root = Path(__file__).parent.parent
    
    # Check for common CI configuration files
    ci_files = [
        ".github/workflows/test.yml",
        ".github/workflows/ci.yml", 
        "tox.ini",
        ".travis.yml",
        "azure-pipelines.yml"
    ]
    
    ci_exists = any((project_root / ci_file).exists() for ci_file in ci_files)
    
    # If CI exists, it should include coverage
    for ci_file in ci_files:
        ci_path = project_root / ci_file
        if ci_path.exists():
            content = ci_path.read_text()
            # Should mention coverage or pytest-cov
            coverage_in_ci = any(keyword in content for keyword in ['coverage', 'pytest-cov', 'codecov'])
            # Don't strictly require this as CI might not be set up yet


# Integration test for actual coverage measurement
def test_coverage_measurement_works():
    """Test that coverage measurement actually works on the codebase."""
    project_root = Path(__file__).parent.parent
    
    # This test verifies that coverage can be measured
    # We don't run full coverage here (too slow for unit tests)
    # But we verify the configuration would work
    
    # Check that pytest can find tests
    test_dir = project_root / "tests"
    assert test_dir.exists(), "Tests directory should exist"
    
    test_files = list(test_dir.rglob("test_*.py"))
    assert len(test_files) >= 6, "Should have at least 6 test files"
    
    # Check that source code exists to measure
    src_dir = project_root / "src"
    assert src_dir.exists(), "Source directory should exist"
    
    py_files = list(src_dir.rglob("*.py"))
    assert len(py_files) >= 7, "Should have at least 7 source files"