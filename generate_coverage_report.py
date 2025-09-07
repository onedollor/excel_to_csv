#!/usr/bin/env python3
"""Coverage Report Generator for Excel-to-CSV Converter.

This script generates comprehensive coverage reports for the Excel-to-CSV converter project.
It provides multiple reporting formats and coverage analysis tools.

Usage:
    python generate_coverage_report.py [options]

Options:
    --format html,xml,term    Report formats to generate
    --threshold 90            Coverage threshold (default: 90)
    --output coverage_output  Output directory for reports
    --validate                Validate coverage configuration only
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class CoverageReportGenerator:
    """Generates and analyzes coverage reports for the Excel-to-CSV converter."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.coverage_dir = self.project_root / "coverage_output"
        self.reports_generated = []
        
    def validate_environment(self) -> bool:
        """Validate that coverage tools are available."""
        try:
            # Check if pytest-cov is available
            result = subprocess.run([
                sys.executable, "-c", 
                "import pytest_cov; import coverage; print('Coverage tools available')"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Coverage tools validated")
                return True
            else:
                print("✗ Coverage tools not available")
                print("Install with: pip install pytest-cov coverage")
                return False
                
        except Exception as e:
            print(f"✗ Error validating coverage tools: {e}")
            return False
    
    def validate_project_structure(self) -> bool:
        """Validate project structure for coverage analysis."""
        required_paths = [
            "src/excel_to_csv",
            "tests",
            "pyproject.toml"
        ]
        
        print("Validating project structure...")
        missing_paths = []
        
        for path in required_paths:
            full_path = self.project_root / path
            if not full_path.exists():
                missing_paths.append(path)
            else:
                print(f"  ✓ {path}")
        
        if missing_paths:
            print("✗ Missing required paths:")
            for path in missing_paths:
                print(f"  - {path}")
            return False
        
        print("✓ Project structure validated")
        return True
    
    def analyze_test_structure(self) -> Dict:
        """Analyze the test structure and count test methods."""
        test_dir = self.project_root / "tests"
        analysis = {
            "test_files": [],
            "total_test_files": 0,
            "total_test_methods": 0,
            "test_categories": {}
        }
        
        if not test_dir.exists():
            return analysis
        
        print("Analyzing test structure...")
        
        for test_file in test_dir.rglob("test_*.py"):
            relative_path = test_file.relative_to(self.project_root)
            
            # Count test methods in file
            try:
                content = test_file.read_text()
                test_methods = content.count("def test_")
                
                # Determine test category
                category = "unit"
                if "integration" in str(test_file):
                    category = "integration"
                elif "performance" in str(test_file):
                    category = "performance"
                
                file_info = {
                    "path": str(relative_path),
                    "test_methods": test_methods,
                    "category": category,
                    "size": test_file.stat().st_size
                }
                
                analysis["test_files"].append(file_info)
                analysis["total_test_methods"] += test_methods
                
                if category not in analysis["test_categories"]:
                    analysis["test_categories"][category] = {"files": 0, "methods": 0}
                
                analysis["test_categories"][category]["files"] += 1
                analysis["test_categories"][category]["methods"] += test_methods
                
                print(f"  ✓ {relative_path}: {test_methods} test methods ({category})")
                
            except Exception as e:
                print(f"  ✗ Error analyzing {relative_path}: {e}")
        
        analysis["total_test_files"] = len(analysis["test_files"])
        print(f"✓ Found {analysis['total_test_files']} test files with {analysis['total_test_methods']} test methods")
        
        return analysis
    
    def generate_coverage_data(self, threshold: int = 90) -> bool:
        """Generate coverage data by running tests."""
        print(f"Generating coverage data (threshold: {threshold}%)...")
        
        # Ensure coverage directory exists
        self.coverage_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "pytest",
            "--cov=excel_to_csv",
            "--cov-branch", 
            f"--cov-fail-under={threshold}",
            "--cov-report=",  # No terminal report during data generation
            "tests/"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ Coverage data generated successfully")
                return True
            else:
                print("✗ Coverage data generation failed")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])
                return False
                
        except subprocess.TimeoutExpired:
            print("✗ Coverage generation timed out")
            return False
        except Exception as e:
            print(f"✗ Error generating coverage data: {e}")
            return False
    
    def generate_html_report(self) -> Optional[Path]:
        """Generate HTML coverage report."""
        print("Generating HTML coverage report...")
        
        html_dir = self.project_root / "htmlcov"
        
        cmd = [
            sys.executable, "-m", "coverage", "html",
            "--directory", str(html_dir),
            "--title", "Excel-to-CSV Converter Coverage Report"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                index_file = html_dir / "index.html"
                if index_file.exists():
                    print(f"✓ HTML report generated: {index_file}")
                    self.reports_generated.append(("HTML", str(index_file)))
                    return index_file
            else:
                print("✗ HTML report generation failed")
                print("Error:", result.stderr)
                
        except Exception as e:
            print(f"✗ Error generating HTML report: {e}")
        
        return None
    
    def generate_xml_report(self) -> Optional[Path]:
        """Generate XML coverage report."""
        print("Generating XML coverage report...")
        
        xml_file = self.coverage_dir / "coverage.xml"
        
        cmd = [
            sys.executable, "-m", "coverage", "xml",
            "-o", str(xml_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and xml_file.exists():
                print(f"✓ XML report generated: {xml_file}")
                self.reports_generated.append(("XML", str(xml_file)))
                return xml_file
            else:
                print("✗ XML report generation failed")
                print("Error:", result.stderr)
                
        except Exception as e:
            print(f"✗ Error generating XML report: {e}")
        
        return None
    
    def generate_terminal_report(self) -> bool:
        """Generate terminal coverage report."""
        print("Generating terminal coverage report...")
        
        cmd = [
            sys.executable, "-m", "coverage", "report",
            "--show-missing",
            "--precision", "2"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓ Terminal report generated:")
                print("-" * 80)
                print(result.stdout)
                print("-" * 80)
                self.reports_generated.append(("Terminal", "Console output"))
                return True
            else:
                print("✗ Terminal report generation failed")
                print("Error:", result.stderr)
                
        except Exception as e:
            print(f"✗ Error generating terminal report: {e}")
        
        return False
    
    def generate_json_report(self) -> Optional[Path]:
        """Generate JSON coverage report."""
        print("Generating JSON coverage report...")
        
        json_file = self.coverage_dir / "coverage.json"
        
        cmd = [
            sys.executable, "-m", "coverage", "json",
            "-o", str(json_file),
            "--pretty-print"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and json_file.exists():
                print(f"✓ JSON report generated: {json_file}")
                self.reports_generated.append(("JSON", str(json_file)))
                return json_file
            else:
                print("✗ JSON report generation failed")
                print("Error:", result.stderr)
                
        except Exception as e:
            print(f"✗ Error generating JSON report: {e}")
        
        return None
    
    def analyze_coverage_gaps(self, json_report_path: Path) -> Dict:
        """Analyze coverage gaps from JSON report."""
        if not json_report_path.exists():
            return {}
        
        try:
            with open(json_report_path) as f:
                coverage_data = json.load(f)
            
            analysis = {
                "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                "files_analyzed": len(coverage_data.get("files", {})),
                "low_coverage_files": [],
                "missing_coverage": [],
                "summary": {}
            }
            
            files = coverage_data.get("files", {})
            for file_path, file_data in files.items():
                coverage_percent = file_data.get("summary", {}).get("percent_covered", 0)
                
                if coverage_percent < 85:  # Flag files below 85%
                    analysis["low_coverage_files"].append({
                        "file": file_path,
                        "coverage": coverage_percent,
                        "missing_lines": file_data.get("missing_lines", [])
                    })
                
                # Collect missing lines
                missing_lines = file_data.get("missing_lines", [])
                if missing_lines:
                    analysis["missing_coverage"].append({
                        "file": file_path,
                        "missing_lines": missing_lines
                    })
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing coverage gaps: {e}")
            return {}
    
    def create_summary_report(self, test_analysis: Dict, coverage_analysis: Dict) -> Path:
        """Create comprehensive summary report."""
        summary_file = self.coverage_dir / "coverage_summary.md"
        
        summary_content = f"""# Coverage Report Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview
- **Project**: Excel-to-CSV Converter
- **Coverage Threshold**: 90%
- **Report Generated**: {datetime.now().isoformat()}

## Test Structure Analysis
- **Total Test Files**: {test_analysis.get('total_test_files', 0)}
- **Total Test Methods**: {test_analysis.get('total_test_methods', 0)}

### Test Categories
"""
        
        for category, data in test_analysis.get('test_categories', {}).items():
            summary_content += f"- **{category.title()} Tests**: {data['files']} files, {data['methods']} methods\\n"
        
        if coverage_analysis:
            summary_content += f"""
## Coverage Analysis
- **Overall Coverage**: {coverage_analysis.get('total_coverage', 'N/A')}%
- **Files Analyzed**: {coverage_analysis.get('files_analyzed', 0)}
- **Low Coverage Files**: {len(coverage_analysis.get('low_coverage_files', []))}

### Coverage Gaps
"""
            for file_info in coverage_analysis.get('low_coverage_files', []):
                summary_content += f"- **{file_info['file']}**: {file_info['coverage']}% coverage\\n"
        
        summary_content += f"""
## Generated Reports
"""
        
        for report_type, report_path in self.reports_generated:
            summary_content += f"- **{report_type} Report**: `{report_path}`\\n"
        
        summary_content += f"""
## Usage Instructions

### View HTML Report
```bash
# Open in browser
open htmlcov/index.html
```

### Generate New Reports
```bash
# Run this script again
python generate_coverage_report.py --format html,xml,term

# Or use coverage directly
python -m coverage html --directory htmlcov
python -m coverage xml -o {self.coverage_dir}/coverage.xml
python -m coverage report --show-missing
```

### Integration with CI/CD
```bash
# Enforce coverage threshold in CI
python -m pytest --cov=excel_to_csv --cov-fail-under=90
```
"""
        
        summary_file.write_text(summary_content)
        print(f"✓ Summary report created: {summary_file}")
        return summary_file
    
    def run_full_analysis(self, formats: List[str], threshold: int) -> bool:
        """Run complete coverage analysis."""
        print("=" * 80)
        print("Excel-to-CSV Converter - Coverage Report Generation")
        print("=" * 80)
        
        # Step 1: Validate environment
        if not self.validate_environment():
            print("\\n✗ Environment validation failed")
            return False
        
        # Step 2: Validate project structure
        if not self.validate_project_structure():
            print("\\n✗ Project structure validation failed")
            return False
        
        # Step 3: Analyze test structure
        test_analysis = self.analyze_test_structure()
        
        # Step 4: Generate coverage data
        print("\\n" + "-" * 40)
        coverage_success = self.generate_coverage_data(threshold)
        
        coverage_analysis = {}
        
        if coverage_success:
            # Step 5: Generate requested report formats
            print("\\n" + "-" * 40)
            
            if "html" in formats:
                self.generate_html_report()
            
            if "xml" in formats:
                self.generate_xml_report()
            
            if "json" in formats:
                json_path = self.generate_json_report()
                if json_path:
                    coverage_analysis = self.analyze_coverage_gaps(json_path)
            
            if "term" in formats:
                self.generate_terminal_report()
        else:
            print("\\n⚠️  Coverage data generation failed - creating analysis report only")
        
        # Step 6: Create summary report
        print("\\n" + "-" * 40)
        self.create_summary_report(test_analysis, coverage_analysis)
        
        # Step 7: Final summary
        print("\\n" + "=" * 80)
        print("COVERAGE REPORT GENERATION COMPLETE")
        print("=" * 80)
        
        print(f"Reports generated: {len(self.reports_generated)}")
        for report_type, report_path in self.reports_generated:
            print(f"  ✓ {report_type}: {report_path}")
        
        if coverage_success:
            print(f"\\n✓ Coverage analysis completed successfully")
            if coverage_analysis.get('total_coverage', 0) >= threshold:
                print(f"✓ Coverage threshold met: {coverage_analysis['total_coverage']}% >= {threshold}%")
            else:
                print(f"⚠️  Coverage below threshold: {coverage_analysis['total_coverage']}% < {threshold}%")
        else:
            print(f"\\n⚠️  Coverage analysis incomplete (dependency or environment issues)")
        
        print(f"\\n📁 Reports available in: {self.coverage_dir}")
        print(f"📊 Summary report: {self.coverage_dir}/coverage_summary.md")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate coverage reports for Excel-to-CSV converter")
    parser.add_argument("--format", default="html,xml,term", 
                       help="Report formats to generate (comma-separated)")
    parser.add_argument("--threshold", type=int, default=90,
                       help="Coverage threshold percentage")
    parser.add_argument("--output", 
                       help="Output directory for reports")
    parser.add_argument("--validate", action="store_true",
                       help="Validate coverage configuration only")
    
    args = parser.parse_args()
    
    # Parse formats
    formats = [f.strip().lower() for f in args.format.split(",")]
    valid_formats = {"html", "xml", "json", "term"}
    formats = [f for f in formats if f in valid_formats]
    
    if not formats:
        formats = ["html", "xml", "term"]
    
    # Initialize generator
    generator = CoverageReportGenerator()
    
    if args.output:
        generator.coverage_dir = Path(args.output)
    
    if args.validate:
        # Validation only mode
        env_ok = generator.validate_environment()
        struct_ok = generator.validate_project_structure()
        
        if env_ok and struct_ok:
            print("✓ Coverage configuration validated successfully")
            return 0
        else:
            print("✗ Coverage configuration validation failed")
            return 1
    
    # Full analysis mode
    success = generator.run_full_analysis(formats, args.threshold)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())