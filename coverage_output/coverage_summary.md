# Coverage Report Summary - 2025-09-07 08:20:07

## Project Overview
- **Project**: Excel-to-CSV Converter
- **Coverage Threshold**: 90%
- **Report Generated**: 2025-09-07T08:20:07.619852

## Test Structure Analysis
- **Total Test Files**: 8
- **Total Test Methods**: 161

### Test Categories
- **Unit Tests**: 6 files, 138 methods\n- **Performance Tests**: 1 files, 11 methods\n- **Integration Tests**: 1 files, 12 methods\n
## Generated Reports

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
python -m coverage xml -o /home/lin/repo/excel_to_csv/coverage_output/coverage.xml
python -m coverage report --show-missing
```

### Integration with CI/CD
```bash
# Enforce coverage threshold in CI
python -m pytest --cov=excel_to_csv --cov-fail-under=90
```
