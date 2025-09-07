# Excel-to-CSV Converter Documentation

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Modes](#usage-modes)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Overview

The Excel-to-CSV Converter is an intelligent automation tool that converts Excel worksheets to CSV files using confidence-based data table detection. It supports both continuous monitoring and on-demand processing.

### Key Concepts

- **Confidence Analysis**: Each worksheet is analyzed to determine if it contains a data table with 90% confidence
- **Service Mode**: Continuous monitoring of folders for new Excel files
- **CLI Mode**: On-demand processing of individual files
- **Multi-threading**: Concurrent processing for improved performance

## Features

### ✨ Core Features

- **Intelligent Worksheet Analysis**: 90% confidence threshold for data table detection
- **Dual Operation Modes**: Service monitoring and CLI processing
- **Multi-format Support**: `.xlsx` and `.xls` file formats
- **Real-time Monitoring**: Watch multiple directories simultaneously
- **Concurrent Processing**: Multi-threaded file processing
- **Comprehensive Logging**: JSON-structured logs with rotation
- **Configurable Output**: Flexible CSV naming patterns and encodings

### 🎯 Analysis Components

The confidence analysis system evaluates worksheets using three weighted factors:

1. **Data Density (40%)**: Percentage of non-empty cells
2. **Header Quality (30%)**: Quality and consistency of column headers  
3. **Data Consistency (30%)**: Type consistency within columns

### 🛡️ Quality Assurance

- **90% Test Coverage**: Comprehensive unit and integration tests
- **Performance Testing**: Validated with 50MB+ files
- **Error Handling**: Graceful handling of corrupt files and edge cases
- **Memory Management**: Efficient processing of large datasets

## Installation

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM recommended for large files
- Write permissions for output directories

### Install from Source

```bash
# Clone the repository
git clone https://github.com/example/excel-to-csv-converter.git
cd excel-to-csv-converter

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install pandas>=2.0.0 openpyxl>=3.1.0 watchdog>=3.0.0 pyyaml>=6.0 click>=8.1.0
```

### Verify Installation

```bash
# Test the CLI
excel-to-csv --help

# Or use the standalone script
python excel_to_csv_converter.py --help
```

## Quick Start

### 1. Basic File Processing

```bash
# Process a single Excel file
excel-to-csv process input.xlsx --output ./converted/

# Preview worksheets without processing
excel-to-csv preview input.xlsx
```

### 2. Service Mode (Continuous Monitoring)

```bash
# Start monitoring with default settings
excel-to-csv service --folders ./input/ --output ./output/

# With custom configuration
excel-to-csv service --config config/my_config.yaml
```

### 3. Configuration Check

```bash
# Validate configuration file
excel-to-csv config-check --config config/production.yaml
```

## Configuration

### Configuration File Structure

```yaml
# monitoring.yaml
monitoring:
  folders:
    - "/path/to/input/folder1"
    - "/path/to/input/folder2"
  file_patterns:
    - "*.xlsx"
    - "*.xls"
  process_existing: true

output:
  folder: "/path/to/output"
  naming_pattern: "{filename}_{worksheet}.csv"
  include_timestamp: true
  encoding: "utf-8"

confidence:
  threshold: 0.9
  weights:
    data_density: 0.4
    header_quality: 0.3
    consistency: 0.3

processing:
  max_concurrent: 4
  max_file_size_mb: 100

logging:
  level: "INFO"
  format: "json"
  rotation_size: "10MB"
  retention_days: 30
```

### Environment Variables

Override configuration values using environment variables:

```bash
export EXCEL_TO_CSV_CONFIDENCE_THRESHOLD=0.8
export EXCEL_TO_CSV_MAX_CONCURRENT=8
export EXCEL_TO_CSV_OUTPUT_FOLDER="/custom/output"
export EXCEL_TO_CSV_LOG_LEVEL="DEBUG"
```

### Output Naming Patterns

Customize CSV file names using pattern variables:

- `{filename}`: Source Excel filename (without extension)
- `{worksheet}`: Worksheet name
- `{timestamp}`: ISO timestamp (if enabled)

Examples:
- `"{filename}_{worksheet}.csv"` → `data_Sheet1.csv`
- `"converted_{timestamp}_{filename}_{worksheet}.csv"` → `converted_20240101_120000_data_Sheet1.csv`

## Usage Modes

### Service Mode

Continuous monitoring mode for production environments:

```bash
# Basic service mode
excel-to-csv service

# With specific folders
excel-to-csv service --folders /data/input /data/archives --output /data/converted

# With configuration file
excel-to-csv service --config production.yaml

# Background service (Linux/macOS)
nohup excel-to-csv service --config production.yaml > service.log 2>&1 &
```

**Service Mode Features:**
- Real-time file system monitoring
- Automatic processing of new files
- Graceful shutdown handling (Ctrl+C, SIGTERM)
- Comprehensive error logging
- Statistics reporting

### CLI Mode

On-demand processing for development and batch operations:

```bash
# Process single file
excel-to-csv process file.xlsx

# Process with custom output location
excel-to-csv process file.xlsx --output /custom/output/

# Preview mode (analyze without converting)
excel-to-csv preview file.xlsx

# Batch processing
find /data/excel/ -name "*.xlsx" -exec excel-to-csv process {} \;
```

### Statistics and Monitoring

```bash
# View processing statistics
excel-to-csv stats

# Real-time monitoring (if service is running)
excel-to-csv stats --watch
```

## API Reference

### Core Classes

#### ExcelToCSVConverter

Main application orchestrator:

```python
from excel_to_csv import ExcelToCSVConverter, Config

# Create converter with custom config
config = Config(
    monitored_folders=[Path("/input")],
    output_folder=Path("/output"),
    confidence_threshold=0.9
)

converter = ExcelToCSVConverter()
converter.config = config

# Process single file
result = converter.process_single_file(Path("data.xlsx"))
print(f"Success: {result.success}")
print(f"CSV files: {result.csv_files}")

# Run service mode
with converter:
    converter.run_service()  # Blocks until shutdown
```

#### ExcelProcessor

Excel file reading and parsing:

```python
from excel_to_csv import ExcelProcessor

processor = ExcelProcessor(max_file_size_mb=50)
worksheets = processor.process_excel_file(Path("data.xlsx"))

for worksheet in worksheets:
    print(f"Sheet: {worksheet.name}")
    print(f"Size: {worksheet.row_count} x {worksheet.column_count}")
    print(f"Data preview:\n{worksheet.data.head()}")
```

#### ConfidenceAnalyzer

Worksheet confidence analysis:

```python
from excel_to_csv import ConfidenceAnalyzer

analyzer = ConfidenceAnalyzer(threshold=0.9)
confidence = analyzer.analyze_worksheet(worksheet_data)

print(f"Overall confidence: {confidence.overall_score:.2f}")
print(f"Data density: {confidence.data_density:.2f}")
print(f"Header quality: {confidence.header_quality:.2f}")
print(f"Consistency: {confidence.consistency_score:.2f}")
print(f"Reasons: {confidence.reasons}")
```

#### CSVGenerator

CSV file generation:

```python
from excel_to_csv import CSVGenerator, OutputConfig

config = OutputConfig(
    folder="/output",
    naming_pattern="{filename}_{worksheet}.csv",
    include_timestamp=True,
    encoding="utf-8"
)

generator = CSVGenerator(output_folder=Path("/output"), config=config)
csv_path = generator.generate_csv(worksheet_data)
print(f"Generated: {csv_path}")
```

### Data Models

#### WorksheetData

```python
@dataclass
class WorksheetData:
    name: str                    # Worksheet name
    data: pd.DataFrame          # Worksheet data
    file_path: Path            # Source Excel file path
    row_count: int             # Number of rows
    column_count: int          # Number of columns
```

#### ConfidenceScore

```python
@dataclass
class ConfidenceScore:
    overall_score: float        # Overall confidence (0-1)
    data_density: float        # Data density score (0-1)
    header_quality: float      # Header quality score (0-1)
    consistency_score: float   # Data consistency score (0-1)
    reasons: List[str]         # Decision reasons
```

#### Config

```python
@dataclass
class Config:
    monitored_folders: List[Path]     # Folders to monitor
    output_folder: Path               # Output directory
    confidence_threshold: float       # Minimum confidence (0-1)
    max_concurrent: int              # Max concurrent processes
    file_patterns: List[str]         # File patterns to match
    output_config: OutputConfig      # CSV output configuration
```

## Performance

### System Requirements

| File Size | RAM Required | Processing Time* |
|-----------|-------------|-----------------|
| < 10MB    | 2GB         | < 30 seconds    |
| 10-50MB   | 4GB         | 1-5 minutes     |
| 50-100MB  | 8GB         | 5-15 minutes    |
| > 100MB   | 16GB+       | 15+ minutes     |

*Times are approximate and depend on worksheet complexity and system performance.

### Performance Optimization

1. **Adjust Concurrent Processing**:
   ```yaml
   processing:
     max_concurrent: 4  # Adjust based on CPU cores and RAM
   ```

2. **File Size Limits**:
   ```yaml
   processing:
     max_file_size_mb: 100  # Prevent memory issues
   ```

3. **Lower Confidence Threshold**:
   ```yaml
   confidence:
     threshold: 0.7  # Process more worksheets (faster analysis)
   ```

4. **Memory Monitoring**:
   ```bash
   # Monitor memory usage during processing
   excel-to-csv stats --watch
   ```

### Performance Testing

Run performance benchmarks:

```bash
# Run performance tests (requires large test files)
pytest tests/performance/ -v -m performance

# Run quick performance regression tests
pytest tests/performance/test_performance.py::test_performance_regression_suite -v
```

## Troubleshooting

### Common Issues

#### 1. "File is locked or in use"

**Cause**: Excel file is open in another application.

**Solution**:
```bash
# Check if file is open
lsof filename.xlsx  # Linux/macOS
# Close Excel application or wait for automatic retry
```

#### 2. "No worksheets meet confidence threshold"

**Cause**: Worksheets don't contain clear data tables.

**Solution**:
```bash
# Preview the file to see confidence scores
excel-to-csv preview problematic_file.xlsx

# Lower the confidence threshold temporarily
excel-to-csv process file.xlsx --confidence-threshold 0.6
```

#### 3. "Memory error during processing"

**Cause**: File too large for available RAM.

**Solution**:
```yaml
# Reduce file size limit in config
processing:
  max_file_size_mb: 50  # Reduce from 100MB

# Or increase system RAM / use swap space
```

#### 4. "Permission denied writing CSV files"

**Cause**: Insufficient permissions for output directory.

**Solution**:
```bash
# Fix permissions
chmod 755 /output/directory
# Or use different output directory
excel-to-csv process file.xlsx --output ~/output/
```

### Debugging

#### Enable Debug Logging

```bash
# Command line
excel-to-csv process file.xlsx --log-level DEBUG

# Environment variable
export EXCEL_TO_CSV_LOG_LEVEL=DEBUG
excel-to-csv service
```

#### Check Configuration

```bash
# Validate configuration file
excel-to-csv config-check --config my_config.yaml

# Show current configuration
excel-to-csv config-check --show-current
```

#### Test Individual Components

```python
# Test Excel reading
from excel_to_csv import ExcelProcessor
processor = ExcelProcessor()
worksheets = processor.process_excel_file(Path("test.xlsx"))

# Test confidence analysis
from excel_to_csv import ConfidenceAnalyzer
analyzer = ConfidenceAnalyzer()
for worksheet in worksheets:
    confidence = analyzer.analyze_worksheet(worksheet)
    print(f"{worksheet.name}: {confidence.overall_score:.2f}")
```

### Log Analysis

Logs are stored in JSON format for easy analysis:

```bash
# View recent errors
tail -f logs/excel_to_csv.log | jq 'select(.level == "ERROR")'

# Count processing results
grep "processing_complete" logs/excel_to_csv.log | jq '.success' | sort | uniq -c

# Performance analysis
grep "processing_duration" logs/excel_to_csv.log | jq '.duration_seconds' | awk '{sum+=$1} END {print "Average:", sum/NR}'
```

## FAQ

### General Questions

**Q: What file formats are supported?**
A: Excel files (.xlsx and .xls formats). Other formats like CSV, ODS are not currently supported.

**Q: Can I process password-protected Excel files?**
A: No, password-protected files are not currently supported.

**Q: How is the 90% confidence threshold calculated?**
A: It's a weighted score combining data density (40%), header quality (30%), and data consistency (30%). See the [confidence analysis section](#-analysis-components) for details.

### Configuration Questions

**Q: Can I monitor network drives or cloud storage?**
A: Yes, as long as the paths are accessible to the Python process and file system events are properly generated.

**Q: How do I change the CSV delimiter?**
A: Currently, only comma-delimited CSV files are supported. This is a planned enhancement.

**Q: Can I exclude specific worksheets?**
A: Not directly, but worksheets that don't meet the confidence threshold are automatically excluded.

### Performance Questions

**Q: How can I speed up processing?**
A: Increase `max_concurrent` workers, lower the confidence threshold, or split large files into smaller ones.

**Q: What's the maximum file size supported?**
A: Default limit is 100MB, configurable up to available system memory. Very large files (>500MB) may cause memory issues.

**Q: Can I run multiple service instances?**
A: Yes, but ensure they monitor different folders or use different output directories to avoid conflicts.

### Integration Questions

**Q: How do I integrate with my existing workflow?**
A: Use the API directly, call the CLI from scripts, or monitor the JSON logs for processing events.

**Q: Can I customize the confidence analysis?**
A: Yes, you can adjust the weights and threshold in the configuration file:
```yaml
confidence:
  threshold: 0.8
  weights:
    data_density: 0.5
    header_quality: 0.3
    consistency: 0.2
```

**Q: How do I handle processing failures?**
A: Check the logs for detailed error messages, enable debug logging, and consider implementing retry logic in your workflow.

### Deployment Questions

**Q: How do I deploy this as a service?**
A: Use systemd on Linux, launchd on macOS, or Windows Service. See the examples directory for sample service configurations.

**Q: Can I run this in Docker?**
A: Yes, ensure volume mounts for input/output directories and proper file permissions.

**Q: How do I monitor the service health?**
A: Use the stats command, monitor log files, or implement health checks based on output file timestamps.

---

## Getting Help

- **Documentation**: This README and examples in the `examples/` directory
- **Issues**: Report bugs and feature requests on GitHub
- **Logs**: Check `logs/excel_to_csv.log` for detailed information
- **Testing**: Run the test suite with `pytest` to verify your installation

For additional support, please check the troubleshooting section or create an issue with:
1. Your configuration file
2. Sample input file (if possible)
3. Complete error logs
4. System information (OS, Python version, RAM)