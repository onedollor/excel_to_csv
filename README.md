# Excel-to-CSV Converter

An intelligent automation tool for converting Excel worksheets to CSV files with confidence-based data table detection. Supports both CLI processing and continuous service monitoring.

## Features

### 🧠 Intelligent Analysis
- **90% Confidence Detection**: Uses sophisticated algorithms to identify data tables with 90% confidence
- **Multi-component Scoring**: Analyzes data density (40%), header quality (30%), and data consistency (30%)
- **Header Detection**: Automatically identifies and validates column headers
- **Data Type Recognition**: Detects numeric, date, and categorical data patterns

### 🔄 Flexible Operation Modes
- **Service Mode**: Continuous monitoring of configured directories for automated processing
- **CLI Mode**: Single-file processing for on-demand conversions
- **Preview Mode**: Analyze files without generating output to test confidence scoring

### 📁 Advanced File Handling
- **Multiple Directory Monitoring**: Watch multiple folders simultaneously
- **Pattern Filtering**: Configurable file patterns (*.xlsx, *.xls)
- **Duplicate Handling**: Timestamp-based naming to avoid file conflicts
- **Safe Processing**: File stability checking and lock detection

### 🛠️ Enterprise-Ready
- **Configurable Settings**: YAML-based configuration with environment overrides
- **Comprehensive Logging**: Structured JSON logging with multiple output formats
- **Error Handling**: Retry logic with exponential backoff
- **Performance Monitoring**: Built-in statistics and reporting
- **Graceful Shutdown**: Signal handling for clean service termination

## Installation

### From Source
```bash
git clone <repository-url>
cd excel_to_csv
pip install -e .
```

### Dependencies
- Python 3.9+
- pandas >= 2.0.0
- openpyxl >= 3.1.0
- watchdog >= 3.0.0
- pyyaml >= 6.0
- click >= 8.1.0

## Quick Start

### Service Mode (Continuous Monitoring)
```bash
# Start service with default configuration
excel-to-csv service

# Start with custom configuration
excel-to-csv --config config.yaml service
```

### CLI Mode (Single File Processing)
```bash
# Process a single file
excel-to-csv process data.xlsx

# Process with custom output directory
excel-to-csv process data.xlsx --output ./csv_files

# Preview analysis without generating files
excel-to-csv preview data.xlsx
```

### Configuration Check
```bash
# Validate configuration
excel-to-csv config-check

# Check with custom config file
excel-to-csv --config custom.yaml config-check
```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
# Directory monitoring
monitoring:
  folders:
    - "./input"
    - "./data"
  file_patterns:
    - "*.xlsx"
    - "*.xls"
  process_existing: true

# Confidence analysis settings
confidence:
  threshold: 0.9
  weights:
    data_density: 0.4
    header_quality: 0.3
    consistency: 0.3

# Output settings
output:
  folder: "./output"
  naming_pattern: "{filename}_{worksheet}.csv"
  include_timestamp: true
  encoding: "utf-8"

# Processing settings
processing:
  max_concurrent: 5
  retry:
    max_attempts: 3
    delay: 5
    backoff_factor: 2

# Logging configuration
logging:
  level: "INFO"
  file:
    enabled: true
    path: "./logs/excel_to_csv.log"
  console:
    enabled: true
```

## Service Mode Operations

### Starting the Service
The service monitors configured directories for Excel files and automatically processes them:

```bash
# Start service (runs until Ctrl+C)
excel-to-csv service

# Service will:
# 1. Scan existing files in monitored folders
# 2. Watch for new/modified Excel files
# 3. Process files using confidence analysis
# 4. Generate CSV files for qualifying worksheets
# 5. Provide periodic statistics reports
```

### Service Features
- **Real-time Monitoring**: Detects new files within 5 seconds
- **Concurrent Processing**: Processes multiple files simultaneously
- **Automatic Retry**: Failed files are retried with exponential backoff
- **Statistics Reporting**: Periodic logging of processing statistics
- **Graceful Shutdown**: Handles SIGINT/SIGTERM for clean shutdown

### Monitoring Statistics
```bash
# View current statistics
excel-to-csv stats

# Continuous monitoring (updates every 30 seconds)
excel-to-csv stats --watch

# Custom update interval
excel-to-csv stats --watch --interval 60
```

## How It Works

### 1. File Detection
- Monitors configured directories using cross-platform file system events
- Filters files based on configurable patterns (*.xlsx, *.xls)
- Handles file stability checking to avoid processing incomplete files

### 2. Excel Processing  
- Reads Excel files using pandas and openpyxl
- Extracts all worksheets with metadata
- Handles various Excel formats and error conditions

### 3. Confidence Analysis
- **Data Density Analysis (40%)**: Evaluates ratio of non-empty cells and data clustering
- **Header Quality (30%)**: Detects and scores potential column headers
- **Data Consistency (30%)**: Analyzes column data type consistency

### 4. CSV Generation
- Converts worksheets meeting 90% confidence threshold
- Preserves data types and formatting
- Generates meaningful filenames: `{source}_{worksheet}.csv`
- Handles duplicate files with timestamps

## Usage Examples

### Processing Business Reports
```bash
# Monitor business reports directory
excel-to-csv service --config business_reports.yaml
```

### Data Pipeline Integration
```python
from excel_to_csv import ExcelToCSVConverter

# Programmatic usage
converter = ExcelToCSVConverter("config.yaml")

# Process single file
success = converter.process_file("report.xlsx")

# Run as service
converter.run_service()  # Runs until shutdown
```

### Batch Processing
```bash
# Process multiple files in directory
for file in *.xlsx; do
    excel-to-csv process "$file"
done
```

## Performance

- **Processing Speed**: <30 seconds for 50MB Excel files
- **Memory Usage**: <1GB during normal operation  
- **Concurrency**: Configurable concurrent processing (default: 5)
- **File Size Limit**: Configurable (default: 100MB)

## Logging

The system provides comprehensive logging:

### Log Levels
- **DEBUG**: Detailed processing information
- **INFO**: General operation status  
- **WARNING**: Non-critical issues
- **ERROR**: Processing failures
- **CRITICAL**: System-level problems

### Log Formats
- **Console**: Human-readable format for interactive use
- **File**: Rotating log files with retention
- **Structured**: JSON format for log aggregation systems

### Key Log Events
- File detection and processing start/completion
- Confidence analysis decisions with scores
- CSV generation with output paths
- Error conditions with retry attempts
- Performance warnings and statistics

## Troubleshooting

### Common Issues

**Files not being processed**
- Check folder permissions and accessibility
- Verify file patterns in configuration
- Review confidence threshold settings
- Check logs for analysis rejection reasons

**Service won't start**
- Validate configuration with `excel-to-csv config-check`
- Ensure monitored directories exist
- Check for port conflicts or resource limits

**Low acceptance rate**
- Lower confidence threshold (but maintain quality)
- Review confidence scoring weights
- Use `excel-to-csv preview` to analyze specific files

**Performance issues**
- Reduce max_concurrent setting
- Increase memory limits
- Check disk I/O performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the project structure
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.