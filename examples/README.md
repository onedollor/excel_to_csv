# Excel-to-CSV Converter - Examples

This directory contains practical examples and configuration templates for the Excel-to-CSV Converter.

## Files Overview

### Configuration Examples

- **`sample_config.yaml`** - Comprehensive configuration template with all available options
- **`production_config.yaml`** - Production-optimized configuration for high-volume environments  
- **`development_config.yaml`** - Development-friendly configuration with debug settings

### Usage Examples

- **`sample_usage.py`** - Interactive Python script demonstrating all major features
- **`docker_service_example.py`** - Docker deployment example with containerization

## Quick Start

### 1. Try the Interactive Examples

```bash
# Run the comprehensive usage examples
python examples/sample_usage.py

# This will demonstrate:
# - Basic file processing
# - Confidence analysis
# - Custom configuration
# - CLI usage patterns
# - Service mode simulation
# - Error handling
```

### 2. Use Configuration Templates

```bash
# Copy a configuration template
cp examples/sample_config.yaml my_config.yaml

# Edit for your specific needs
nano my_config.yaml

# Test the configuration
excel-to-csv config-check --config my_config.yaml

# Use with the converter
excel-to-csv service --config my_config.yaml
```

### 3. Environment-Specific Configurations

**Development:**
```bash
excel-to-csv service --config examples/development_config.yaml
```

**Production:**
```bash
excel-to-csv service --config examples/production_config.yaml
```

## Configuration Templates

### Sample Config Features
- **Comprehensive**: All available options with explanations
- **Well-documented**: Inline comments explaining each setting
- **Examples**: Multiple naming patterns, environment overrides
- **Best practices**: Recommended settings for different scenarios

### Production Config Features
- **High-volume optimized**: 8 concurrent workers, 200MB file limit
- **Strict quality**: 90% confidence threshold
- **Robust logging**: JSON format, 90-day retention, compression
- **Error handling**: Automatic file movement for failed/low-confidence files
- **Statistics**: Automated performance monitoring

### Development Config Features
- **Debug-friendly**: DEBUG log level, console output, colors
- **Fast iteration**: Lower confidence threshold, quick failure
- **Local paths**: Relative paths for easy setup
- **Simplified**: Minimal configuration for testing

## Usage Examples

The `sample_usage.py` script provides six comprehensive examples:

1. **Basic File Processing** - Simple API usage with a single Excel file
2. **Confidence Analysis** - Deep dive into worksheet analysis
3. **Custom Configuration** - Advanced settings and CSV generation
4. **CLI Usage** - Command-line interface patterns
5. **Service Mode** - Continuous monitoring simulation
6. **Error Handling** - Troubleshooting and debugging techniques

Run individual examples:
```python
# Import specific examples if running in Python REPL
from examples.sample_usage import example_1_basic_file_processing
example_1_basic_file_processing()
```

## Docker Deployment

The `docker_service_example.py` shows how to containerize the service:

```bash
# Generate Docker deployment files
python examples/docker_service_example.py

# The script explains creating:
# - Dockerfile for the service
# - docker-compose.yml for orchestration
# - Health checks and monitoring
# - Volume mounts for data persistence
```

## Common Use Cases

### 1. Batch Processing
```bash
# Process all Excel files in a directory
find /data/excel/ -name "*.xlsx" -exec excel-to-csv process {} --output /data/csv/ \;
```

### 2. Continuous Monitoring
```bash
# Start service with production config
excel-to-csv service --config examples/production_config.yaml

# Background service (Linux/macOS)
nohup excel-to-csv service --config examples/production_config.yaml > service.log 2>&1 &
```

### 3. Custom Analysis
```python
from excel_to_csv import ConfidenceAnalyzer

# Custom confidence weights
analyzer = ConfidenceAnalyzer(
    threshold=0.8,
    weights={
        'data_density': 0.6,    # Prioritize data density
        'header_quality': 0.2,
        'consistency': 0.2
    }
)
```

### 4. Integration with Workflows
```bash
# Use in shell scripts
if excel-to-csv process input.xlsx --output output/; then
    echo "Processing successful"
    # Continue with next step
else
    echo "Processing failed"
    exit 1
fi
```

## Environment Variables

All configuration options can be overridden with environment variables:

```bash
# Common overrides
export EXCEL_TO_CSV_CONFIDENCE_THRESHOLD=0.8
export EXCEL_TO_CSV_OUTPUT_FOLDER="/custom/output"
export EXCEL_TO_CSV_LOG_LEVEL="DEBUG"
export EXCEL_TO_CSV_PROCESSING_MAX_CONCURRENT=6

# Run with environment overrides
excel-to-csv service --config examples/sample_config.yaml
```

## Troubleshooting Examples

### Debug Configuration Issues
```bash
# Check configuration validity
excel-to-csv config-check --config my_config.yaml

# Show current effective configuration
excel-to-csv config-check --show-current

# Enable debug logging
excel-to-csv process file.xlsx --log-level DEBUG
```

### Preview Before Processing
```bash
# See confidence scores without processing
excel-to-csv preview data.xlsx

# This shows which worksheets would be processed
# and their confidence scores
```

### Monitor Service Health
```bash
# View processing statistics
excel-to-csv stats

# Watch statistics in real-time
excel-to-csv stats --watch

# Check log files
tail -f logs/excel_to_csv.log
```

## Next Steps

1. **Start Simple**: Use `development_config.yaml` for initial testing
2. **Test Your Files**: Use `excel-to-csv preview` to check compatibility
3. **Customize**: Copy `sample_config.yaml` and adjust for your needs
4. **Deploy**: Use `production_config.yaml` for production environments
5. **Monitor**: Set up log monitoring and statistics collection

For detailed documentation, see `docs/README.md`.