# Excel-to-CSV Converter - System Design

## Overview

The Excel-to-CSV Converter is an intelligent automation system that provides **dual operation modes**:
- **Service Mode**: Continuous monitoring and automated processing
- **CLI Mode**: On-demand file processing and analysis

## Architecture

### Service Mode Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Service Orchestrator                        │
│                   (ExcelToCSVConverter)                        │
├─────────────────────────────────────────────────────────────────┤
│  • Signal Handling (SIGINT/SIGTERM)                           │
│  • Thread Pool Management (Configurable Concurrency)          │
│  • Statistics Reporting (Every 5 minutes)                     │
│  • Retry Logic with Exponential Backoff                       │
│  • Graceful Shutdown with Resource Cleanup                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  File Monitor   │    │ Processing Queue │    │   Thread Pool   │
│                 │    │                 │    │                 │
│ • Real-time     │───▶│ • Thread-safe   │───▶│ • Concurrent    │
│   Monitoring    │    │   Queue         │    │   Processing    │
│ • Pattern       │    │ • Debouncing    │    │ • Max Workers:  │
│   Filtering     │    │   (2 seconds)   │    │   Configurable  │
│ • Multi-folder  │    │ • File          │    │ • Error         │
│   Support       │    │   Stability     │    │   Isolation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
              ┌─────────────────────────────────────────────────┐
              │            Processing Pipeline                   │
              ├─────────────────────────────────────────────────┤
              │                                                 │
              │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
              │  │   Excel     │  │ Confidence  │  │   CSV   │  │
              │  │ Processor   │─▶│  Analyzer   │─▶│Generator│  │
              │  │             │  │             │  │         │  │
              │  │ • pandas    │  │ • 90% AI    │  │ • Safe  │  │
              │  │ • openpyxl  │  │   Threshold │  │   Names │  │
              │  │ • Metadata  │  │ • Multi-    │  │ • Type  │  │
              │  │   Extract   │  │   Component │  │   Preserv │  │
              │  └─────────────┘  │   Scoring   │  │ • Dedup │  │
              │                   └─────────────┘  └─────────┘  │
              └─────────────────────────────────────────────────┘
```

## Core Components

### 1. Service Orchestrator (`ExcelToCSVConverter`)

**Responsibilities:**
- Coordinate all system components
- Manage service lifecycle (start/stop)
- Handle concurrent processing with thread pools
- Implement retry logic for failed files
- Provide statistics and monitoring
- Ensure graceful shutdown

**Key Features:**
- **Multi-threaded Processing**: Configurable concurrent file processing
- **Signal Handling**: Responds to SIGINT/SIGTERM for clean shutdown  
- **Statistics Reporting**: Periodic logging of processing metrics
- **Error Recovery**: Retry failed files with exponential backoff
- **Resource Management**: Proper cleanup of threads and file handles

### 2. File Monitor (`FileMonitor`)

**Responsibilities:**
- Watch configured directories for Excel files
- Filter files based on patterns (*.xlsx, *.xls)
- Handle file system events (create, modify)
- Implement debouncing to ensure file stability

**Key Features:**
- **Cross-platform Monitoring**: Uses watchdog library
- **Event Debouncing**: 2-second delay to ensure files are complete
- **Pattern Filtering**: Configurable file extension matching
- **Multi-directory Support**: Monitor multiple folders simultaneously
- **Initial Scanning**: Process existing files on startup

### 3. Excel Processor (`ExcelProcessor`)

**Responsibilities:**
- Read and parse Excel files (.xlsx, .xls)
- Extract worksheet data and metadata
- Handle various Excel formats and error conditions
- Provide memory-efficient processing

**Key Features:**
- **Dual Library Support**: pandas for data, openpyxl for metadata
- **Format Support**: .xlsx and .xls files
- **Error Handling**: Graceful handling of corrupt/locked files
- **Memory Management**: Configurable file size limits
- **Data Preservation**: Maintains original data types and formatting

### 4. Confidence Analyzer (`ConfidenceAnalyzer`)

**Responsibilities:**
- Analyze worksheets to determine if they contain data tables
- Apply 90% confidence threshold for decision making
- Provide detailed reasoning for accept/reject decisions

**Scoring Components:**
- **Data Density (40% weight)**: Ratio of non-empty cells, clustering patterns
- **Header Quality (30% weight)**: Header detection, pattern matching
- **Data Consistency (30% weight)**: Column type consistency, pattern regularity

**Intelligence Features:**
- **Pattern Recognition**: Detects common header patterns
- **Data Clustering**: Identifies rectangular data regions
- **Type Detection**: Recognizes numeric, date, and categorical data
- **Quality Scoring**: Multi-factor analysis with detailed reasoning

### 5. CSV Generator (`CSVGenerator`)

**Responsibilities:**
- Convert qualified worksheets to CSV format
- Generate safe, meaningful filenames
- Handle encoding and formatting options
- Manage duplicate file scenarios

**Key Features:**
- **Smart Naming**: `{filename}_{worksheet}.csv` pattern
- **Duplicate Handling**: Timestamp-based versioning
- **Character Safety**: Filesystem-safe filename generation
- **Type Preservation**: Maintains numeric and date formatting
- **Encoding Support**: Configurable character encoding

## Operation Modes

### Service Mode Usage

```bash
# Start continuous monitoring service
excel-to-csv service

# Service operations:
# 1. Loads configuration from config.yaml
# 2. Sets up logging (console + file + structured JSON)
# 3. Initializes all processing components
# 4. Starts file monitoring on configured directories
# 5. Processes files through full pipeline
# 6. Reports statistics every 5 minutes
# 7. Runs until SIGINT/SIGTERM received
```

**Service Mode Features:**
- **Continuous Operation**: Runs indefinitely until stopped
- **Automatic Processing**: No manual intervention required
- **Concurrent Handling**: Process multiple files simultaneously
- **Error Recovery**: Retry failed files with backoff
- **Performance Monitoring**: Real-time statistics and health checks
- **Clean Shutdown**: Graceful termination with resource cleanup

### CLI Mode Usage

```bash
# Process single file
excel-to-csv process data.xlsx

# Preview analysis (no output)
excel-to-csv preview data.xlsx --max-rows 20

# View statistics
excel-to-csv stats --watch --interval 30

# Validate configuration
excel-to-csv config-check
```

## Configuration System

### Hierarchical Configuration
1. **Default Values**: Built into application
2. **YAML Config File**: User-specified settings
3. **Environment Variables**: Runtime overrides
4. **Command-line Options**: Immediate overrides

### Key Configuration Areas

**Monitoring Settings:**
```yaml
monitoring:
  folders: ["./input", "./data"]
  file_patterns: ["*.xlsx", "*.xls"]
  process_existing: true
  max_file_size: 100  # MB
```

**Intelligence Settings:**
```yaml
confidence:
  threshold: 0.9
  weights:
    data_density: 0.4
    header_quality: 0.3
    consistency: 0.3
```

**Processing Settings:**
```yaml
processing:
  max_concurrent: 5
  retry:
    max_attempts: 3
    delay: 5
    backoff_factor: 2
```

## Error Handling Strategy

### Three-Tier Error Handling

1. **Component Level**: Each component handles its specific errors
2. **Pipeline Level**: Processing pipeline manages workflow errors  
3. **Service Level**: Orchestrator handles system-level errors

### Error Categories

**Recoverable Errors:**
- File access denied (retry with backoff)
- Network drive temporarily unavailable
- Temporary resource constraints

**Non-Recoverable Errors:**
- Corrupt Excel files
- Invalid configuration
- System resource exhaustion

**User Errors:**
- Invalid file paths
- Unsupported file formats
- Configuration syntax errors

## Performance Characteristics

### Service Mode Performance
- **File Detection**: <5 seconds from filesystem event
- **Processing Throughput**: 5 concurrent files (configurable)
- **Memory Usage**: <1GB during normal operation
- **File Size Limit**: 100MB per file (configurable)

### Processing Performance
- **Small Files** (<1MB): <2 seconds
- **Medium Files** (1-10MB): <15 seconds  
- **Large Files** (10-50MB): <30 seconds
- **Confidence Analysis**: <5 seconds per worksheet

## Monitoring and Observability

### Logging Levels
- **DEBUG**: Detailed component operations
- **INFO**: Processing events and decisions
- **WARNING**: Recoverable error conditions
- **ERROR**: Failed operations requiring attention
- **CRITICAL**: System-level failures

### Log Formats
- **Console**: Human-readable for interactive use
- **File**: Structured format with rotation
- **JSON**: Machine-readable for log aggregation

### Statistics Tracking
- Files processed/failed counts
- Worksheet acceptance rates
- Processing times and throughput
- Error frequency and types
- Service uptime and health metrics

## Deployment Considerations

### System Requirements
- **Python**: 3.9+ with pip package management
- **Memory**: 2GB minimum, 4GB recommended
- **CPU**: Multi-core recommended for concurrent processing
- **Storage**: SSD recommended for file I/O performance

### Production Deployment
- **Service Management**: systemd integration for Linux
- **Log Rotation**: Automatic log file management
- **Health Monitoring**: Statistics endpoint for monitoring systems
- **Configuration Management**: Environment-specific configs
- **Security**: File permission validation and path sanitization

This design enables both automated production workflows and flexible development/testing scenarios while maintaining high reliability and performance standards.