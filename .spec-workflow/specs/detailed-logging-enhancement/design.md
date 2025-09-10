# Detailed Logging Enhancement - Design Document

## Overview
This design document outlines the technical approach for implementing comprehensive logging across the excel_to_csv system to enable detailed operation tracking and analysis.

## Architecture

### Core Components

#### 1. Correlation ID System
```python
# New utility for correlation tracking
class CorrelationContext:
    _context: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id')
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        cls._context.set(correlation_id)
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        return cls._context.get(None)
    
    @classmethod
    def generate_correlation_id(cls) -> str:
        return str(uuid.uuid4())
```

#### 2. Enhanced Logger Configuration
```python
# Structured logging with correlation ID injection
class CorrelationFormatter(logging.Formatter):
    def format(self, record):
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = "NONE"
        return super().format(record)

# Enhanced log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
```

#### 3. Daily Log Rotation and Archival System
```python
# Daily rotating log handler with automatic archival
class DailyRotatingLogHandler:
    def __init__(self, log_dir: Path, base_filename: str = "excel_to_csv"):
        self.log_dir = log_dir
        self.base_filename = base_filename
        self.current_date = None
        self.current_handler = None
        self._setup_log_directory()
    
    def _setup_log_directory(self):
        """Create log directory structure"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "archive").mkdir(exist_ok=True)
    
    def get_current_handler(self) -> logging.Handler:
        """Get handler for current date, rotating if needed"""
        today = datetime.now().date()
        
        if self.current_date != today:
            self._rotate_logs(today)
        
        return self.current_handler
    
    def _rotate_logs(self, new_date: date):
        """Rotate to new log file and archive previous"""
        if self.current_handler:
            self._archive_current_log()
            self.current_handler.close()
        
        # Create new log file for today
        log_filename = f"{self.base_filename}_{new_date.strftime('%Y%m%d')}.log"
        log_path = self.log_dir / log_filename
        
        self.current_handler = logging.FileHandler(log_path, mode='a')
        self.current_date = new_date
    
    def _archive_current_log(self):
        """Archive current log file with compression"""
        if not self.current_handler:
            return
            
        current_log_path = Path(self.current_handler.baseFilename)
        if current_log_path.exists() and current_log_path.stat().st_size > 0:
            # Compress and move to archive
            archive_path = self.log_dir / "archive" / f"{current_log_path.stem}.gz"
            
            with open(current_log_path, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original log file
            current_log_path.unlink()

# Enhanced logging configuration with daily rotation
def setup_enhanced_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    structured_format: bool = True,
    daily_rotation: bool = True
) -> logging.Logger:
    """Setup enhanced logging with daily rotation and archival"""
    
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    
    # Setup daily rotation handler
    if daily_rotation:
        rotating_handler = DailyRotatingLogHandler(log_dir)
        file_handler = rotating_handler.get_current_handler()
    else:
        log_file = log_dir / "excel_to_csv.log"
        file_handler = logging.FileHandler(log_file)
    
    # Setup formatter
    if structured_format:
        formatter = CorrelationFormatter(LOG_FORMAT)
    else:
        formatter = logging.Formatter(LOG_FORMAT)
    
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    
    # Also add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger
```

#### 4. Operation Metrics Tracking
```python
@dataclass
class OperationMetrics:
    operation_name: str
    correlation_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool, error_type: Optional[str] = None):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_type = error_type
```

#### 5. Logging Decorators
```python
def log_operation(operation_name: str, log_args: bool = True, log_result: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            correlation_id = CorrelationContext.get_correlation_id()
            
            # Start metrics
            metrics = OperationMetrics(
                operation_name=operation_name,
                correlation_id=correlation_id or "NONE",
                start_time=time.time()
            )
            
            # Log operation start
            log_data = {"operation": operation_name, "status": "START"}
            if log_args:
                log_data["args"] = _sanitize_args(args, kwargs)
            
            logger.info("Operation started", extra={"structured": log_data})
            
            try:
                result = func(*args, **kwargs)
                
                # Complete metrics
                metrics.complete(success=True)
                
                # Log success
                success_data = {
                    "operation": operation_name,
                    "status": "SUCCESS",
                    "duration_ms": metrics.duration_ms
                }
                if log_result:
                    success_data["result"] = _sanitize_result(result)
                
                logger.info("Operation completed successfully", extra={"structured": success_data})
                return result
                
            except Exception as e:
                # Complete metrics with error
                metrics.complete(success=False, error_type=type(e).__name__)
                
                # Log error
                error_data = {
                    "operation": operation_name,
                    "status": "ERROR",
                    "duration_ms": metrics.duration_ms,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                logger.error("Operation failed", extra={"structured": error_data}, exc_info=True)
                raise
        
        return wrapper
    return decorator
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. **Logging Configuration Enhancement** (`src/excel_to_csv/utils/logging_config.py`)
   - Implement CorrelationContext and CorrelationFormatter
   - Implement DailyRotatingLogHandler for automatic log rotation
   - Update setup_logging() to use structured format with daily rotation
   - Add correlation ID injection to all log records
   - Add automatic log archival with compression

2. **Operation Metrics System** (`src/excel_to_csv/utils/metrics.py`)
   - Create OperationMetrics dataclass
   - Implement metrics collection and aggregation
   - Add performance tracking utilities

3. **Logging Decorators** (`src/excel_to_csv/utils/logging_decorators.py`)
   - Implement @log_operation decorator
   - Add argument and result sanitization
   - Create context managers for operation tracking

### Phase 2: Core Components Integration
1. **ExcelToCSVConverter Integration**
   - Add correlation ID generation for each file processing
   - Implement detailed operation logging for:
     - File processing pipeline stages
     - Worksheet analysis and acceptance/rejection
     - CSV generation attempts and results
     - Archive operations
   - Add performance metrics for processing times

2. **Component-Level Logging**
   - **ExcelProcessor**: Log worksheet extraction, data validation
   - **ConfidenceAnalyzer**: Log analysis criteria and scoring
   - **CSVGenerator**: Log generation attempts, success/failure reasons
   - **ArchiveManager**: Log archiving operations, conflict resolution
   - **FileMonitor**: Log file detection, debouncing, callback execution

### Phase 3: Advanced Features
1. **Structured Log Analysis**
   - JSON log output format option
   - Log aggregation and filtering utilities
   - Performance bottleneck identification

2. **Audit Trail Generation**
   - Complete operation history tracking
   - Decision point logging (why worksheets accepted/rejected)
   - File lineage tracking

## Technical Details

### Logging Levels Strategy
- **DEBUG**: Detailed internal state, variable values
- **INFO**: Operation milestones, decisions, results
- **WARNING**: Recoverable issues, fallbacks used
- **ERROR**: Operation failures, exceptions
- **CRITICAL**: System-level failures

### Structured Logging Format
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "correlation_id": "123e4567-e89b-12d3-a456-426614174000",
  "logger": "excel_to_csv.converter",
  "function": "process_file",
  "line": 142,
  "message": "Worksheet analysis completed",
  "structured": {
    "operation": "analyze_worksheet",
    "worksheet_name": "Sheet1",
    "confidence_score": 0.85,
    "accepted": true,
    "criteria": {
      "data_density": 0.9,
      "header_quality": 0.8,
      "structure_score": 0.85
    }
  }
}
```

### Performance Considerations
- Lazy evaluation for expensive log data
- Configurable log levels to control verbosity
- Asynchronous logging for high-throughput scenarios
- Daily log rotation with automatic archival and compression
- Configurable log retention policies (default: keep 30 days)
- Archive cleanup for old compressed logs

### Integration Points
- **CLI**: Correlation ID generation per command execution
- **File Monitor**: Correlation ID per file event
- **Batch Processing**: Correlation ID per batch, sub-IDs per file
- **Configuration**: Logging level configuration via config files
- **Testing**: Mock loggers and metrics validation

## Log File Management

### Daily Rotation Strategy
- **Log Files**: Named `excel_to_csv_YYYYMMDD.log` for each day
- **Current Log**: Always the file for today's date
- **Rotation Trigger**: Automatic at midnight or first log entry of new day
- **Archive Location**: `logs/archive/` directory

### Archival Process
1. **Compression**: Previous day's log compressed to `.gz` format
2. **Naming**: Archived as `excel_to_csv_YYYYMMDD.gz`
3. **Storage**: Moved to archive subdirectory
4. **Cleanup**: Original uncompressed log file deleted after compression

### Retention Policy
- **Active Logs**: Current day's log file (uncompressed)
- **Recent Archives**: Last 7 days accessible in archive
- **Long-term Storage**: Up to 30 days by default (configurable)
- **Cleanup**: Automated deletion of logs older than retention period

### Directory Structure
```
logs/
├── excel_to_csv_20240109.log          # Current day
├── archive/
│   ├── excel_to_csv_20240108.gz       # Yesterday
│   ├── excel_to_csv_20240107.gz       # Day before
│   └── excel_to_csv_20240106.gz       # Etc.
```

## Benefits
1. **Complete Operation Visibility**: Every action and decision logged
2. **Performance Analysis**: Detailed timing and bottleneck identification
3. **Error Investigation**: Full context for failures and exceptions
4. **Audit Trail**: Complete history of file processing decisions
5. **Correlation Tracking**: Link related operations across components
6. **Structured Analysis**: Machine-readable logs for automated analysis
7. **Automatic Log Management**: Daily rotation and archival with compression
8. **Space Efficiency**: Compressed archives save disk space
9. **Historical Analysis**: Access to historical logs for trend analysis

This design provides a comprehensive logging enhancement that will enable detailed analysis of system behavior while maintaining performance and code clarity, with automatic log management to prevent disk space issues.