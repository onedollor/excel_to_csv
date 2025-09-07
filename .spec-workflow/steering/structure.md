# Project Structure

## Directory Organization

```
excel_to_csv/                   # Project root
├── src/                        # Source code (src layout)
│   └── excel_to_csv/          # Main package
│       ├── __init__.py        # Package initialization
│       ├── main.py            # Application entry point
│       ├── cli.py             # Command-line interface
│       ├── models/            # Data models and structures
│       │   ├── __init__.py
│       │   └── data_models.py # WorksheetData, Config, etc.
│       ├── config/            # Configuration management
│       │   ├── __init__.py
│       │   └── config_manager.py
│       ├── processors/        # Excel file processing
│       │   ├── __init__.py
│       │   └── excel_processor.py
│       ├── analysis/          # Confidence analysis
│       │   ├── __init__.py
│       │   └── confidence_analyzer.py
│       ├── generators/        # CSV output generation
│       │   ├── __init__.py
│       │   └── csv_generator.py
│       ├── monitoring/        # File system monitoring
│       │   ├── __init__.py
│       │   └── file_monitor.py
│       └── utils/             # Shared utilities
│           ├── __init__.py
│           └── logger.py
├── tests/                     # Test suite (mirrors src structure)
│   ├── __init__.py
│   ├── conftest.py           # pytest configuration and fixtures
│   ├── unit/                 # Unit tests
│   │   ├── models/
│   │   ├── config/
│   │   ├── processors/
│   │   ├── analysis/
│   │   ├── generators/
│   │   ├── monitoring/
│   │   └── utils/
│   ├── integration/          # Integration tests
│   │   └── test_end_to_end.py
│   ├── performance/          # Performance tests
│   │   └── test_performance.py
│   └── fixtures/            # Test data and sample files
│       ├── sample_excel_files/
│       └── expected_csv_outputs/
├── config/                  # Default configuration files
│   ├── default.yaml        # Default application config
│   └── logging.yaml        # Logging configuration
├── docs/                   # Documentation
│   ├── README.md          # Main documentation
│   ├── api/               # API documentation
│   └── user_guide/        # User guides and tutorials
├── examples/              # Usage examples
│   ├── sample_config.yaml # Example configuration
│   └── sample_usage.py   # Example Python usage
├── scripts/               # Development and deployment scripts
│   ├── setup_dev.sh      # Development environment setup
│   └── run_tests.sh      # Test execution script
├── pyproject.toml        # Python project configuration
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
└── README.md            # Project overview and quick start
```

## Naming Conventions

### Files
- **Modules**: `snake_case.py` (e.g., `excel_processor.py`, `confidence_analyzer.py`)
- **Packages**: `snake_case` directory names with `__init__.py`
- **Classes**: `PascalCase` within files (e.g., `ExcelProcessor`, `ConfidenceAnalyzer`)
- **Tests**: `test_[module_name].py` (e.g., `test_excel_processor.py`)
- **Configuration**: `lowercase.yaml` (e.g., `default.yaml`, `logging.yaml`)

### Code
- **Classes/Types**: `PascalCase` (e.g., `WorksheetData`, `ConfidenceScore`)
- **Functions/Methods**: `snake_case` (e.g., `process_excel_file()`, `calculate_confidence()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONFIDENCE_THRESHOLD`, `MAX_FILE_SIZE`)
- **Variables**: `snake_case` (e.g., `file_path`, `confidence_score`, `worksheet_data`)
- **Private members**: Leading underscore `_private_method()`, `_internal_variable`

## Import Patterns

### Import Order (following PEP 8)
1. **Standard library imports**
2. **Third-party library imports** (pandas, openpyxl, watchdog, etc.)
3. **Local application imports** (relative and absolute)

### Module Organization Example
```python
# Standard library imports
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Third-party imports
import pandas as pd
import yaml
from watchdog.observers import Observer

# Local imports
from excel_to_csv.models.data_models import WorksheetData, Config
from excel_to_csv.utils.logger import setup_logging
from .confidence_analyzer import ConfidenceAnalyzer  # Relative import within package
```

### Package Import Strategy
- **Absolute imports** from `excel_to_csv` package root for cross-module dependencies
- **Relative imports** within the same module/package for internal dependencies
- **Explicit imports** preferred over wildcard imports (`from module import *`)

## Code Structure Patterns

### Module/Class Organization
```python
# 1. Module docstring and metadata
"""Excel processor module for reading and analyzing Excel files."""

__version__ = "1.0.0"
__author__ = "Excel-to-CSV Converter Team"

# 2. Imports (standard, third-party, local)
import logging
import pandas as pd
from excel_to_csv.models.data_models import WorksheetData

# 3. Module-level constants
DEFAULT_CHUNK_SIZE = 10000
SUPPORTED_EXCEL_FORMATS = ['.xlsx', '.xls']

# 4. Type definitions and protocols
from typing import Protocol
class ExcelProcessorProtocol(Protocol):
    def process_file(self, path: Path) -> List[WorksheetData]: ...

# 5. Main class implementations
class ExcelProcessor:
    """Main Excel file processor implementation."""
    pass

# 6. Module-level utility functions
def _validate_excel_file(file_path: Path) -> bool:
    """Private helper function for file validation."""
    pass

# 7. Public API functions
def create_excel_processor(config: Config) -> ExcelProcessor:
    """Factory function for creating Excel processor instances."""
    pass
```

### Function/Method Organization
```python
def process_excel_file(self, file_path: Path) -> List[WorksheetData]:
    """Process Excel file and return worksheet data.
    
    Args:
        file_path: Path to the Excel file to process
        
    Returns:
        List of WorksheetData objects for each valid worksheet
        
    Raises:
        ExcelProcessingError: If file cannot be processed
    """
    # 1. Input validation and early returns
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    # 2. Setup and initialization
    logger = logging.getLogger(__name__)
    worksheets_data = []
    
    # 3. Core processing logic
    try:
        with pd.ExcelFile(file_path) as excel_file:
            for sheet_name in excel_file.sheet_names:
                worksheet_data = self._process_worksheet(excel_file, sheet_name)
                if worksheet_data:
                    worksheets_data.append(worksheet_data)
    except Exception as e:
        logger.error(f"Failed to process Excel file {file_path}: {e}")
        raise ExcelProcessingError(f"Processing failed: {e}") from e
    
    # 4. Result validation and return
    logger.info(f"Processed {len(worksheets_data)} worksheets from {file_path}")
    return worksheets_data
```

### File Organization Principles
- **Single Responsibility**: Each file contains related functionality (e.g., `excel_processor.py` only handles Excel operations)
- **Public API First**: Public classes and functions at the top, private helpers at the bottom
- **Clear Separation**: Separate concerns into different files (processing vs. analysis vs. configuration)

## Code Organization Principles

1. **Single Responsibility**: Each module handles one aspect of the system (e.g., file monitoring, Excel processing, confidence analysis)
2. **Modularity**: Components can be imported and tested independently
3. **Testability**: Each module has corresponding test files with comprehensive coverage
4. **Consistency**: All modules follow the same organizational patterns and naming conventions
5. **Documentation**: Every public class and function has docstrings with type hints

## Module Boundaries

### Dependency Direction
```
CLI Layer (cli.py, main.py)
    ↓
Business Logic Layer (processors/, analysis/, generators/)
    ↓
Data Layer (models/, config/)
    ↓
Utility Layer (utils/, monitoring/)
```

### Boundary Rules
- **Core vs CLI**: Core business logic independent of CLI interface
- **Processing vs Configuration**: Processing modules receive configuration but don't load it
- **Analysis vs Generation**: Confidence analysis is separate from CSV generation
- **Monitoring vs Processing**: File monitoring triggers processing but doesn't perform it
- **Models vs Logic**: Data models contain no business logic, only structure and validation

### Inter-module Communication
- **Data Models**: Standardized data structures passed between modules
- **Configuration Objects**: Centralized configuration passed to modules that need it
- **Event-driven**: File monitor publishes events, processors consume them
- **Interface Contracts**: Clear method signatures with type hints

## Code Size Guidelines

- **File size**: Maximum 500 lines per file (excluding tests)
- **Function/Method size**: Maximum 50 lines per function, ideally 20-30 lines
- **Class complexity**: Maximum 10 public methods per class
- **Nesting depth**: Maximum 4 levels of nesting (use early returns and guard clauses)
- **Function parameters**: Maximum 5 parameters (use dataclasses for complex parameter sets)

## Testing Structure

### Test Organization
- **Mirror source structure**: Each source file has a corresponding test file
- **Test categories**: Unit tests (isolated), integration tests (multi-component), performance tests
- **Test fixtures**: Shared test data in `tests/fixtures/` with sample Excel files and expected outputs
- **Test configuration**: Central `conftest.py` with pytest fixtures and configuration

### Test File Naming
```python
# Source: src/excel_to_csv/processors/excel_processor.py
# Test:   tests/unit/processors/test_excel_processor.py

class TestExcelProcessor:
    """Test cases for ExcelProcessor class."""
    
    def test_process_valid_excel_file(self):
        """Test processing of valid Excel file."""
        pass
    
    def test_process_invalid_excel_file_raises_error(self):
        """Test error handling for invalid Excel files."""
        pass
```

## Documentation Standards

- **Module docstrings**: Every module starts with a docstring describing its purpose
- **Class docstrings**: Every class has a docstring with purpose and usage examples
- **Function docstrings**: Google-style docstrings with Args, Returns, and Raises sections
- **Type hints**: All public functions and methods have complete type annotations
- **README files**: Each major package has a README.md explaining its purpose and structure
- **API documentation**: Generated using Sphinx with automatic type hint extraction

### Documentation Example
```python
class ConfidenceAnalyzer:
    """Analyzes Excel worksheets to determine data table confidence.
    
    The ConfidenceAnalyzer uses multiple algorithms to determine whether
    an Excel worksheet contains a meaningful data table:
    - Data density analysis
    - Header detection
    - Column consistency analysis
    
    Example:
        >>> analyzer = ConfidenceAnalyzer(threshold=0.9)
        >>> score = analyzer.analyze_worksheet(worksheet_data)
        >>> if score.overall_score >= analyzer.threshold:
        ...     print("Worksheet contains a data table")
    
    Attributes:
        threshold: Minimum confidence score to consider worksheet valid
        algorithms: List of analysis algorithms to apply
    """
    
    def analyze_worksheet(self, data: WorksheetData) -> ConfidenceScore:
        """Analyze worksheet data and return confidence score.
        
        Args:
            data: WorksheetData object containing Excel worksheet information
            
        Returns:
            ConfidenceScore object with overall score and component scores
            
        Raises:
            AnalysisError: If worksheet data is invalid or analysis fails
        """
        pass
```