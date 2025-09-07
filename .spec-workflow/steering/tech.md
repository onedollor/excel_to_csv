# Technology Stack

## Project Type
Command-line automation tool with optional web dashboard for monitoring. The project is designed as a standalone Python application that provides intelligent Excel file processing with configurable automation capabilities.

## Core Technologies

### Primary Language(s)
- **Language**: Python 3.11+
- **Runtime**: CPython with virtual environment isolation
- **Language-specific tools**: pip for package management, pyproject.toml for modern Python project configuration

### Key Dependencies/Libraries
- **pandas (>= 2.0.0)**: Primary library for Excel reading and CSV writing operations with robust data type handling
- **openpyxl (>= 3.1.0)**: Advanced Excel file manipulation and metadata extraction for confidence analysis  
- **watchdog (>= 3.0.0)**: Cross-platform file system event monitoring for real-time directory watching
- **pyyaml (>= 6.0)**: Configuration file parsing with support for complex nested structures
- **click (>= 8.1.0)**: Command-line interface framework with argument validation and help generation
- **pytest (>= 7.0.0)**: Testing framework with fixtures and parametrization support
- **pytest-cov (>= 4.0.0)**: Coverage testing integration with HTML reporting capabilities

### Application Architecture
Event-driven pipeline architecture with modular component design:
- **Event-Driven Core**: File system events trigger processing workflows
- **Pipeline Pattern**: Sequential processing stages (Monitor → Process → Analyze → Convert → Output)
- **Component Isolation**: Independent modules communicating through well-defined data structures
- **Configuration-Driven**: Runtime behavior controlled by YAML configuration files

### Data Storage
- **Primary storage**: File system for Excel inputs and CSV outputs
- **Configuration storage**: YAML files for application settings and user preferences
- **Logging storage**: Structured log files with JSON formatting for integration with log aggregators
- **Data formats**: CSV output, JSON structured logs, YAML configuration files

### External Integrations
- **File System APIs**: Native OS file system monitoring and I/O operations
- **Excel Format Support**: Microsoft Office Excel (.xlsx) and legacy Excel (.xls) file formats
- **Network Drives**: SMB/CIFS network drive support for enterprise environments

## Development Environment

### Build & Development Tools
- **Build System**: setuptools with pyproject.toml configuration for modern Python packaging
- **Package Management**: pip with requirements.txt for development dependencies and pip-tools for dependency management
- **Development workflow**: Python virtual environments with hot-reload capability during development

### Code Quality Tools
- **Static Analysis**: 
  - mypy for type checking and static analysis
  - ruff for fast linting and code formatting (replaces flake8, isort, black)
  - pylint for additional code quality checks
- **Formatting**: ruff format for consistent code style enforcement
- **Testing Framework**: 
  - pytest for unit, integration, and end-to-end testing
  - pytest-cov for coverage reporting with 90% minimum threshold
  - pytest-mock for test isolation and mocking
- **Documentation**: Sphinx for API documentation generation with NumPy-style docstrings

### Version Control & Collaboration
- **VCS**: Git with conventional commit messages for automated changelog generation
- **Branching Strategy**: GitHub Flow with feature branches and pull request reviews
- **Code Review Process**: Required PR reviews with automated CI/CD checks before merge

## Deployment & Distribution

### Target Platform(s)
- **Primary**: Linux servers and workstations (Ubuntu 20.04+, RHEL 8+)
- **Secondary**: Windows 10+ and macOS 12+ for desktop deployments
- **Deployment**: Standalone Python application with systemd service support on Linux

### Distribution Method
- **Package Distribution**: Python wheel packages via PyPI for easy installation
- **Container Support**: Docker images for containerized deployments
- **Installation Requirements**: Python 3.11+, 2GB RAM minimum, file system access permissions

### Update Mechanism
- **Version Management**: Semantic versioning with automated dependency updates
- **Update Delivery**: pip-based updates from PyPI with configuration migration support

## Technical Requirements & Constraints

### Performance Requirements
- **Processing Speed**: <30 seconds for Excel files up to 50MB with complex worksheets
- **Memory Usage**: <1GB peak memory usage during processing of large files
- **Concurrent Processing**: Support for 5+ simultaneous file processing operations
- **Startup Time**: <5 seconds from command invocation to monitoring startup

### Compatibility Requirements  
- **Platform Support**: Linux (primary), Windows 10+, macOS 12+ (cross-platform testing required)
- **Python Versions**: Python 3.11+ with backward compatibility testing to 3.9
- **Excel Format Support**: Microsoft Excel 2007+ (.xlsx), legacy Excel (.xls), OpenDocument (.ods) formats
- **File System**: POSIX and Windows file systems, network drives (SMB/CIFS)

### Security & Compliance
- **File Access Security**: Principle of least privilege for file system access with permission validation
- **Input Validation**: Sanitization of all file paths and configuration inputs to prevent path traversal attacks
- **Error Handling**: No sensitive information exposure in error messages or logs
- **Data Privacy**: No data retention beyond processing duration, configurable log retention policies

### Scalability & Reliability
- **Expected Load**: 1000+ Excel files per day per instance with configurable processing queues
- **Availability Requirements**: 99.5% uptime with automatic recovery from transient failures
- **Growth Projections**: Horizontal scaling through multiple process instances with shared configuration

## Technical Decisions & Rationale

### Decision Log
1. **Python 3.11+ Requirement**: Chosen for modern type hints, improved performance, and excellent data processing ecosystem (pandas/openpyxl). Alternative considered: Go (rejected due to limited Excel processing libraries).

2. **pandas + openpyxl Combination**: pandas provides robust CSV writing and data manipulation, openpyxl enables advanced Excel metadata analysis. Alternative considered: xlrd/xlwt (rejected due to limited .xlsx support and maintenance status).

3. **watchdog for File Monitoring**: Cross-platform file system monitoring with reliable event handling. Alternative considered: inotify (Linux-only), polling (high resource usage).

4. **YAML Configuration Format**: Human-readable configuration with support for complex nested structures and comments. Alternative considered: JSON (no comments), TOML (less familiar to ops teams).

5. **Event-Driven Architecture**: Enables real-time processing and modular component design. Alternative considered: Batch processing (less responsive), cron-based (less flexible).

6. **pytest Testing Framework**: Mature Python testing ecosystem with excellent fixture support and plugin ecosystem. Alternative considered: unittest (more verbose), nose2 (less active development).

## Known Limitations

- **Excel Password Protection**: Cannot process password-protected Excel files - requires manual password removal or pre-processing
- **Large File Memory Usage**: Files >100MB may require significant memory allocation - consider streaming processing for future versions  
- **Network Drive Reliability**: Network connectivity issues may impact monitoring reliability - implement exponential backoff retry mechanisms
- **Excel Macro Content**: VBA macros and embedded objects are ignored during processing - document this limitation for users with macro-dependent workflows
- **Concurrent File Access**: Cannot process files that are currently open in Excel - implement file locking detection and retry logic