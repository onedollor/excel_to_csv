#!/usr/bin/env python3
"""Sample usage examples for Excel-to-CSV Converter.

This script demonstrates various ways to use the Excel-to-CSV converter
both programmatically and through the command-line interface.

Run this script to see example outputs:
    python examples/sample_usage.py
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import subprocess
from datetime import datetime

# Add the src directory to Python path for imports
script_dir = Path(__file__).parent.parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from excel_to_csv import (
        ExcelToCSVConverter,
        ExcelProcessor,
        ConfidenceAnalyzer,
        CSVGenerator,
        Config,
        WorksheetData,
        OutputConfig
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Excel-to-CSV modules: {e}")
    print("This example will only show CLI usage patterns.")
    IMPORTS_AVAILABLE = False


def create_sample_excel_file(file_path: Path):
    """Create a sample Excel file for demonstration."""
    # Sample data that should meet the confidence threshold
    data = {
        'Employee_ID': range(1, 101),
        'Name': [f'Employee_{i:03d}' for i in range(1, 101)],
        'Department': ['Engineering', 'Sales', 'Marketing', 'HR'] * 25,
        'Salary': [50000 + (i * 1000) for i in range(100)],
        'Start_Date': pd.date_range('2020-01-01', periods=100, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(file_path) as writer:
        # Main data sheet (should pass confidence test)
        df.to_excel(writer, sheet_name='Employee_Data', index=False)
        
        # Summary sheet (may or may not pass confidence test)
        summary = pd.DataFrame({
            'Department': ['Engineering', 'Sales', 'Marketing', 'HR'],
            'Count': [25, 25, 25, 25],
            'Avg_Salary': [75000, 65000, 60000, 55000]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sparse sheet (should fail confidence test)
        sparse = pd.DataFrame(index=range(20), columns=range(10))
        sparse.iloc[0, 0] = "Title"
        sparse.iloc[2, 1] = "Some data"
        sparse.to_excel(writer, sheet_name='Notes', index=False)
    
    return file_path


def example_1_basic_file_processing():
    """Example 1: Basic file processing with the API."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic File Processing")
    print("="*80)
    
    if not IMPORTS_AVAILABLE:
        print("Skipping API example - imports not available")
        return
    
    # Create temporary directories and sample file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "sample_data.xlsx"
        output_dir = temp_path / "output"
        output_dir.mkdir()
        
        # Create sample Excel file
        create_sample_excel_file(input_file)
        print(f"✓ Created sample Excel file: {input_file}")
        
        # Configure the converter
        config = Config(
            monitored_folders=[temp_path],
            output_folder=output_dir,
            confidence_threshold=0.8,  # Lower threshold for demo
            max_concurrent=1
        )
        
        # Initialize converter
        converter = ExcelToCSVConverter()
        converter.config = config
        
        print(f"✓ Initialized converter with config")
        print(f"  - Input file: {input_file}")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Confidence threshold: {config.confidence_threshold}")
        
        # Process the file
        print("\n📊 Processing Excel file...")
        result = converter.process_single_file(input_file)
        
        # Display results
        print(f"\n📋 Processing Results:")
        print(f"  - Success: {result.success}")
        print(f"  - CSV files generated: {len(result.csv_files)}")
        print(f"  - Processing time: {result.processing_duration:.2f} seconds")
        
        if result.csv_files:
            print(f"\n📂 Generated CSV files:")
            for csv_file in result.csv_files:
                file_size = csv_file.stat().st_size / 1024  # KB
                print(f"  - {csv_file.name} ({file_size:.1f} KB)")
                
                # Show first few rows of CSV
                try:
                    df = pd.read_csv(csv_file)
                    print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"    Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"    Error reading CSV: {e}")
        
        if result.errors:
            print(f"\n❌ Errors encountered:")
            for error in result.errors:
                print(f"  - {error}")


def example_2_confidence_analysis():
    """Example 2: Detailed confidence analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Confidence Analysis Deep Dive")
    print("="*80)
    
    if not IMPORTS_AVAILABLE:
        print("Skipping API example - imports not available")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "analysis_sample.xlsx"
        
        # Create sample file
        create_sample_excel_file(input_file)
        
        # Process Excel file to get worksheets
        processor = ExcelProcessor()
        worksheets = processor.process_excel_file(input_file)
        
        print(f"✓ Found {len(worksheets)} worksheets in {input_file.name}")
        
        # Analyze each worksheet
        analyzer = ConfidenceAnalyzer(threshold=0.9)
        
        print(f"\n🔍 Confidence Analysis (threshold: 0.9):")
        print("-" * 60)
        
        for worksheet in worksheets:
            confidence = analyzer.analyze_worksheet(worksheet)
            
            print(f"\n📄 Worksheet: {worksheet.name}")
            print(f"  Shape: {worksheet.row_count} rows × {worksheet.column_count} columns")
            print(f"  Overall Score: {confidence.overall_score:.3f} ({'✓ PASS' if confidence.overall_score >= 0.9 else '✗ FAIL'})")
            print(f"  Components:")
            print(f"    - Data Density:    {confidence.data_density:.3f} (weight: 40%)")
            print(f"    - Header Quality:  {confidence.header_quality:.3f} (weight: 30%)")  
            print(f"    - Consistency:     {confidence.consistency_score:.3f} (weight: 30%)")
            
            if confidence.reasons:
                print(f"  Decision Factors:")
                for reason in confidence.reasons:
                    print(f"    - {reason}")
            
            # Show data preview
            if not worksheet.data.empty:
                print(f"  Data Preview:")
                print(f"    Columns: {list(worksheet.data.columns)[:5]}...")  # First 5 columns
                non_null_counts = worksheet.data.count()
                print(f"    Non-null counts: {dict(non_null_counts.head())}")


def example_3_custom_configuration():
    """Example 3: Custom configuration and CSV generation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Configuration and CSV Generation")
    print("="*80)
    
    if not IMPORTS_AVAILABLE:
        print("Skipping API example - imports not available")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_file = temp_path / "custom_sample.xlsx"
        output_dir = temp_path / "custom_output"
        output_dir.mkdir()
        
        create_sample_excel_file(input_file)
        
        # Custom output configuration
        output_config = OutputConfig(
            folder=str(output_dir),
            naming_pattern="converted_{timestamp}_{filename}_{worksheet}.csv",
            include_timestamp=True,
            encoding="utf-8"
        )
        
        # Custom confidence weights
        custom_weights = {
            'data_density': 0.5,      # Higher weight on data density
            'header_quality': 0.3,
            'consistency': 0.2        # Lower weight on consistency
        }
        
        print("🛠️ Custom Configuration:")
        print(f"  - Output pattern: {output_config.naming_pattern}")
        print(f"  - Include timestamp: {output_config.include_timestamp}")
        print(f"  - Encoding: {output_config.encoding}")
        print(f"  - Custom weights: {custom_weights}")
        
        # Process with custom configuration
        processor = ExcelProcessor()
        worksheets = processor.process_excel_file(input_file)
        
        analyzer = ConfidenceAnalyzer(threshold=0.7, weights=custom_weights)
        generator = CSVGenerator(output_folder=output_dir, config=output_config)
        
        print(f"\n🔄 Processing with custom settings...")
        
        for worksheet in worksheets:
            confidence = analyzer.analyze_worksheet(worksheet)
            print(f"\n📄 {worksheet.name}: confidence = {confidence.overall_score:.3f}")
            
            if confidence.overall_score >= 0.7:  # Custom threshold
                csv_file = generator.generate_csv(worksheet)
                print(f"  ✓ Generated: {csv_file.name}")
                
                # Verify timestamp in filename
                if "converted_" in csv_file.name:
                    print(f"  ✓ Timestamp included in filename")
            else:
                print(f"  ✗ Skipped (below threshold)")


def example_4_cli_usage():
    """Example 4: Command-line interface usage."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Command-Line Interface Usage")
    print("="*80)
    
    print("Here are common CLI usage patterns:\n")
    
    cli_examples = [
        {
            "title": "Process a single Excel file",
            "command": "excel-to-csv process data/input.xlsx --output data/converted/",
            "description": "Converts worksheets from input.xlsx to CSV files in the converted directory"
        },
        {
            "title": "Preview worksheets before processing",
            "command": "excel-to-csv preview data/input.xlsx",
            "description": "Shows confidence scores for each worksheet without creating CSV files"
        },
        {
            "title": "Start service mode with configuration",
            "command": "excel-to-csv service --config config/production.yaml",
            "description": "Runs continuous monitoring using settings from production.yaml"
        },
        {
            "title": "Service mode with specific folders",
            "command": "excel-to-csv service --folders /data/input /data/archive --output /data/csv",
            "description": "Monitors multiple input folders and outputs CSV files to a specific directory"
        },
        {
            "title": "Check configuration validity",
            "command": "excel-to-csv config-check --config config/my_settings.yaml",
            "description": "Validates the configuration file for syntax and logical errors"
        },
        {
            "title": "Process with custom confidence threshold",
            "command": "excel-to-csv process file.xlsx --confidence-threshold 0.8",
            "description": "Processes with a lower confidence threshold (80% instead of default 90%)"
        },
        {
            "title": "View processing statistics",
            "command": "excel-to-csv stats",
            "description": "Shows current processing statistics and performance metrics"
        },
        {
            "title": "Enable debug logging",
            "command": "excel-to-csv process file.xlsx --log-level DEBUG",
            "description": "Runs with detailed debug output for troubleshooting"
        },
        {
            "title": "Background service (Linux/macOS)",
            "command": "nohup excel-to-csv service --config production.yaml > service.log 2>&1 &",
            "description": "Runs service in background, logging to service.log file"
        }
    ]
    
    for i, example in enumerate(cli_examples, 1):
        print(f"{i:2d}. {example['title']}")
        print(f"    Command: {example['command']}")
        print(f"    Purpose: {example['description']}\n")
    
    # Try to demonstrate actual CLI if available
    try:
        # Check if CLI is available
        result = subprocess.run([
            sys.executable, "-c", 
            "from excel_to_csv.cli import main; print('CLI available')"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("💡 The CLI commands above are available in your environment!")
            print("   Try running: python -m excel_to_csv.cli --help")
        else:
            print("💡 CLI commands shown above (install package to use them)")
            
    except Exception:
        print("💡 CLI commands shown above (install package to use them)")


def example_5_service_mode_simulation():
    """Example 5: Service mode simulation."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Service Mode Simulation")
    print("="*80)
    
    print("Service mode continuously monitors folders for new Excel files.")
    print("Here's how it works:\n")
    
    print("🔄 Service Mode Workflow:")
    print("1. Monitor specified folders for Excel files")
    print("2. Detect new files using file system events")
    print("3. Process files automatically when they appear")
    print("4. Generate CSV files for qualifying worksheets")
    print("5. Log all activities for monitoring")
    print("6. Handle errors gracefully and continue running")
    
    print(f"\n⚙️ Configuration for service mode:")
    print("```yaml")
    print("monitoring:")
    print("  folders:")
    print("    - '/data/excel_input'")
    print("  process_existing: true")
    print("output:")
    print("  folder: '/data/csv_output'")
    print("processing:")
    print("  max_concurrent: 4")
    print("```")
    
    print(f"\n🚀 Starting service mode:")
    print("$ excel-to-csv service --config config.yaml")
    print("Service starting...")
    print("Monitoring folders: ['/data/excel_input']")
    print("Output directory: /data/csv_output")
    print("Press Ctrl+C to stop")
    
    print(f"\n📂 What happens when files are added:")
    print("- File detected: /data/excel_input/report.xlsx")
    print("- Processing started...")
    print("- Worksheet 'Data' analyzed: confidence = 0.95 ✓")
    print("- Generated: /data/csv_output/report_Data.csv")
    print("- Worksheet 'Notes' analyzed: confidence = 0.45 ✗")
    print("- Processing complete: 1 CSV file generated")
    
    print(f"\n🛑 Graceful shutdown:")
    print("- Ctrl+C pressed")
    print("- Stopping file monitoring...")
    print("- Waiting for active processing to complete...")
    print("- Service stopped gracefully")


def example_6_error_handling():
    """Example 6: Error handling and troubleshooting."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Error Handling and Troubleshooting")
    print("="*80)
    
    print("Common error scenarios and how they're handled:\n")
    
    error_scenarios = [
        {
            "error": "File is locked or in use",
            "cause": "Excel file is open in another application",
            "handling": "Automatic retry after delay, or skip with warning",
            "user_action": "Close the Excel file or wait for auto-retry"
        },
        {
            "error": "Memory error during processing",
            "cause": "Excel file is too large for available RAM",
            "handling": "Graceful failure with detailed error message",
            "user_action": "Reduce file size limit or increase system memory"
        },
        {
            "error": "Permission denied writing CSV",
            "cause": "Insufficient permissions for output directory",
            "handling": "Clear error message with suggested solutions",
            "user_action": "Fix directory permissions or change output location"
        },
        {
            "error": "Corrupt Excel file",
            "cause": "File is damaged or not a valid Excel format",
            "handling": "Skip file and log detailed error information",
            "user_action": "Repair the Excel file or check file format"
        },
        {
            "error": "No worksheets meet confidence threshold",
            "cause": "Worksheets don't contain clear tabular data",
            "handling": "Log analysis results and skip conversion",
            "user_action": "Review confidence scores or lower threshold"
        }
    ]
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"{i}. {scenario['error']}")
        print(f"   Cause: {scenario['cause']}")
        print(f"   Handling: {scenario['handling']}")
        print(f"   User Action: {scenario['user_action']}\n")
    
    print("🔧 Debugging Tools:")
    print("- Enable debug logging: --log-level DEBUG")
    print("- Preview mode: excel-to-csv preview file.xlsx")
    print("- Configuration check: excel-to-csv config-check")
    print("- Processing statistics: excel-to-csv stats")
    
    print(f"\n📊 Log Analysis Examples:")
    print("# View recent errors in JSON logs")
    print("tail -f logs/excel_to_csv.log | jq 'select(.level == \"ERROR\")'")
    print()
    print("# Count successful vs failed processing")
    print("grep 'processing_complete' logs/excel_to_csv.log | jq '.success' | sort | uniq -c")


def main():
    """Run all examples."""
    print("Excel-to-CSV Converter - Usage Examples")
    print("=" * 80)
    print("This script demonstrates various usage patterns and features.")
    print(f"Python version: {sys.version}")
    print(f"Script location: {Path(__file__).absolute()}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if IMPORTS_AVAILABLE:
        print("✓ Excel-to-CSV modules successfully imported")
    else:
        print("⚠️ Excel-to-CSV modules not available (showing CLI examples only)")
    
    # Run examples
    try:
        example_1_basic_file_processing()
        example_2_confidence_analysis()
        example_3_custom_configuration()
        example_4_cli_usage()
        example_5_service_mode_simulation()
        example_6_error_handling()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext Steps:")
        print("1. Copy examples/sample_config.yaml and customize for your needs")
        print("2. Test with your Excel files using: excel-to-csv preview yourfile.xlsx")
        print("3. Start with CLI mode before deploying service mode")
        print("4. Monitor logs and adjust configuration as needed")
        print("\nFor more information, see docs/README.md")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())