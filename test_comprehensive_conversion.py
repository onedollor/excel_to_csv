#!/usr/bin/env python3
"""
Comprehensive test script for Excel to CSV conversion with enhanced logging.

This script tests the complete Excel to CSV conversion system using the 
enhanced logging infrastructure with all test files created.
"""

import sys
import logging
import tempfile
from pathlib import Path
import pandas as pd
import traceback
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from excel_to_csv.utils.logging_config import setup_enhanced_logging
from excel_to_csv.utils.correlation import CorrelationContext
from excel_to_csv.utils.logging_decorators import operation_context
from excel_to_csv.processors.excel_processor import ExcelProcessor
from excel_to_csv.generators.csv_generator import CSVGenerator
from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
from excel_to_csv.archiving.archive_manager import ArchiveManager
from excel_to_csv.models.data_models import (
    OutputConfig, ArchiveConfig, RetryConfig, Config
)


class ComprehensiveTestRunner:
    """Test runner for comprehensive Excel to CSV conversion testing."""
    
    def __init__(self, log_dir: Path):
        """Initialize test runner with logging setup."""
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
        # Initialize components
        self.excel_processor = None
        self.csv_generator = None
        self.confidence_analyzer = None
        self.archive_manager = None
        
        # Test files
        self.test_files = {
            'comprehensive': Path('/tmp/comprehensive_test_file.xlsx'),
            'empty': Path('/tmp/empty_test.xlsx'),
            'formulas': Path('/tmp/formulas_test.xlsx'),
            'problematic': Path('/tmp/problematic_test.xlsx')
        }
    
    def setup_components(self):
        """Setup all conversion components with configurations."""
        self.logger.info("Setting up conversion components...")
        
        # Retry config for archive manager
        retry_config = RetryConfig(
            max_attempts=2,
            delay=1.0,
            backoff_factor=2.0,
            max_delay=5.0
        )
        
        # Initialize components with minimal configuration
        self.excel_processor = ExcelProcessor()
        self.csv_generator = CSVGenerator()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.archive_manager = ArchiveManager(retry_config)
        
        self.logger.info("All components initialized successfully")
    
    def test_single_file(self, file_name: str, file_path: Path) -> dict:
        """Test conversion of a single Excel file."""
        with operation_context(
            "comprehensive_file_test",
            self.logger,
            file_name=file_name,
            file_path=str(file_path),
            file_size=file_path.stat().st_size if file_path.exists() else 0
        ) as metrics:
            
            test_result = {
                'file_name': file_name,
                'file_path': str(file_path),
                'success': False,
                'worksheets_processed': 0,
                'worksheets_converted': 0,
                'errors': [],
                'warnings': [],
                'output_files': [],
                'processing_time': 0,
                'confidence_scores': {}
            }
            
            if not file_path.exists():
                error_msg = f"Test file not found: {file_path}"
                test_result['errors'].append(error_msg)
                self.logger.error(error_msg)
                return test_result
            
            try:
                # Step 1: Process Excel file
                self.logger.info(f"Processing Excel file: {file_name}")
                worksheets = self.excel_processor.process_file(file_path)
                
                metrics.add_metadata("excel_worksheets", len(worksheets))
                test_result['worksheets_processed'] = len(worksheets)
                
                if not worksheets:
                    warning_msg = f"No worksheets found or processed in {file_name}"
                    test_result['warnings'].append(warning_msg)
                    self.logger.warning(warning_msg)
                    return test_result
                
                # Step 2: Analyze each worksheet and convert if suitable
                output_dir = self.log_dir / 'output' / file_name.replace('.xlsx', '')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for worksheet in worksheets:
                    worksheet_name = worksheet.worksheet_name
                    
                    self.logger.info(f"Analyzing worksheet: {worksheet_name}")
                    
                    # Confidence analysis
                    try:
                        confidence_result = self.confidence_analyzer.analyze_worksheet(worksheet)
                        
                        confidence_score = confidence_result.overall_confidence
                        test_result['confidence_scores'][worksheet_name] = confidence_score
                        
                        self.logger.info(
                            f"Confidence analysis for '{worksheet_name}': {confidence_score:.2f}",
                            extra={
                                "structured": {
                                    "operation": "worksheet_confidence_analysis",
                                    "worksheet_name": worksheet_name,
                                    "confidence_score": confidence_score,
                                    "is_suitable": confidence_result.is_suitable_for_csv,
                                    "data_rows": confidence_result.analysis_metrics.get('data_rows', 0),
                                    "data_columns": confidence_result.analysis_metrics.get('data_columns', 0)
                                }
                            }
                        )
                        
                        # Step 3: Convert suitable worksheets
                        if confidence_result.is_suitable_for_csv:
                            self.logger.info(f"Converting worksheet '{worksheet_name}' to CSV...")
                            
                            # Generate CSV
                            csv_result = self.csv_generator.generate_csv(
                                worksheet,
                                output_dir,
                                OutputConfig()
                            )
                            
                            if csv_result.success:
                                test_result['worksheets_converted'] += 1
                                test_result['output_files'].append(str(csv_result.output_path))
                                
                                self.logger.info(
                                    f"Successfully converted '{worksheet_name}' to {csv_result.output_path.name}",
                                    extra={
                                        "structured": {
                                            "operation": "csv_conversion_success",
                                            "worksheet_name": worksheet_name,
                                            "output_file": str(csv_result.output_path),
                                            "rows_written": csv_result.rows_written,
                                            "columns_written": csv_result.columns_written,
                                            "file_size": csv_result.output_path.stat().st_size
                                        }
                                    }
                                )
                            else:
                                error_msg = f"Failed to convert '{worksheet_name}': {csv_result.error_message}"
                                test_result['errors'].append(error_msg)
                                self.logger.error(error_msg)
                        else:
                            warning_msg = f"Worksheet '{worksheet_name}' not suitable for CSV conversion (confidence: {confidence_score:.2f})"
                            test_result['warnings'].append(warning_msg)
                            self.logger.warning(warning_msg)
                            
                    except Exception as e:
                        error_msg = f"Error processing worksheet '{worksheet_name}': {str(e)}"
                        test_result['errors'].append(error_msg)
                        self.logger.error(error_msg, exc_info=True)
                
                test_result['success'] = True
                metrics.add_metadata("worksheets_converted", test_result['worksheets_converted'])
                metrics.add_metadata("conversion_success", True)
                
            except Exception as e:
                error_msg = f"Fatal error processing {file_name}: {str(e)}"
                test_result['errors'].append(error_msg)
                self.logger.error(error_msg, exc_info=True)
                metrics.add_metadata("conversion_success", False)
                metrics.add_metadata("error_type", type(e).__name__)
            
            return test_result
    
    def run_all_tests(self):
        """Run tests on all test files."""
        with operation_context("comprehensive_test_suite", self.logger) as metrics:
            
            self.logger.info("🧪 Starting comprehensive Excel to CSV conversion tests")
            self.setup_components()
            
            # Test each file
            for file_name, file_path in self.test_files.items():
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Testing file: {file_name} ({file_path})")
                self.logger.info(f"{'='*60}")
                
                result = self.test_single_file(file_name, file_path)
                self.results[file_name] = result
            
            # Generate summary report
            self.generate_summary_report()
            
            metrics.add_metadata("total_files_tested", len(self.test_files))
            metrics.add_metadata("successful_files", sum(1 for r in self.results.values() if r['success']))
            
    def generate_summary_report(self):
        """Generate and log a comprehensive summary report."""
        self.logger.info("\n" + "🎯 COMPREHENSIVE TEST SUMMARY REPORT")
        self.logger.info("=" * 80)
        
        total_files = len(self.results)
        successful_files = sum(1 for r in self.results.values() if r['success'])
        total_worksheets = sum(r['worksheets_processed'] for r in self.results.values())
        total_converted = sum(r['worksheets_converted'] for r in self.results.values())
        total_output_files = sum(len(r['output_files']) for r in self.results.values())
        
        summary_stats = {
            "operation": "test_suite_summary",
            "total_files_tested": total_files,
            "successful_files": successful_files,
            "total_worksheets_processed": total_worksheets,
            "total_worksheets_converted": total_converted,
            "total_csv_files_generated": total_output_files,
            "success_rate": f"{(successful_files/total_files)*100:.1f}%" if total_files > 0 else "0%"
        }
        
        self.logger.info(
            f"📊 Overall Results: {successful_files}/{total_files} files processed successfully",
            extra={"structured": summary_stats}
        )
        
        # Detailed per-file results
        for file_name, result in self.results.items():
            file_summary = {
                "operation": "file_test_summary",
                "file_name": file_name,
                "success": result['success'],
                "worksheets_processed": result['worksheets_processed'],
                "worksheets_converted": result['worksheets_converted'],
                "errors_count": len(result['errors']),
                "warnings_count": len(result['warnings']),
                "output_files_count": len(result['output_files'])
            }
            
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            self.logger.info(
                f"{status} {file_name}: {result['worksheets_converted']}/{result['worksheets_processed']} worksheets converted",
                extra={"structured": file_summary}
            )
            
            # Log confidence scores
            if result['confidence_scores']:
                self.logger.info(f"  Confidence scores:")
                for worksheet, score in result['confidence_scores'].items():
                    suitability = "✅ Suitable" if score >= 0.3 else "❌ Unsuitable"
                    self.logger.info(f"    {worksheet}: {score:.2f} ({suitability})")
            
            # Log errors and warnings
            if result['errors']:
                self.logger.info(f"  ❌ Errors ({len(result['errors'])}):")
                for error in result['errors']:
                    self.logger.info(f"    - {error}")
            
            if result['warnings']:
                self.logger.info(f"  ⚠️  Warnings ({len(result['warnings'])}):")
                for warning in result['warnings']:
                    self.logger.info(f"    - {warning}")
            
            if result['output_files']:
                self.logger.info(f"  📁 Output files ({len(result['output_files'])}):")
                for output_file in result['output_files']:
                    output_path = Path(output_file)
                    if output_path.exists():
                        size = output_path.stat().st_size
                        self.logger.info(f"    - {output_path.name} ({size:,} bytes)")
                    else:
                        self.logger.info(f"    - {output_path.name} (file not found)")
        
        self.logger.info("=" * 80)
        self.logger.info("🎉 Comprehensive testing completed!")


def main():
    """Main function to run comprehensive tests."""
    
    # Setup logging
    log_dir = Path("/tmp/excel_csv_test_logs")
    log_dir.mkdir(exist_ok=True)
    
    setup_enhanced_logging(
        log_level="DEBUG",
        log_dir=log_dir,
        structured_format=True,
        daily_rotation=False,  # Disable for testing
        console_output=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting comprehensive Excel to CSV conversion testing")
    
    # Ensure correlation context
    correlation_id = CorrelationContext.ensure_correlation_id()
    logger.info(f"Test correlation ID: {correlation_id}")
    
    try:
        # Create and run test suite
        test_runner = ComprehensiveTestRunner(log_dir)
        test_runner.run_all_tests()
        
        print(f"\n✅ Testing completed! Check logs in: {log_dir}")
        print(f"   Log files: {list(log_dir.glob('*.log'))}")
        print(f"   Output files: {log_dir / 'output'}")
        
    except Exception as e:
        logger.error(f"Fatal error in test runner: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())