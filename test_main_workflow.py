#!/usr/bin/env python3
"""
End-to-end integration test for the complete Excel to CSV workflow.

This script tests the complete workflow:
1. Drop Excel file (simulate main entry point)
2. Extract CSV files 
3. Archive results
4. Verify each step works correctly

Tests the main functions and integration points to identify any workflow issues.
"""

import sys
import logging
import tempfile
from pathlib import Path
import shutil
import json
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from excel_to_csv.utils.logging_config import setup_enhanced_logging
from excel_to_csv.utils.correlation import CorrelationContext
from excel_to_csv.utils.logging_decorators import operation_context
from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.models.data_models import (
    Config, OutputConfig, ArchiveConfig, RetryConfig
)


class WorkflowIntegrationTester:
    """End-to-end workflow integration tester."""
    
    def __init__(self, log_dir: Path):
        """Initialize the workflow tester."""
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.temp_dir = None
        
        # Test files to use (actual files in input directory)
        self.test_files = {
            'normal': Path('/home/lin/repo/excel_to_csv/input/normal_input.xlsx'),
            'formulas': Path('/home/lin/repo/excel_to_csv/input/formulas_input.xlsx'),
            'problematic': Path('/home/lin/repo/excel_to_csv/input/problematic_input.xlsx')
        }
    
    def setup_test_environment(self):
        """Setup standard input/output directories as requested."""
        self.logger.info("Setting up standard input/output environment...")
        
        # Use standard project directories as requested
        project_root = Path("/home/lin/repo/excel_to_csv")
        self.temp_dir = project_root
        
        # Use proper input/output structure
        self.input_dir = project_root / "input"
        self.output_dir = project_root / "output" 
        self.archive_dir = self.input_dir / "archive"  # Archive within input folder
        
        # Ensure directories exist
        for dir_path in [self.input_dir, self.output_dir, self.archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Using standard project directories:")
        self.logger.info(f"  Input: {self.input_dir}")
        self.logger.info(f"  Output: {self.output_dir}")
        self.logger.info(f"  Archive: {self.archive_dir}")
        
        return True
    
    def test_step_1_file_drop(self, test_file_key: str, test_file_path: Path) -> dict:
        """Test Step 1: Simulate dropping/providing an Excel file."""
        with operation_context(
            "workflow_step_1_file_drop", 
            self.logger,
            test_file=test_file_key,
            file_path=str(test_file_path)
        ) as metrics:
            
            step_result = {
                'step': 'file_drop',
                'success': False,
                'file_key': test_file_key,
                'file_path': str(test_file_path),
                'file_exists': False,
                'file_size': 0,
                'error': None
            }
            
            try:
                # Check if test file exists
                if not test_file_path.exists():
                    step_result['error'] = f"Test file not found: {test_file_path}"
                    self.logger.error(step_result['error'])
                    return step_result
                
                # Check if file is already in input directory
                input_file = self.input_dir / f"{test_file_key}_input.xlsx"
                
                if test_file_path == input_file:
                    # File is already in input directory, just verify it
                    self.logger.info(f"File '{test_file_key}' already in input directory")
                else:
                    # Copy test file to input directory (simulate file drop)
                    shutil.copy2(test_file_path, input_file)
                
                step_result['file_exists'] = True
                step_result['file_size'] = input_file.stat().st_size
                step_result['copied_to'] = str(input_file)
                step_result['success'] = True
                
                metrics.add_metadata("file_size", step_result['file_size'])
                metrics.add_metadata("input_file", str(input_file))
                
                self.logger.info(f"✅ Step 1 Success: File '{test_file_key}' copied to input directory ({step_result['file_size']:,} bytes)")
                
            except Exception as e:
                step_result['error'] = str(e)
                self.logger.error(f"❌ Step 1 Failed: {e}", exc_info=True)
                metrics.add_metadata("error", str(e))
            
            return step_result
    
    def test_step_2_csv_extraction(self, input_file_path: Path, test_file_key: str) -> dict:
        """Test Step 2: CSV extraction using individual components."""
        with operation_context(
            "workflow_step_2_csv_extraction",
            self.logger,
            input_file=str(input_file_path),
            test_file=test_file_key
        ) as metrics:
            
            step_result = {
                'step': 'csv_extraction',
                'success': False,
                'input_file': str(input_file_path),
                'output_files': [],
                'worksheets_processed': 0,
                'worksheets_converted': 0,
                'processing_time': 0,
                'error': None
            }
            
            try:
                # Use individual components like the comprehensive test
                from excel_to_csv.processors.excel_processor import ExcelProcessor
                from excel_to_csv.generators.csv_generator import CSVGenerator
                from excel_to_csv.analysis.confidence_analyzer import ConfidenceAnalyzer
                
                start_time = datetime.now()
                
                # Initialize components
                excel_processor = ExcelProcessor()
                csv_generator = CSVGenerator()
                confidence_analyzer = ConfidenceAnalyzer()
                
                # Step 1: Process Excel file
                self.logger.info(f"Processing Excel file: {input_file_path}")
                worksheets = excel_processor.process_file(input_file_path)
                
                step_result['worksheets_processed'] = len(worksheets)
                
                if not worksheets:
                    self.logger.warning(f"No worksheets found in {input_file_path}")
                    step_result['success'] = True  # Not an error, just empty
                    return step_result
                
                # Step 2: Analyze and convert suitable worksheets
                output_dir = self.output_dir
                
                for worksheet in worksheets:
                    worksheet_name = worksheet.worksheet_name
                    
                    # Confidence analysis
                    confidence_result = confidence_analyzer.analyze_worksheet(worksheet)
                    confidence_score = confidence_result.overall_score
                    
                    self.logger.info(f"Worksheet '{worksheet_name}': confidence {confidence_score:.2f}")
                    
                    # Convert if suitable
                    if confidence_result.is_confident:
                        self.logger.info(f"Converting worksheet '{worksheet_name}' to CSV...")
                        
                        # Generate CSV
                        output_config = OutputConfig()
                        output_config.folder = output_dir
                        output_config.include_timestamp = False  # Don't clutter with timestamps for testing
                        output_path = csv_generator.generate_csv(worksheet, output_config)
                        
                        step_result['worksheets_converted'] += 1
                        step_result['output_files'].append(str(output_path))
                        
                        self.logger.info(f"Successfully converted '{worksheet_name}' to {output_path.name}")
                    else:
                        self.logger.info(f"Worksheet '{worksheet_name}' not suitable for CSV (confidence: {confidence_score:.2f})")
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                step_result['processing_time'] = processing_time
                
                # Check actual output files created
                output_csv_files = list(output_dir.glob("*.csv"))
                step_result['actual_output_files'] = [str(f) for f in output_csv_files]
                
                step_result['success'] = len(output_csv_files) > 0
                
                metrics.add_metadata("worksheets_processed", step_result['worksheets_processed'])
                metrics.add_metadata("worksheets_converted", step_result['worksheets_converted'])
                metrics.add_metadata("output_files_count", len(step_result['actual_output_files']))
                metrics.add_metadata("processing_time", processing_time)
                
                if step_result['success']:
                    self.logger.info(f"✅ Step 2 Success: {len(step_result['actual_output_files'])} CSV files created in {processing_time:.2f}s")
                else:
                    self.logger.warning(f"⚠️ Step 2 Partial: Processing completed but no CSV files found")
                
            except Exception as e:
                step_result['error'] = str(e)
                self.logger.error(f"❌ Step 2 Failed: {e}", exc_info=True)
                metrics.add_metadata("error", str(e))
            
            return step_result
    
    def test_step_3_archiving(self, input_excel_file: Path, test_file_key: str) -> dict:
        """Test Step 3: Archive the input Excel file after processing."""
        with operation_context(
            "workflow_step_3_archiving", 
            self.logger,
            input_file=str(input_excel_file),
            test_file=test_file_key
        ) as metrics:
            
            step_result = {
                'step': 'archiving',
                'success': False,
                'input_file': str(input_excel_file),
                'archived_file': None,
                'original_size': 0,
                'archive_size': 0,
                'compression_ratio': 0,
                'error': None
            }
            
            try:
                if not input_excel_file.exists():
                    step_result['error'] = f"Input Excel file not found: {input_excel_file}"
                    self.logger.error(step_result['error'])
                    return step_result
                
                # Get original file size
                step_result['original_size'] = input_excel_file.stat().st_size
                
                # Create archive file in input/archive folder (no compression)
                import shutil as file_shutil
                
                archive_file_path = self.archive_dir / input_excel_file.name
                
                # Copy the Excel file to archive (no compression)
                file_shutil.copy2(input_excel_file, archive_file_path)
                
                step_result['archived_file'] = str(archive_file_path)
                step_result['archive_size'] = archive_file_path.stat().st_size
                
                # No compression, so ratio is 0
                step_result['compression_ratio'] = 0
                
                step_result['success'] = True
                
                metrics.add_metadata("original_size", step_result['original_size'])
                metrics.add_metadata("archive_size", step_result['archive_size'])
                metrics.add_metadata("compression_ratio", step_result['compression_ratio'])
                
                self.logger.info(f"✅ Step 3 Success: Excel file archived")
                self.logger.info(f"   Original: {input_excel_file.name} ({step_result['original_size']} bytes)")
                self.logger.info(f"   Archived: {archive_file_path.name} ({step_result['archive_size']} bytes)")
                self.logger.info(f"   Archive location: input/archive/{archive_file_path.name}")
                
                # Keep the original Excel file in input after archiving (user may want to reprocess)
                self.logger.info(f"   Original file kept in input: {input_excel_file}")
                
            except Exception as e:
                step_result['error'] = str(e)
                self.logger.error(f"❌ Step 3 Failed: {e}", exc_info=True)
                metrics.add_metadata("error", str(e))
            
            return step_result
    
    def test_step_4_verification(self, test_file_key: str, all_step_results: dict) -> dict:
        """Test Step 4: Verify the complete workflow worked correctly."""
        with operation_context(
            "workflow_step_4_verification",
            self.logger,
            test_file=test_file_key
        ) as metrics:
            
            step_result = {
                'step': 'verification',
                'success': False,
                'workflow_success': False,
                'issues': [],
                'summary': {},
                'error': None
            }
            
            try:
                issues = []
                
                # Check Step 1
                if not all_step_results['step1']['success']:
                    issues.append(f"Step 1 (File Drop) failed: {all_step_results['step1'].get('error', 'Unknown error')}")
                
                # Check Step 2 
                if not all_step_results['step2']['success']:
                    issues.append(f"Step 2 (CSV Extraction) failed: {all_step_results['step2'].get('error', 'Unknown error')}")
                elif len(all_step_results['step2']['actual_output_files']) == 0:
                    issues.append("Step 2 (CSV Extraction) produced no output files")
                
                # Check Step 3
                if not all_step_results['step3']['success'] and all_step_results['step2']['success']:
                    issues.append(f"Step 3 (Archiving) failed: {all_step_results['step3'].get('error', 'Unknown error')}")
                
                # Check file existence
                input_dir_files = list(self.input_dir.glob("*"))
                output_dir_files = list(self.output_dir.glob("*"))
                archive_dir_files = list(self.archive_dir.glob("*"))
                
                step_result['summary'] = {
                    'input_files': len(input_dir_files),
                    'output_files': len(output_dir_files),
                    'archive_files': len(archive_dir_files),
                    'total_processing_time': all_step_results['step2'].get('processing_time', 0),
                    'worksheets_processed': all_step_results['step2'].get('worksheets_processed', 0),
                    'worksheets_converted': all_step_results['step2'].get('worksheets_converted', 0)
                }
                
                step_result['issues'] = issues
                step_result['workflow_success'] = len(issues) == 0
                step_result['success'] = True  # Verification step itself succeeded
                
                metrics.add_metadata("workflow_success", step_result['workflow_success'])
                metrics.add_metadata("issues_count", len(issues))
                metrics.add_metadata("summary", step_result['summary'])
                
                if step_result['workflow_success']:
                    self.logger.info(f"✅ Step 4 Success: Complete workflow verified successfully")
                    self.logger.info(f"   Summary: {step_result['summary']['worksheets_converted']}/{step_result['summary']['worksheets_processed']} worksheets converted")
                else:
                    self.logger.error(f"❌ Step 4 Issues Found: {len(issues)} workflow issues detected")
                    for issue in issues:
                        self.logger.error(f"   - {issue}")
                
            except Exception as e:
                step_result['error'] = str(e)
                self.logger.error(f"❌ Step 4 Failed: {e}", exc_info=True)
                metrics.add_metadata("error", str(e))
            
            return step_result
    
    def test_single_file_workflow(self, test_file_key: str) -> dict:
        """Test complete workflow for a single file."""
        with operation_context(
            "complete_workflow_test",
            self.logger,
            test_file=test_file_key
        ) as metrics:
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🧪 Testing Complete Workflow: {test_file_key}")
            self.logger.info(f"{'='*60}")
            
            test_file_path = self.test_files[test_file_key]
            
            workflow_result = {
                'test_file_key': test_file_key,
                'test_file_path': str(test_file_path),
                'overall_success': False,
                'steps': {}
            }
            
            # Step 1: File Drop
            self.logger.info("🗂️ Step 1: Testing file drop simulation...")
            step1_result = self.test_step_1_file_drop(test_file_key, test_file_path)
            workflow_result['steps']['step1'] = step1_result
            
            if not step1_result['success']:
                self.logger.error(f"Workflow stopped: Step 1 failed for {test_file_key}")
                return workflow_result
            
            # Step 2: CSV Extraction
            self.logger.info("📊 Step 2: Testing CSV extraction...")
            input_file = Path(step1_result['copied_to'])
            step2_result = self.test_step_2_csv_extraction(input_file, test_file_key)
            workflow_result['steps']['step2'] = step2_result
            
            # Step 3: Archiving the input Excel file after processing
            self.logger.info("🗄️ Step 3: Testing archiving...")
            input_excel_path = self.input_dir / test_file_path.name
            step3_result = self.test_step_3_archiving(input_excel_path, test_file_key)
            workflow_result['steps']['step3'] = step3_result
            
            # Step 4: Verification
            self.logger.info("✅ Step 4: Testing workflow verification...")
            step4_result = self.test_step_4_verification(test_file_key, workflow_result['steps'])
            workflow_result['steps']['step4'] = step4_result
            
            workflow_result['overall_success'] = step4_result['workflow_success']
            
            metrics.add_metadata("overall_success", workflow_result['overall_success'])
            metrics.add_metadata("steps_completed", len(workflow_result['steps']))
            
            return workflow_result
    
    def run_all_workflow_tests(self):
        """Run complete workflow tests on all test files."""
        with operation_context("all_workflow_tests", self.logger) as metrics:
            
            self.logger.info("🚀 Starting End-to-End Workflow Integration Tests")
            
            # Setup test environment
            if not self.setup_test_environment():
                self.logger.error("Failed to setup test environment")
                return
            
            # Test each file
            for test_file_key in self.test_files:
                try:
                    result = self.test_single_file_workflow(test_file_key)
                    self.test_results[test_file_key] = result
                except Exception as e:
                    self.logger.error(f"Fatal error testing {test_file_key}: {e}", exc_info=True)
                    self.test_results[test_file_key] = {
                        'test_file_key': test_file_key,
                        'overall_success': False,
                        'fatal_error': str(e)
                    }
            
            # Generate final report
            self.generate_workflow_report()
            
            metrics.add_metadata("total_tests", len(self.test_files))
            metrics.add_metadata("successful_workflows", sum(1 for r in self.test_results.values() if r.get('overall_success', False)))
    
    def generate_workflow_report(self):
        """Generate comprehensive workflow test report."""
        self.logger.info(f"\n{'🎯 WORKFLOW INTEGRATION TEST REPORT':=^80}")
        
        total_tests = len(self.test_results)
        successful_workflows = sum(1 for r in self.test_results.values() if r.get('overall_success', False))
        
        summary_stats = {
            "operation": "workflow_test_summary",
            "total_workflows_tested": total_tests,
            "successful_workflows": successful_workflows,
            "success_rate": f"{(successful_workflows/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
        }
        
        self.logger.info(
            f"📊 Overall Results: {successful_workflows}/{total_tests} workflows completed successfully",
            extra={"structured": summary_stats}
        )
        
        # Detailed results for each test file
        for test_key, result in self.test_results.items():
            status = "✅ SUCCESS" if result.get('overall_success', False) else "❌ FAILED"
            self.logger.info(f"\n{status} {test_key.upper()} WORKFLOW:")
            
            if 'fatal_error' in result:
                self.logger.info(f"   💀 Fatal Error: {result['fatal_error']}")
                continue
            
            steps = result.get('steps', {})
            for step_name, step_data in steps.items():
                step_status = "✅" if step_data.get('success', False) else "❌"
                step_label = step_data.get('step', step_name).replace('_', ' ').title()
                self.logger.info(f"   {step_status} {step_label}")
                
                if step_data.get('error'):
                    self.logger.info(f"      Error: {step_data['error']}")
                
                # Show key metrics for each step
                if step_name == 'step2' and step_data.get('success'):
                    self.logger.info(f"      Processed: {step_data.get('worksheets_processed', 0)} worksheets")
                    self.logger.info(f"      Converted: {step_data.get('worksheets_converted', 0)} to CSV")
                    self.logger.info(f"      Time: {step_data.get('processing_time', 0):.2f}s")
                
                if step_name == 'step3' and step_data.get('success'):
                    archived_file = step_data.get('archived_file')
                    if archived_file:
                        self.logger.info(f"      Archived: Excel file (no compression)")
                        self.logger.info(f"      Location: input/archive/{Path(archived_file).name}")
            
            # Show verification summary
            if 'step4' in steps and 'summary' in steps['step4']:
                summary = steps['step4']['summary']
                self.logger.info(f"   📈 Summary: {summary.get('output_files', 0)} CSV files, {summary.get('archive_files', 0)} archived")
                
                if steps['step4'].get('issues'):
                    self.logger.info("   ⚠️ Issues Found:")
                    for issue in steps['step4']['issues']:
                        self.logger.info(f"      - {issue}")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🎉 Workflow Integration Testing Completed!")
        self.logger.info(f"   Test Environment: {self.temp_dir}")
        self.logger.info(f"   📁 Check results in:")
        self.logger.info(f"      Input files: input/")
        self.logger.info(f"      Output CSV files: output/") 
        self.logger.info(f"      Archived files: archive/")
    
    def cleanup(self):
        """Cleanup test environment."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Could not cleanup test directory: {e}")


def main():
    """Main function to run workflow integration tests."""
    
    # Setup logging in project directory
    log_dir = Path("/home/lin/repo/excel_to_csv/logs")
    log_dir.mkdir(exist_ok=True)
    
    setup_enhanced_logging(
        log_level="DEBUG",
        log_dir=log_dir,
        structured_format=True,
        daily_rotation=False,
        console_output=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Excel to CSV Workflow Integration Testing")
    
    # Ensure correlation context
    correlation_id = CorrelationContext.ensure_correlation_id()
    logger.info(f"Test correlation ID: {correlation_id}")
    
    try:
        # Create and run workflow tests
        tester = WorkflowIntegrationTester(log_dir)
        tester.run_all_workflow_tests()
        
        print(f"\n✅ Workflow testing completed! Check logs in: {log_dir}")
        print(f"   Log files: {list(log_dir.glob('*.log'))}")
        print(f"   Test environment preserved at: {tester.temp_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in workflow tester: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup if desired (commented out to preserve test environment for inspection)
        # if 'tester' in locals():
        #     tester.cleanup()
        pass


if __name__ == "__main__":
    sys.exit(main())