"""Command-line interface for Excel-to-CSV converter.

This module provides a comprehensive CLI with support for:
- Service mode for continuous monitoring
- Single file processing
- Configuration management
- Statistics and monitoring
"""

import sys
from pathlib import Path
from typing import Optional

import click

from excel_to_csv.excel_to_csv_converter import ExcelToCSVConverter
from excel_to_csv.config.config_manager import config_manager


@click.group(invoke_without_command=True)
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--version', is_flag=True, help='Show version information')
@click.pass_context
def main(ctx: click.Context, config: Optional[str], version: bool) -> None:
    """Excel-to-CSV Converter - Intelligent automation for Excel to CSV conversion.
    
    Monitors directories for Excel files and converts worksheets to CSV when
    90% confident they contain data tables.
    """
    if version:
        click.echo("Excel-to-CSV Converter v1.0.0")
        return
    
    # Store config path in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    
    # If no command specified, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.pass_context
def service(ctx: click.Context) -> None:
    """Run in service mode - continuous monitoring and processing.
    
    Monitors configured directories for new Excel files and automatically
    processes them when detected. Runs until stopped with Ctrl+C.
    """
    config_path = ctx.obj.get('config_path')
    
    try:
        click.echo("Starting Excel-to-CSV Converter Service...")
        click.echo("Press Ctrl+C to stop")
        click.echo()
        
        with ExcelToCSVConverter(config_path) as converter:
            converter.run_service()
            
    except KeyboardInterrupt:
        click.echo("\nService stopped by user")
    except Exception as e:
        click.echo(f"Service error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Output directory')
@click.pass_context
def process(ctx: click.Context, file_path: Path, output: Optional[Path]) -> None:
    """Process a single Excel file.
    
    Analyzes the specified Excel file and converts qualifying worksheets
    to CSV files based on confidence analysis.
    
    FILE_PATH: Path to the Excel file to process
    """
    config_path = ctx.obj.get('config_path')
    
    try:
        click.echo(f"Processing file: {file_path}")
        
        converter = ExcelToCSVConverter(config_path)
        
        # Override output folder if specified
        if output:
            converter.config.output_config.folder = output
            click.echo(f"Output directory: {output}")
        
        success = converter.process_file(file_path)
        
        if success:
            stats = converter.get_statistics()
            click.echo(f"\nProcessing completed successfully!")
            click.echo(f"Worksheets analyzed: {stats['worksheets_analyzed']}")
            click.echo(f"Worksheets accepted: {stats['worksheets_accepted']}")
            click.echo(f"CSV files generated: {stats['csv_files_generated']}")
            
            if stats.get('acceptance_rate'):
                click.echo(f"Acceptance rate: {stats['acceptance_rate']:.1f}%")
        else:
            click.echo("Processing failed!", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Processing error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def config_check(ctx: click.Context) -> None:
    """Validate and display current configuration."""
    config_path = ctx.obj.get('config_path')
    
    try:
        click.echo("Loading and validating configuration...")
        
        config = config_manager.load_config(config_path)
        
        click.echo("✓ Configuration loaded successfully")
        click.echo()
        click.echo("Configuration Summary:")
        click.echo(f"  Monitored folders: {len(config.monitored_folders)}")
        for folder in config.monitored_folders:
            status = "✓" if folder.exists() else "✗"
            click.echo(f"    {status} {folder}")
        
        click.echo(f"  File patterns: {', '.join(config.file_patterns)}")
        click.echo(f"  Confidence threshold: {config.confidence_threshold}")
        click.echo(f"  Max concurrent: {config.max_concurrent}")
        click.echo(f"  Max file size: {config.max_file_size_mb}MB")
        click.echo(f"  Output folder: {config.output_folder or 'Same as source'}")
        click.echo(f"  Logging level: {config.logging.level}")
        
    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--max-rows', default=10, help='Maximum rows to show in preview')
@click.pass_context
def preview(ctx: click.Context, file_path: Path, max_rows: int) -> None:
    """Preview Excel file analysis without generating CSV.
    
    Shows confidence analysis results for each worksheet in the file
    without actually generating CSV output.
    
    FILE_PATH: Path to the Excel file to preview
    """
    config_path = ctx.obj.get('config_path')
    
    try:
        click.echo(f"Analyzing file: {file_path}")
        click.echo()
        
        converter = ExcelToCSVConverter(config_path)
        
        # Process file to get worksheet analysis
        worksheets = converter.excel_processor.process_file(file_path)
        
        if not worksheets:
            click.echo("No worksheets found in file")
            return
        
        click.echo(f"Found {len(worksheets)} worksheets:")
        click.echo()
        
        for i, worksheet in enumerate(worksheets, 1):
            confidence_score = converter.confidence_analyzer.analyze_worksheet(worksheet)
            
            click.echo(f"{i}. Worksheet: '{worksheet.worksheet_name}'")
            click.echo(f"   Size: {worksheet.row_count} rows × {worksheet.column_count} columns")
            click.echo(f"   Data density: {worksheet.data_density:.3f}")
            click.echo(f"   Confidence: {confidence_score.overall_score:.3f}")
            click.echo(f"   Decision: {'ACCEPT' if confidence_score.is_confident else 'REJECT'}")
            
            if confidence_score.reasons:
                click.echo("   Reasons:")
                for reason in confidence_score.reasons[:3]:  # Show top 3 reasons
                    click.echo(f"     - {reason}")
            
            # Show data preview for accepted worksheets
            if confidence_score.is_confident:
                click.echo("   Preview:")
                preview_data = worksheet.data.head(min(max_rows, 5))
                click.echo(f"     {preview_data.to_string(max_rows=5, max_cols=6)}")
            
            click.echo()
        
        accepted_count = sum(1 for ws in worksheets 
                           if converter.confidence_analyzer.analyze_worksheet(ws).is_confident)
        
        click.echo(f"Summary: {accepted_count}/{len(worksheets)} worksheets would be converted")
        
    except Exception as e:
        click.echo(f"Preview error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--watch', is_flag=True, help='Continuously monitor and display stats')
@click.option('--interval', default=30, help='Update interval in seconds (with --watch)')
@click.pass_context
def stats(ctx: click.Context, watch: bool, interval: int) -> None:
    """Display processing statistics.
    
    Shows current processing statistics. Use --watch to continuously
    monitor statistics in real-time.
    """
    config_path = ctx.obj.get('config_path')
    
    try:
        converter = ExcelToCSVConverter(config_path)
        
        if not watch:
            # Show current stats once
            _display_stats(converter)
        else:
            # Continuous monitoring mode
            import time
            click.echo("Monitoring statistics (Press Ctrl+C to stop)")
            click.echo()
            
            try:
                while True:
                    click.clear()
                    _display_stats(converter)
                    time.sleep(interval)
            except KeyboardInterrupt:
                click.echo("\nMonitoring stopped")
                
    except Exception as e:
        click.echo(f"Stats error: {e}", err=True)
        sys.exit(1)


def _display_stats(converter: ExcelToCSVConverter) -> None:
    """Display formatted statistics."""
    stats = converter.get_statistics()
    
    click.echo("=== Processing Statistics ===")
    click.echo(f"Files processed: {stats['files_processed']}")
    click.echo(f"Files failed: {stats['files_failed']}")
    click.echo(f"Worksheets analyzed: {stats['worksheets_analyzed']}")
    click.echo(f"Worksheets accepted: {stats['worksheets_accepted']}")
    click.echo(f"CSV files generated: {stats['csv_files_generated']}")
    click.echo(f"Processing errors: {stats['processing_errors']}")
    
    if stats.get('acceptance_rate'):
        click.echo(f"Acceptance rate: {stats['acceptance_rate']:.1f}%")
    
    click.echo(f"Service status: {'Running' if stats['is_running'] else 'Stopped'}")
    
    if 'monitor' in stats:
        monitor = stats['monitor']
        click.echo(f"Monitored folders: {monitor['folders_count']}")
        click.echo(f"Files in queue: {monitor.get('pending_files', 0)}")
    
    if stats['failed_files']:
        click.echo(f"Failed files: {len(stats['failed_files'])}")


if __name__ == '__main__':
    main()