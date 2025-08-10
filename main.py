#!/usr/bin/env python3
"""
Research Paper Tagger - Command Line Interface
Automatically assigns category tags to research papers based on their content.
"""
import sys
import logging
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.paper_tagger import PaperTagger
from src.output_handlers import OutputHandler
from src.config_loader import ConfigLoader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Research Paper Tagger - Automatically categorize research papers."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), 
              help='Custom configuration file')
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output directory')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'json', 'both']), 
              default='csv', help='Output format')
@click.option('--threshold', '-t', type=float, default=0.3, 
              help='Confidence threshold for tag assignment (0.0-1.0)')
@click.option('--max-tags', type=int, default=5, 
              help='Maximum number of tags per paper')
@click.option('--extraction-method', type=click.Choice(['auto', 'pymupdf', 'pdfplumber', 'pypdf2']), 
              default='auto', help='PDF text extraction method')
@click.pass_context
def tag_single(ctx, pdf_path, config, output, output_format, threshold, max_tags, extraction_method):
    """Tag a single PDF file."""
    try:
        # Load configuration
        config_loader = ConfigLoader(config)
        
        # Override with command line options
        config_loader.set('processing.confidence_threshold', threshold)
        config_loader.set('processing.max_tags', max_tags)
        config_loader.set('processing.extraction_method', extraction_method)
        
        # Initialize tagger
        tagger = PaperTagger(
            config_path=config,
            confidence_threshold=threshold,
            max_tags=max_tags
        )
        
        # Initialize output handler
        output_handler = OutputHandler(output)
        
        click.echo(f"Processing: {pdf_path.name}")
        
        # Tag the paper
        result = tagger.tag_paper(pdf_path, extraction_method)
        
        if result.success:
            click.echo(f"✓ Successfully tagged with {len(result.tags)} tags:")
            for tag, confidence in result.tags:
                click.echo(f"  - {tag} (confidence: {confidence:.3f})")
            
            # Export results
            if output_format in ['csv', 'both']:
                csv_path = output_handler.export_to_csv([result])
                click.echo(f"✓ Results exported to: {csv_path}")
            
            if output_format in ['json', 'both']:
                json_path = output_handler.export_to_json([result])
                click.echo(f"✓ Results exported to: {json_path}")
        else:
            click.echo(f"✗ Failed to process: {result.error_message}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pdf_directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path), 
              help='Custom configuration file')
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output directory')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'json', 'both']), 
              default='csv', help='Output format')
@click.option('--threshold', '-t', type=float, default=0.3, 
              help='Confidence threshold for tag assignment (0.0-1.0)')
@click.option('--max-tags', type=int, default=5, 
              help='Maximum number of tags per paper')
@click.option('--extraction-method', type=click.Choice(['auto', 'pymupdf', 'pdfplumber', 'pypdf2']), 
              default='auto', help='PDF text extraction method')
@click.option('--pattern', default='*.pdf', help='File pattern to match')
@click.option('--organize', is_flag=True, help='Organize files by tags')
@click.option('--rename', is_flag=True, help='Rename files with tags')
@click.option('--summary', is_flag=True, default=True, help='Create summary report')
@click.pass_context
def tag_batch(ctx, pdf_directory, config, output, output_format, threshold, max_tags, 
              extraction_method, pattern, organize, rename, summary):
    """Tag multiple PDF files in a directory."""
    try:
        # Load configuration
        config_loader = ConfigLoader(config)
        
        # Override with command line options
        config_loader.set('processing.confidence_threshold', threshold)
        config_loader.set('processing.max_tags', max_tags)
        config_loader.set('processing.extraction_method', extraction_method)
        
        # Initialize tagger
        tagger = PaperTagger(
            config_path=config,
            confidence_threshold=threshold,
            max_tags=max_tags
        )
        
        # Initialize output handler
        output_handler = OutputHandler(output)
        
        click.echo(f"Processing PDFs in: {pdf_directory}")
        click.echo(f"Pattern: {pattern}")
        
        # Find PDF files
        pdf_files = list(pdf_directory.glob(pattern))
        if not pdf_files:
            click.echo(f"No PDF files found matching pattern: {pattern}")
            return
        
        click.echo(f"Found {len(pdf_files)} PDF files")
        
        # Process files with progress bar
        results = []
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for pdf_file in pdf_files:
                result = tagger.tag_paper(pdf_file, extraction_method)
                results.append(result)
                
                if result.success:
                    pbar.set_postfix(tags=len(result.tags))
                else:
                    pbar.set_postfix(status="FAILED")
                
                pbar.update(1)
        
        # Summary statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        click.echo(f"\nProcessing complete:")
        click.echo(f"  ✓ Successful: {successful}")
        click.echo(f"  ✗ Failed: {failed}")
        click.echo(f"  Success rate: {successful/len(results)*100:.1f}%")
        
        if successful > 0:
            # Export results
            if output_format in ['csv', 'both']:
                csv_path = output_handler.export_to_csv(results)
                click.echo(f"✓ CSV results exported to: {csv_path}")
            
            if output_format in ['json', 'both']:
                json_path = output_handler.export_to_json(results)
                click.echo(f"✓ JSON results exported to: {json_path}")
            
            # Create summary report
            if summary:
                summary_path = output_handler.create_tag_summary(results)
                click.echo(f"✓ Summary report created: {summary_path}")
            
            # Additional operations
            if organize:
                tag_to_files = output_handler.organize_papers_by_tags(results)
                click.echo(f"✓ Organized files into {len(tag_to_files)} tag directories")
            
            if rename:
                renamed_files = output_handler.rename_files_with_tags(results)
                click.echo(f"✓ Renamed {len(renamed_files)} files with tags")
            
            # Show tag statistics
            tag_stats = tagger.get_tag_statistics(results)
            if tag_stats:
                click.echo(f"\nTop tags:")
                for i, (tag, count) in enumerate(list(tag_stats.items())[:10], 1):
                    click.echo(f"  {i:2}. {tag:<30} ({count} papers)")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output file path')
def create_config(output):
    """Create a sample configuration file."""
    try:
        if not output:
            output = Path("custom_config.yaml")
        
        # Load default config and save it
        config_loader = ConfigLoader()
        config_loader.save_config(output)
        
        click.echo(f"✓ Sample configuration created: {output}")
        click.echo("Edit this file to customize categories and settings.")
        
    except Exception as e:
        click.echo(f"Error creating config: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
def validate_config(config_path):
    """Validate a configuration file."""
    try:
        config_loader = ConfigLoader(config_path)
        
        if config_loader.validate_config():
            click.echo("✓ Configuration is valid")
        else:
            click.echo("✗ Configuration has errors", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error validating config: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_categories():
    """List available categories and their keywords."""
    try:
        config_loader = ConfigLoader()
        tagger = PaperTagger()
        
        categories = tagger.get_available_categories()
        
        click.echo("Available categories:")
        click.echo("=" * 50)
        
        for category in sorted(categories):
            keywords = tagger.classifier.get_category_keywords(category)
            click.echo(f"\n{category}:")
            click.echo(f"  Keywords: {', '.join(keywords[:10])}...")
            if len(keywords) > 10:
                click.echo(f"  ({len(keywords)} total keywords)")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--method', type=click.Choice(['pymupdf', 'pdfplumber', 'pypdf2']), 
              default='pymupdf', help='Extraction method to test')
def test_extraction(pdf_path, method):
    """Test PDF text extraction on a single file."""
    try:
        from src.pdf_extractor import PDFExtractor
        
        extractor = PDFExtractor()
        
        click.echo(f"Testing extraction from: {pdf_path.name}")
        click.echo(f"Method: {method}")
        
        sections = extractor.extract_text(pdf_path, method)
        
        click.echo("\nExtracted sections:")
        for section, content in sections.items():
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                click.echo(f"\n{section.upper()}:")
                click.echo(f"  Length: {len(content)} characters")
                click.echo(f"  Preview: {preview}")
            else:
                click.echo(f"\n{section.upper()}: (empty)")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
