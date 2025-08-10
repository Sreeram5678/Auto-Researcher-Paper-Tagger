"""
Output handlers for saving and organizing tagged papers.
"""
import csv
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

import pandas as pd

from .paper_tagger import PaperResult


class OutputHandler:
    """Handle various output formats for tagged papers."""
    
    def __init__(self, output_directory: Optional[Path] = None):
        """
        Initialize output handler.
        
        Args:
            output_directory: Directory for output files
        """
        self.logger = logging.getLogger(__name__)
        self.output_directory = output_directory or Path("output")
        self.output_directory.mkdir(exist_ok=True)
    
    def export_to_csv(self, results: List[PaperResult], 
                     filename: Optional[str] = None) -> Path:
        """
        Export tagging results to CSV file.
        
        Args:
            results: List of paper results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the created CSV file
        """
        if not filename:
            # If single paper, use PDF name; otherwise use timestamp
            if len(results) == 1 and results[0].success:
                pdf_name = results[0].file_path.stem  # Get filename without extension
                # Clean the name for filesystem
                clean_name = "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_name = clean_name.replace(' ', '_')
                filename = f"{clean_name}_tags.csv"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"paper_tags_{timestamp}.csv"
        
        csv_path = self.output_directory / filename
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {
                'filename': result.filename,
                'file_path': str(result.file_path),
                'success': result.success,
                'processing_time': result.processing_time,
                'num_tags': len(result.tags) if result.success else 0,
                'tags': '; '.join([tag for tag, _ in result.tags]) if result.success else '',
                'tag_scores': '; '.join([f"{tag}:{score:.3f}" for tag, score in result.tags]) if result.success else '',
                'error_message': result.error_message or '',
                'title': result.metadata.get('title', '') if result.success else '',
                'author': result.metadata.get('author', '') if result.success else '',
                'abstract_length': len(result.text_sections.get('abstract', '')) if result.success else 0,
            }
            
            # Add individual tag columns
            for i, (tag, score) in enumerate(result.tags[:5]):  # Top 5 tags
                row[f'tag_{i+1}'] = tag
                row[f'tag_{i+1}_score'] = score
            
            csv_data.append(row)
        
        # Write to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Exported {len(results)} results to {csv_path}")
        else:
            self.logger.warning("No data to export")
        
        return csv_path
    
    def export_to_json(self, results: List[PaperResult], 
                      filename: Optional[str] = None) -> Path:
        """
        Export tagging results to JSON file.
        
        Args:
            results: List of paper results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the created JSON file
        """
        if not filename:
            # If single paper, use PDF name; otherwise use timestamp
            if len(results) == 1 and results[0].success:
                pdf_name = results[0].file_path.stem  # Get filename without extension
                # Clean the name for filesystem
                clean_name = "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_name = clean_name.replace(' ', '_')
                filename = f"{clean_name}_tags.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"paper_tags_{timestamp}.json"
        
        json_path = self.output_directory / filename
        
        # Prepare data for JSON
        json_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_papers': len(results),
            'successful_papers': sum(1 for r in results if r.success),
            'papers': []
        }
        
        for result in results:
            paper_data = {
                'filename': result.filename,
                'file_path': str(result.file_path),
                'success': result.success,
                'processing_time': result.processing_time,
                'tags': [{'name': tag, 'confidence': score} for tag, score in result.tags],
                'metadata': result.metadata,
                'error_message': result.error_message,
                'text_sections': {
                    'abstract_length': len(result.text_sections.get('abstract', '')),
                    'keywords': result.text_sections.get('keywords', ''),
                    'has_introduction': bool(result.text_sections.get('introduction')),
                    'has_conclusion': bool(result.text_sections.get('conclusion'))
                } if result.success else {}
            }
            json_data['papers'].append(paper_data)
        
        # Write to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported {len(results)} results to {json_path}")
        return json_path
    
    def create_tag_summary(self, results: List[PaperResult], 
                          filename: Optional[str] = None) -> Path:
        """
        Create a summary report of tag statistics.
        
        Args:
            results: List of paper results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the created summary file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tag_summary_{timestamp}.txt"
        
        summary_path = self.output_directory / filename
        
        # Calculate statistics
        total_papers = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_papers - successful
        
        tag_counts = {}
        tag_scores = {}
        
        for result in results:
            if result.success:
                for tag, score in result.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    if tag not in tag_scores:
                        tag_scores[tag] = []
                    tag_scores[tag].append(score)
        
        # Calculate average scores
        tag_avg_scores = {tag: sum(scores) / len(scores) for tag, scores in tag_scores.items()}
        
        # Generate summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("RESEARCH PAPER TAGGING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total papers processed: {total_papers}\n")
            f.write(f"Successfully tagged: {successful}\n")
            f.write(f"Failed to process: {failed}\n")
            f.write(f"Success rate: {successful/total_papers*100:.1f}%\n\n")
            
            if successful > 0:
                avg_tags = sum(len(r.tags) for r in results if r.success) / successful
                f.write(f"Average tags per paper: {avg_tags:.2f}\n")
                f.write(f"Total unique tags: {len(tag_counts)}\n\n")
                
                f.write("TAG FREQUENCY DISTRIBUTION:\n")
                f.write("-" * 30 + "\n")
                sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
                for tag, count in sorted_tags:
                    avg_score = tag_avg_scores[tag]
                    f.write(f"{tag:<30} {count:>3} papers ({avg_score:.3f} avg score)\n")
                
                f.write("\nMOST COMMON TAGS:\n")
                f.write("-" * 20 + "\n")
                for i, (tag, count) in enumerate(sorted_tags[:10], 1):
                    f.write(f"{i:>2}. {tag} ({count} papers)\n")
        
        self.logger.info(f"Created tag summary at {summary_path}")
        return summary_path
    
    def organize_papers_by_tags(self, results: List[PaperResult], 
                               copy_files: bool = True,
                               create_symlinks: bool = False) -> Dict[str, List[Path]]:
        """
        Organize papers into directories based on their tags.
        
        Args:
            results: List of paper results
            copy_files: Whether to copy files (True) or move them (False)
            create_symlinks: Create symbolic links instead of copying
            
        Returns:
            Dictionary mapping tags to lists of organized files
        """
        organized_dir = self.output_directory / "organized_by_tags"
        organized_dir.mkdir(exist_ok=True)
        
        tag_to_files = {}
        
        for result in results:
            if not result.success or not result.tags:
                continue
            
            # Use the top tag for organization
            primary_tag = result.tags[0][0]
            
            # Clean tag name for directory
            clean_tag = "".join(c for c in primary_tag if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_tag = clean_tag.replace(' ', '_')
            
            tag_dir = organized_dir / clean_tag
            tag_dir.mkdir(exist_ok=True)
            
            # Determine destination filename
            dest_file = tag_dir / result.filename
            
            try:
                if create_symlinks:
                    if not dest_file.exists():
                        dest_file.symlink_to(result.file_path.resolve())
                elif copy_files:
                    shutil.copy2(result.file_path, dest_file)
                else:
                    shutil.move(str(result.file_path), str(dest_file))
                
                if primary_tag not in tag_to_files:
                    tag_to_files[primary_tag] = []
                tag_to_files[primary_tag].append(dest_file)
                
            except Exception as e:
                self.logger.error(f"Failed to organize {result.filename}: {e}")
        
        self.logger.info(f"Organized {sum(len(files) for files in tag_to_files.values())} files into {len(tag_to_files)} tag directories")
        return tag_to_files
    
    def rename_files_with_tags(self, results: List[PaperResult], 
                              max_tags: int = 2,
                              separator: str = "_") -> List[Path]:
        """
        Rename files to include their top tags in the filename.
        
        Args:
            results: List of paper results
            max_tags: Maximum number of tags to include in filename
            separator: Separator between tags and original filename
            
        Returns:
            List of new file paths
        """
        renamed_files = []
        
        for result in results:
            if not result.success or not result.tags:
                continue
            
            # Get top tags
            top_tags = result.tags[:max_tags]
            
            # Clean tag names for filename
            clean_tags = []
            for tag, _ in top_tags:
                clean_tag = "".join(c for c in tag if c.isalnum() or c in (' ', '-')).strip()
                clean_tag = clean_tag.replace(' ', '-')
                clean_tags.append(clean_tag)
            
            # Create new filename
            original_stem = result.file_path.stem
            original_suffix = result.file_path.suffix
            tag_string = separator.join(clean_tags)
            
            new_filename = f"[{tag_string}]{separator}{original_stem}{original_suffix}"
            new_path = result.file_path.parent / new_filename
            
            try:
                result.file_path.rename(new_path)
                renamed_files.append(new_path)
                self.logger.info(f"Renamed {result.filename} to {new_filename}")
            except Exception as e:
                self.logger.error(f"Failed to rename {result.filename}: {e}")
        
        return renamed_files
    
    def create_bibliography(self, results: List[PaperResult], 
                           format_style: str = "apa",
                           filename: Optional[str] = None) -> Path:
        """
        Create a bibliography file from the processed papers.
        
        Args:
            results: List of paper results
            format_style: Bibliography format ("apa", "mla", "chicago")
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to the created bibliography file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bibliography_{timestamp}.txt"
        
        bib_path = self.output_directory / filename
        
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(f"BIBLIOGRAPHY - {format_style.upper()} FORMAT\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                if not result.success:
                    continue
                
                title = result.metadata.get('title', result.filename)
                author = result.metadata.get('author', 'Unknown Author')
                
                # Basic citation format
                if format_style.lower() == "apa":
                    citation = f"{author}. {title}. Retrieved from {result.file_path.name}"
                elif format_style.lower() == "mla":
                    citation = f"{author}. \"{title}.\" {result.file_path.name}."
                else:  # Chicago
                    citation = f"{author}. \"{title}.\" {result.file_path.name}."
                
                f.write(f"{i}. {citation}\n")
                
                # Add tags as keywords
                if result.tags:
                    tags_str = ", ".join([tag for tag, _ in result.tags])
                    f.write(f"   Keywords: {tags_str}\n")
                
                f.write("\n")
        
        self.logger.info(f"Created bibliography at {bib_path}")
        return bib_path
