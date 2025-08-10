"""
Main paper tagger module that coordinates PDF extraction and NLP classification.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .pdf_extractor import PDFExtractor
from .nlp_classifier import ResearchPaperClassifier


@dataclass
class PaperResult:
    """Result of paper tagging operation."""
    file_path: Path
    filename: str
    tags: List[Tuple[str, float]]
    metadata: Dict[str, str]
    text_sections: Dict[str, str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class PaperTagger:
    """Main class for automatically tagging research papers."""
    
    def __init__(self, config_path: Optional[Path] = None, 
                 confidence_threshold: float = 0.3,
                 max_tags: int = 5):
        """
        Initialize the paper tagger.
        
        Args:
            config_path: Path to configuration file for custom categories
            confidence_threshold: Minimum confidence for tag assignment
            max_tags: Maximum number of tags to assign per paper
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.max_tags = max_tags
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.classifier = ResearchPaperClassifier(config_path)
        
        self.logger.info("Paper tagger initialized successfully")
    
    def tag_paper(self, pdf_path: Union[str, Path], 
                  extraction_method: str = "auto") -> PaperResult:
        """
        Tag a single research paper.
        
        Args:
            pdf_path: Path to the PDF file
            extraction_method: PDF extraction method to use
            
        Returns:
            PaperResult object containing tagging results
        """
        start_time = datetime.now()
        pdf_path = Path(pdf_path)
        
        try:
            self.logger.info(f"Processing paper: {pdf_path.name}")
            
            # Extract text from PDF
            text_sections = self.pdf_extractor.extract_text(pdf_path, extraction_method)
            
            # Extract metadata
            metadata = self.pdf_extractor.extract_metadata(pdf_path)
            
            # Classify the paper
            tags = self.classifier.classify_text(text_sections, self.confidence_threshold)
            
            # Limit number of tags
            tags = tags[:self.max_tags]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Successfully tagged {pdf_path.name} with {len(tags)} tags")
            
            return PaperResult(
                file_path=pdf_path,
                filename=pdf_path.name,
                tags=tags,
                metadata=metadata,
                text_sections=text_sections,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Failed to process {pdf_path.name}: {str(e)}"
            self.logger.error(error_msg)
            
            return PaperResult(
                file_path=pdf_path,
                filename=pdf_path.name,
                tags=[],
                metadata={},
                text_sections={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def tag_papers_batch(self, pdf_directory: Union[str, Path], 
                        pattern: str = "*.pdf",
                        extraction_method: str = "auto") -> List[PaperResult]:
        """
        Tag multiple papers in a directory.
        
        Args:
            pdf_directory: Directory containing PDF files
            pattern: File pattern to match (default: "*.pdf")
            extraction_method: PDF extraction method to use
            
        Returns:
            List of PaperResult objects
        """
        pdf_directory = Path(pdf_directory)
        
        if not pdf_directory.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")
        
        # Find PDF files
        pdf_files = list(pdf_directory.glob(pattern))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_directory}")
            return []
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            self.logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            result = self.tag_paper(pdf_file, extraction_method)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Batch processing complete: {successful}/{len(results)} successful")
        
        return results
    
    def get_tag_statistics(self, results: List[PaperResult]) -> Dict[str, int]:
        """
        Get statistics about tag distribution.
        
        Args:
            results: List of paper results
            
        Returns:
            Dictionary with tag counts
        """
        tag_counts = {}
        
        for result in results:
            if result.success:
                for tag, _ in result.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by frequency
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
    
    def filter_papers_by_tag(self, results: List[PaperResult], 
                           target_tag: str) -> List[PaperResult]:
        """
        Filter papers that contain a specific tag.
        
        Args:
            results: List of paper results
            target_tag: Tag to filter by
            
        Returns:
            Filtered list of results
        """
        filtered = []
        target_tag_lower = target_tag.lower()
        
        for result in results:
            if result.success:
                for tag, _ in result.tags:
                    if target_tag_lower in tag.lower():
                        filtered.append(result)
                        break
        
        return filtered
    
    def get_papers_summary(self, results: List[PaperResult]) -> Dict[str, any]:
        """
        Get summary statistics for processed papers.
        
        Args:
            results: List of paper results
            
        Returns:
            Summary statistics
        """
        total_papers = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_papers - successful
        
        if successful > 0:
            avg_tags = sum(len(r.tags) for r in results if r.success) / successful
            avg_processing_time = sum(r.processing_time for r in results if r.success) / successful
        else:
            avg_tags = 0
            avg_processing_time = 0
        
        tag_stats = self.get_tag_statistics(results)
        
        return {
            'total_papers': total_papers,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_papers if total_papers > 0 else 0,
            'average_tags_per_paper': avg_tags,
            'average_processing_time': avg_processing_time,
            'tag_distribution': tag_stats,
            'most_common_tag': list(tag_stats.keys())[0] if tag_stats else None,
            'unique_tags': len(tag_stats)
        }
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update the confidence threshold for tag assignment."""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            self.logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def update_max_tags(self, new_max_tags: int):
        """Update the maximum number of tags per paper."""
        if new_max_tags > 0:
            self.max_tags = new_max_tags
            self.logger.info(f"Updated max tags to {new_max_tags}")
        else:
            raise ValueError("Max tags must be greater than 0")
    
    def add_custom_category(self, category_name: str, keywords: List[str]):
        """Add a custom category to the classifier."""
        self.classifier.add_custom_category(category_name, keywords)
        self.logger.info(f"Added custom category: {category_name}")
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories."""
        return self.classifier.get_available_categories()
    
    def save_classifier(self, model_path: Path):
        """Save the classifier model."""
        self.classifier.save_model(model_path)
    
    def load_classifier(self, model_path: Path):
        """Load a saved classifier model."""
        self.classifier.load_model(model_path)
