"""
Research Paper Tagger - Automatic categorization of research papers.
"""

__version__ = "1.0.0"
__author__ = "Sreeram Lagisetty"
__description__ = "Automatically assign category tags to research papers based on their content"
__url__ = "https://github.com/Sreeram5678/research-paper-tagger"

from .pdf_extractor import PDFExtractor
from .nlp_classifier import ResearchPaperClassifier
from .paper_tagger import PaperTagger, PaperResult
from .output_handlers import OutputHandler
from .config_loader import ConfigLoader

__all__ = [
    'PDFExtractor',
    'ResearchPaperClassifier', 
    'PaperTagger',
    'PaperResult',
    'OutputHandler',
    'ConfigLoader'
]
