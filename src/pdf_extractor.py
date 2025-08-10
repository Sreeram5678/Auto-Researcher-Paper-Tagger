"""
PDF text extraction module for research paper processing.
"""
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List

import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader


class PDFExtractor:
    """Extract text from PDF files using multiple backends for robustness."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, pdf_path: Path, method: str = "auto") -> Dict[str, str]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ('auto', 'pymupdf', 'pdfplumber', 'pypdf2')
            
        Returns:
            Dictionary containing extracted text sections
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if method == "auto":
            # Try methods in order of preference
            for extraction_method in ["pymupdf", "pdfplumber", "pypdf2"]:
                try:
                    return self._extract_with_method(pdf_path, extraction_method)
                except Exception as e:
                    self.logger.warning(f"Failed to extract with {extraction_method}: {e}")
                    continue
            raise Exception("All extraction methods failed")
        else:
            return self._extract_with_method(pdf_path, method)
    
    def _extract_with_method(self, pdf_path: Path, method: str) -> Dict[str, str]:
        """Extract text using specific method."""
        if method == "pymupdf":
            return self._extract_with_pymupdf(pdf_path)
        elif method == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path)
        elif method == "pypdf2":
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text using PyMuPDF (fitz)."""
        doc = fitz.open(str(pdf_path))
        full_text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n"
        
        doc.close()
        return self._parse_paper_sections(full_text)
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text using pdfplumber."""
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        return self._parse_paper_sections(full_text)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Dict[str, str]:
        """Extract text using PyPDF2."""
        full_text = ""
        
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
        
        return self._parse_paper_sections(full_text)
    
    def _parse_paper_sections(self, text: str) -> Dict[str, str]:
        """
        Parse academic paper into sections.
        
        Args:
            text: Full text of the paper
            
        Returns:
            Dictionary with sections (abstract, introduction, conclusion, etc.)
        """
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        
        sections = {
            'full_text': text,
            'abstract': '',
            'introduction': '',
            'conclusion': '',
            'keywords': ''
        }
        
        # Extract abstract
        abstract_patterns = [
            r'abstract\s*[:\-]?\s*(.*?)(?=1\.|introduction|keywords|\n\n|\d+\s+introduction)',
            r'abstract\s*[:\-]?\s*(.*?)(?=\n[A-Z][a-z]|\n\d)',
            r'abstract[\s\n]*[:\-]?\s*(.*?)(?=\nkeywords|\nintroduction|\n1\.)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['abstract'] = match.group(1).strip()[:2000]  # Limit length
                break
        
        # Extract keywords
        keywords_patterns = [
            r'keywords?\s*[:\-]?\s*(.*?)(?=\n[A-Z]|\n\d|\nintroduction)',
            r'key\s*words?\s*[:\-]?\s*(.*?)(?=\n[A-Z]|\n\d|\nintroduction)',
        ]
        
        for pattern in keywords_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['keywords'] = match.group(1).strip()[:500]  # Limit length
                break
        
        # Extract introduction (first few paragraphs)
        intro_patterns = [
            r'(?:1\.?\s*)?introduction\s*[:\-]?\s*(.*?)(?=\n2\.|related work|methodology|method)',
            r'introduction\s*[:\-]?\s*(.*?)(?=\n[A-Z][a-z]+|\n\d)',
        ]
        
        for pattern in intro_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['introduction'] = match.group(1).strip()[:3000]  # Limit length
                break
        
        # Extract conclusion
        conclusion_patterns = [
            r'(?:\d+\.?\s*)?conclusion\s*[:\-]?\s*(.*?)(?=references|bibliography|\nreferences)',
            r'conclusion\s*[:\-]?\s*(.*?)(?=\n[A-Z][a-z]+|\nreferences)',
            r'concluding remarks\s*[:\-]?\s*(.*?)(?=references|bibliography)',
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections['conclusion'] = match.group(1).strip()[:2000]  # Limit length
                break
        
        return sections
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, str]:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                if reader.metadata:
                    metadata.update({
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'producer': reader.metadata.get('/Producer', ''),
                        'creation_date': str(reader.metadata.get('/CreationDate', '')),
                        'modification_date': str(reader.metadata.get('/ModDate', '')),
                    })
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
