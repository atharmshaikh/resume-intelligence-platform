"""
Parser module for extracting raw text from resume documents.

Supported formats:
- PDF
- DOCX
"""

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .docx_parser import DocxParser

__all__ = [
    "BaseParser",
    "PDFParser",
    "DocxParser",
]