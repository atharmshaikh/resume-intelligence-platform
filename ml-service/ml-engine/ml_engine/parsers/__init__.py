"""
Parsers module for resume text extraction.

Provides:
- PDF parsing with 2-column layout handling
- DOCX parsing
- Text cleaning utilities
- Layout normalization
- Block segmentation
- Section classification
- Entity extraction
- Complete parsing pipeline
"""

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .text_cleaner import clean_text, split_into_lines, is_bullet_line, is_heading_line
from .layout_parser import LayoutParser, normalize_layout, detect_layout_type
from .block_segmenter import BlockSegmenter, TextBlock, segment_text
from .section_classifier import SectionClassifier, classify_section, classify_sections
from .entity_extractor import EntityExtractor, extract_entities, ExtractedEntities
from .parsing_pipeline import ParsingPipeline, parse_resume, parse_resume_minimal

__all__ = [
    # Base
    "BaseParser",
    # Parsers
    "PDFParser",
    "DocxParser",
    # Text cleaning
    "clean_text",
    "split_into_lines",
    "is_bullet_line",
    "is_heading_line",
    # Layout
    "LayoutParser",
    "normalize_layout",
    "detect_layout_type",
    # Blocks
    "BlockSegmenter",
    "TextBlock",
    "segment_text",
    # Sections
    "SectionClassifier",
    "classify_section",
    "classify_sections",
    # Entities
    "EntityExtractor",
    "extract_entities",
    "ExtractedEntities",
    # Pipeline
    "ParsingPipeline",
    "parse_resume",
    "parse_resume_minimal",
]
