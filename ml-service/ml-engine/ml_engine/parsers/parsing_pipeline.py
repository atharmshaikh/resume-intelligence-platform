"""
Parsing Pipeline - Main Orchestrator

Orchestrates the complete resume parsing flow:
1. Raw text extraction (PDF/DOCX)
2. Text cleaning
3. Layout normalization (2-column handling)
4. Block segmentation
5. Section detection
6. Section classification
7. Entity extraction

Returns structured output compatible with existing pipeline.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ml_engine.parsers.pdf_parser import PDFParser
from ml_engine.parsers.docx_parser import DocxParser
from ml_engine.parsers.text_cleaner import clean_text
from ml_engine.parsers.layout_parser import normalize_layout, detect_layout_type
from ml_engine.parsers.block_segmenter import BlockSegmenter
from ml_engine.parsers.section_classifier import SectionClassifier, classify_sections
from ml_engine.parsers.entity_extractor import extract_entities
from ml_engine.extraction.section_detector import detect_sections as detect_sections_legacy

logger = logging.getLogger(__name__)


class ParsingPipeline:
    """
    Main orchestrator for resume parsing.
    
    Implements a 7-stage pipeline for robust resume parsing.
    """
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()
        self.block_segmenter = BlockSegmenter()
        self.section_classifier = SectionClassifier()
        
        logger.info("ParsingPipeline initialized")
    
    def parse(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Parse resume file through complete pipeline.
        
        Args:
            file_path: Path to resume file (PDF or DOCX)
            
        Returns:
            Dictionary with:
            - raw_text: Original extracted text
            - cleaned_text: Cleaned text
            - layout_type: 'single-column' or 'two-column'
            - sections: Detected sections
            - entities: Extracted entities
            - blocks: Segmented blocks (optional)
        """
        file_path = Path(file_path)
        logger.info(f"Starting parsing pipeline for: {file_path.name}")
        
        result: Dict[str, Any] = {
            "file": str(file_path),
            "raw_text": "",
            "cleaned_text": "",
            "layout_type": "unknown",
            "sections": {},
            "entities": {},
            "blocks": [],
        }
        
        try:
            # Stage 1: Raw text extraction
            logger.info("[1/7] Extracting raw text...")
            raw_text = self._extract_text(file_path)
            result["raw_text"] = raw_text
            logger.info(f"    Extracted {len(raw_text)} characters")
            
            # Stage 2: Text cleaning
            logger.info("[2/7] Cleaning text...")
            cleaned_text = clean_text(raw_text)
            result["cleaned_text"] = cleaned_text
            logger.info(f"    Cleaned {len(cleaned_text)} characters")
            
            # Stage 3: Layout normalization
            logger.info("[3/7] Normalizing layout...")
            layout_type = detect_layout_type(cleaned_text)
            result["layout_type"] = layout_type
            
            if layout_type == "two-column":
                normalized_text = normalize_layout(cleaned_text)
                logger.info("    Converted 2-column → single-column")
            else:
                normalized_text = cleaned_text
                logger.info("    Single-column layout detected")
            
            # Stage 4: Block segmentation
            logger.info("[4/7] Segmenting into blocks...")
            blocks = self.block_segmenter.segment(normalized_text)
            result["blocks"] = [b.to_dict() for b in blocks]
            logger.info(f"    Segmented into {len(blocks)} blocks")
            
            # Stage 5: Section detection (use legacy for compatibility)
            logger.info("[5/7] Detecting sections...")
            sections = detect_sections_legacy(normalized_text)
            logger.info(f"    Detected sections: {list(sections.keys())}")
            result["sections"] = sections
            
            # Stage 6: Section classification
            logger.info("[6/7] Classifying sections...")
            classified = self.section_classifier.classify_all(sections)
            for section_name, classification in classified.items():
                logger.debug(f"    '{section_name}' → {classification.category}")
            
            # Stage 7: Entity extraction
            logger.info("[7/7] Extracting entities...")
            entities = extract_entities(normalized_text)
            result["entities"] = entities
            logger.info(f"    Entities: name={entities.get('name')}, email={entities.get('email')}")
            
            logger.info(f"Parsing pipeline completed for {file_path.name}")
            return result
            
        except Exception as e:
            logger.exception(f"Parsing pipeline failed: {e}")
            raise
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text based on file type."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.pdf_parser.parse(str(file_path))
        elif suffix == '.docx':
            return self.docx_parser.parse(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def parse_minimal(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Parse with minimal processing (faster, less detailed).
        
        For batch processing where only sections + entities are needed.
        
        Args:
            file_path: Path to resume file
            
        Returns:
            Dictionary with sections and entities only
        """
        result = self.parse(file_path)
        
        # Return only essential fields
        return {
            "raw_text": result["raw_text"],
            "cleaned_text": result["cleaned_text"],
            "sections": result["sections"],
            "entities": result["entities"],
        }


def parse_resume(file_path: str | Path) -> Dict[str, Any]:
    """
    Convenience function to parse resume.
    
    Args:
        file_path: Path to resume file
        
    Returns:
        Parsed resume data
    """
    pipeline = ParsingPipeline()
    return pipeline.parse(file_path)


def parse_resume_minimal(file_path: str | Path) -> Dict[str, Any]:
    """
    Convenience function for minimal parsing.
    
    Args:
        file_path: Path to resume file
        
    Returns:
        Minimal parsed resume data
    """
    pipeline = ParsingPipeline()
    return pipeline.parse_minimal(file_path)
