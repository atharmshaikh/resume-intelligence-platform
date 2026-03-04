"""
Main orchestration pipeline for resume processing.

Pipeline stages:
1. File validation
2. Document parsing
3. Text cleaning
4. Section detection
5. ATS normalization
"""

import os
from pathlib import Path

from ml_engine.config.settings import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from ml_engine.parsers.pdf_parser import PDFParser
from ml_engine.parsers.docx_parser import DocxParser
from ml_engine.utils.text_cleaner import clean_text
from ml_engine.extraction.section_detector import detect_sections
from ml_engine.normalization.ats_builder import build_ats_structure
from ml_engine.extraction.entity_extractor import extract_entities

class ResumePipeline:
    """
    Central pipeline controller for resume parsing.
    """

    def __init__(self):

        # Initialize document parsers
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()

    def _validate_file(self, file_path: Path):
        """
        Validate resume file before processing.
        """

        if not file_path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")

        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File is not readable: {file_path}")

        # File size validation
        size_mb = file_path.stat().st_size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large ({size_mb:.2f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB"
            )

    def _select_parser(self, file_path: Path):
        """
        Select parser based on file extension.
        """

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self.pdf_parser

        if suffix == ".docx":
            return self.docx_parser

        raise ValueError(f"No parser available for {suffix}")

    from typing import Union

    def parse(self, file_path: Union[str, Path]):
        """
        Execute resume processing pipeline.
        """

        file_path = Path(file_path)

        # -------------------------
        # 1. Validate file
        # -------------------------
        self._validate_file(file_path)

        # -------------------------
        # 2. Parse document
        # -------------------------
        parser = self._select_parser(file_path)

        raw_text = parser.parse(str(file_path))

        if not raw_text:
            raise ValueError("Resume parsing produced empty text")

        # -------------------------
        # 3. Clean extracted text
        # -------------------------
        cleaned_text = clean_text(raw_text)

        # -------------------------
        # 4. Detect sections
        # -------------------------
        sections = detect_sections(cleaned_text)

        # -------------------------
        # 5. Build ATS structure
        # -------------------------
        resume_object = build_ats_structure(cleaned_text, sections)
        resume_object = build_ats_structure(cleaned_text, sections)

        entities = extract_entities(cleaned_text)

        resume_object.name = entities["name"]
        resume_object.email = entities["email"]
        resume_object.phone = entities["phone"]
        resume_object.location = entities["location"]
        
        return resume_object