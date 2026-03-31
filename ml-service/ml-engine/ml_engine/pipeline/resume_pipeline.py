"""
Main orchestration pipeline for resume processing.

Pipeline stages:
1. File validation
2. Document parsing (with Overload Protection / Timeout)
3. Text cleaning
4. Section detection
5. ATS normalization
6. Scoring & Feature Extraction
"""

import os
import logging
from pathlib import Path
from typing import Union
import concurrent.futures

from ml_engine.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from ml_engine.parsers import PDFParser, DocxParser
from ml_engine.utils import (
    clean_text,
    ResumeEngineError,
    ResumeParserError,
    PipelineTimeoutError
)
from ml_engine.extraction import detect_sections, extract_entities
from ml_engine.normalization import build_ats_structure
from ml_engine.features import extract_features
from ml_engine.scoring import score_resume
from ml_engine.quality import count_typos

logger = logging.getLogger(__name__)

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
            msg = f"Resume file not found: {file_path}"
            logger.error(msg)
            raise ResumeEngineError(msg)

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            msg = f"Unsupported file format: {suffix}"
            logger.error(msg)
            raise ResumeParserError(msg)

        if not os.access(file_path, os.R_OK):
            msg = f"File is not readable: {file_path}"
            logger.error(msg)
            raise ResumeEngineError(msg)

        # File size validation
        size_mb = file_path.stat().st_size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            msg = f"File too large ({size_mb:.2f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB"
            logger.warning(msg)
            raise ResumeParserError(msg)

    def _select_parser(self, file_path: Path):
        """
        Select parser based on file extension.
        """

        suffix = file_path.suffix.lower()

        parser_map = {
            ".pdf": self.pdf_parser,
            ".docx": self.docx_parser,
        }

        parser = parser_map.get(suffix)

        if not parser:
            msg = f"No parser available for {suffix}"
            logger.error(msg)
            raise ResumeParserError(msg)

        return parser

    def parse(self, file_path: Union[str, Path], timeout_seconds: float = 30.0):
        """
        Execute resume processing pipeline safely.
        """

        file_path = Path(file_path).expanduser()

        try:
            file_path = file_path.resolve(strict=True)
        except FileNotFoundError:
            msg = f"Resume file not found: {file_path}"
            logger.error(msg)
            raise ResumeEngineError(msg)

        logger.info(f"Starting ML Engine pipeline for: {file_path.name}")

        # -------------------------
        # 1. Validate file
        # -------------------------
        self._validate_file(file_path)

        # -------------------------
        # 2. Parse document (with ThreadPool Timeout Protection)
        # -------------------------
        parser = self._select_parser(file_path)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(parser.parse, str(file_path))
                raw_text = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            msg = f"Pipeline Timeout: Parsing {file_path.name} exceeded {timeout_seconds} seconds. Aborted to prevent server overload."
            logger.error(msg)
            raise PipelineTimeoutError(msg) from exc
        except ResumeParserError:
            raise
        except Exception as exc:
            msg = f"Critical crash during document parsing for {file_path.name}"
            logger.exception(msg)
            raise ResumeParserError(msg) from exc

        if not raw_text or len(raw_text.split()) < 15:
            msg = f"Parser returned insufficient text for resume: {file_path.name}"
            logger.warning(msg)
            raise ResumeParserError(msg)
        
        # -------------------------
        # 3. Clean extracted text
        # -------------------------
        cleaned_text = clean_text(raw_text)

        # Prevent excessive processing on very large resumes for internal layers
        MAX_TEXT_LENGTH = 200000

        if len(cleaned_text) > MAX_TEXT_LENGTH:
            logger.warning(f"Truncating internal processing length for huge resume: {file_path.name}")
            cleaned_text = cleaned_text[:MAX_TEXT_LENGTH]

        # -------------------------
        # 4. Detect sections
        # -------------------------
        sections = detect_sections(cleaned_text)

        # -------------------------
        # 5. Build ATS structure
        # -------------------------
        resume_object = build_ats_structure(cleaned_text, sections)

        # Ensure schema retains cleaned raw text
        resume_object.raw_text = cleaned_text

        # Candidate identity extraction (name, email, phone, location)
        # -------------------------
        # 6. Extract entities
        # -------------------------
        entities = extract_entities(cleaned_text)

        if entities.get("name"):
            resume_object.name = entities["name"]
        if entities.get("email"):
            resume_object.email = entities["email"]
        if entities.get("phone"):
            resume_object.phone = entities["phone"]
        if entities.get("location"):
            resume_object.location = entities["location"]
        
        # -------------------------
        # 7. Feature extraction (Stage 3)
        # -------------------------
        try:
            features = extract_features(resume_object)
        except Exception as exc:
            msg = f"Feature extraction crash for {file_path.name}"
            logger.exception(msg)
            raise ResumeEngineError(msg) from exc

        resume_object.features = features

        # -------------------------
        # 8. ATS scoring
        # -------------------------
        try:
            scores = score_resume(features, cleaned_text)
        except Exception as exc:
            msg = f"ATS scoring crash for {file_path.name}"
            logger.exception(msg)
            raise ResumeEngineError(msg) from exc

        resume_object.scores = scores

        # -------------------------
        # 9. Quality analysis
        # -------------------------
        if not hasattr(resume_object, "quality") or resume_object.quality is None:
            resume_object.quality = {}

        resume_object.quality["typos"] = count_typos(cleaned_text)
        
        logger.info(f"Successfully finished pipeline processing for {file_path.name}")
        return resume_object