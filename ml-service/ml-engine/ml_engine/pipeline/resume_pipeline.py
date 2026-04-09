"""
Resume Processing Pipeline

Orchestrates the complete resume processing flow:
1. Parse raw text from file
2. Detect sections
3. Extract entities
4. Normalize into ATS format
5. Build features
6. Run ML prediction
7. Save result
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ml_engine.parsers import PDFParser, DocxParser
from ml_engine.utils import clean_text, ResumeParserError
from ml_engine.extraction import detect_sections, extract_entities
from ml_engine.normalization import build_ats_structure
from ml_engine.features import extract_features
from ml_engine.ml.inference.predictor import ResumePredictor

logger = logging.getLogger(__name__)


class ResumePipeline:
    """
    Main pipeline for resume processing.
    
    Orchestrates all steps from raw file to ML prediction.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize pipeline.
        
        Args:
            config_path: Optional path to ML config file
        """
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()
        self.predictor = ResumePredictor(config_path) if config_path else None
        
        logger.info("ResumePipeline initialized")

    def process(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Process a resume file through the complete pipeline.
        
        Args:
            file_path: Path to resume file (PDF or DOCX)
            
        Returns:
            Dictionary containing parsed data, features, and ML prediction
        """
        file_path = Path(file_path)
        result: Dict[str, Any] = {
            "file": str(file_path),
            "identity": {},
            "normalized_resume": {},
            "features": {},
            "ml_prediction": None,
        }
        
        try:
            # [1/6] Parse resume
            logger.info("[1/6] Parsing resume...")
            raw_text = self._parse_file(file_path)
            cleaned_text = clean_text(raw_text)
            logger.info(f"    Parsed {len(cleaned_text)} characters")
            
            # [2/6] Detect sections
            logger.info("[2/6] Detecting sections...")
            sections = detect_sections(cleaned_text)
            logger.info(f"    Found {len(sections)} sections: {list(sections.keys())}")
            
            # [3/6] Extract entities
            logger.info("[3/6] Extracting entities...")
            entities = extract_entities(cleaned_text)
            logger.info(f"    Extracted: name={entities.get('name')}, email={entities.get('email')}")
            
            # [4/6] Build ATS structure
            logger.info("[4/6] Building ATS structure...")
            resume_obj = build_ats_structure(cleaned_text, sections, entities)
            logger.info(f"    Skills: {len(resume_obj.skills)}, Projects: {len(resume_obj.project_details)}")
            
            # [5/6] Generate features
            logger.info("[5/6] Generating features...")
            features = extract_features(resume_obj)
            logger.info(f"    Generated {len(features)} features")
            
            # [6/6] Run ML prediction
            logger.info("[6/6] Running ML prediction...")
            ml_result = None
            if self.predictor:
                ml_result = self.predictor.predict(features)
                logger.info(f"    Prediction complete")
            
            # Build result
            result["identity"] = {
                "name": resume_obj.name,
                "email": resume_obj.email,
                "phone": resume_obj.phone,
                "location": resume_obj.location,
            }
            result["normalized_resume"] = {
                "skills": list(resume_obj.skills),
                "education": list(resume_obj.education_details) if resume_obj.education_details else [],
                "experience": list(resume_obj.experience) if resume_obj.experience else [],
                "projects": list(resume_obj.project_details) if resume_obj.project_details else [],
            }
            result["features"] = features
            result["ml_prediction"] = ml_result
            
            logger.info("Pipeline completed successfully")
            return result
            
        except ResumeParserError as e:
            logger.error(f"Parsing failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise

    def _parse_file(self, file_path: Path) -> str:
        """
        Parse resume file based on extension.
        
        Args:
            file_path: Path to resume file
            
        Returns:
            Raw text content
        """
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return self.pdf_parser.parse(str(file_path))
        elif suffix == ".docx":
            return self.docx_parser.parse(str(file_path))
        else:
            raise ResumeParserError(f"Unsupported file format: {suffix}")

    def parse_only(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Parse resume without ML prediction.
        
        Useful for debugging or batch feature extraction.
        
        Args:
            file_path: Path to resume file
            
        Returns:
            Dictionary with parsed data and features (no ML prediction)
        """
        file_path = Path(file_path)
        
        # Temporarily disable predictor
        original_predictor = self.predictor
        self.predictor = None
        
        try:
            result = self.process(file_path)
            return result
        finally:
            self.predictor = original_predictor
