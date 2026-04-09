"""
Main orchestration pipeline for resume processing.

Pipeline stages:
1. File validation
2. Document parsing
3. Text cleaning
4. Section detection
5. ATS normalization
6. Feature extraction
7. Hard rule filtering

This module uses the new pipeline/resume_pipeline.py for core processing.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import concurrent.futures
import re

from ml_engine.config import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from ml_engine.parsers import PDFParser, DocxParser
from ml_engine.utils import (
    clean_text,
    ResumeEngineError,
    ResumeParserError,
    PipelineTimeoutError
)
from ml_engine.extraction import detect_sections, extract_entities, load_wordlist
from ml_engine.normalization import build_ats_structure, fix_location
from ml_engine.features import extract_features
from ml_engine.quality import count_typos
from ml_engine.pipeline import ResumePipeline

logger = logging.getLogger(__name__)

# Use new pipeline for core processing
_core_pipeline = ResumePipeline()

_ALLOWED_SECTIONS = {
    "skills", "education", "projects", "experience",
    "achievements", "languages", "interests", "summary",
}


def _wl(name: str) -> set[str]:
    return {
        str(x).strip().lower()
        for x in (load_wordlist(name) or [])
        if str(x).strip() and not str(x).strip().startswith("#")
    }


def merge_skills(persona: Dict, sections: Dict) -> tuple[List[str], int]:
    """Merge skills from identity and sections, avoiding duplicates."""
    from ml_engine.normalization.skill_extractor import extract_skills
    from ml_engine.normalization.typo_detector import check_contact_typos

    all_raw_strings = []
    if persona.get("skills") and isinstance(persona["skills"], list):
        all_raw_strings.extend(persona["skills"])
    
    if "skills" in sections and isinstance(sections["skills"], list):
        all_raw_strings.extend(sections["skills"])
        
    merged_tokens, typo_count = extract_skills(all_raw_strings)
    
    # Add Contact Typos
    contact_typos = check_contact_typos(persona.get("email", ""), persona.get("phone", ""))
    typo_count += len(contact_typos)

    return sorted(list(set(merged_tokens))), typo_count


def compute_candidate_score(features: dict) -> float:
    """
    Compute overall candidate score using weighted formula.

    Normalization:
    - skills_score = min(skills_count / 10, 1)
    - projects_score = 1 if has_projects else 0
    - experience_score = 1 if has_experience else 0
    - education_score: if score >= 8 → 1, elif >= 6 → 0.7, else → 0.4

    Formula:
    score = (skills_score * 50) + (projects_score * 25) +
            (experience_score * 15) + (education_score * 10)

    Score range: 0-100
    """
    skills_count = features.get("skills_count", 0)
    skills_score = min(skills_count / 10.0, 1.0)

    projects_score = 1.0 if features.get("has_projects") else 0.0
    experience_score = 1.0 if features.get("has_experience") else 0.0

    education_raw = float(features.get("score", 0) or 0)
    if education_raw >= 8:
        education_score = 1.0
    elif education_raw >= 6:
        education_score = 0.7
    else:
        education_score = 0.4

    final_score = (
        (skills_score * 50) +
        (projects_score * 25) +
        (experience_score * 15) +
        (education_score * 10)
    )

    return round(max(0.0, min(100.0, final_score)), 2)


def apply_hard_rules(features: dict, identity: dict) -> Tuple[bool, List[str]]:
    """
    Apply hard rejection rules before ML.
    
    Returns:
        Tuple of (passes_rules, rejection_reasons)
    """
    reasons: List[str] = []
    
    has_email = bool(identity.get("email"))
    has_phone = bool(identity.get("phone"))
    has_valid_contact = int(has_email and has_phone)
    
    if has_valid_contact == 0:
        reasons.append("no_valid_contact")
    
    if features.get("skills_count", 0) == 0:
        reasons.append("no_skills")
    
    has_projects = features.get("has_projects", 0)
    has_experience = features.get("has_experience", 0)
    if has_projects == 0 and has_experience == 0:
        reasons.append("no_projects_and_no_experience")
    
    return (len(reasons) == 0, reasons)


_DATE_RANGE_RE = re.compile(
    r"(?:\d{1,2}/)?(?:19|20)\d{2}\s*[-–—]\s*(?:\d{1,2}/)?(?:19|20)\d{2}|present",
    re.IGNORECASE,
)
_DATE_ONLY_RE = re.compile(
    r"^(?:(?:\d{1,2}/)?(?:19|20)\d{2})\s*[-–—]\s*(?:(?:\d{1,2}/)?(?:19|20)\d{2}|present)$",
    re.IGNORECASE,
)
_COMMON_LANGS = set(load_wordlist("common_languages.txt"))
_TECH_TOKENS = (
    set(load_wordlist("programming_languages.txt"))
    | set(load_wordlist("frameworks.txt"))
    | set(load_wordlist("databases.txt"))
    | set(load_wordlist("tools.txt"))
    | set(load_wordlist("tech_terms.txt"))
)
_HEADING_ALIASES = {
    "skills": "skills",
    "technical skills": "skills",
    "education": "education",
    "academic details": "education",
    "projects": "projects",
    "project": "projects",
    "experience": "experience",
    "work experience": "experience",
    "languages": "languages",
    "language": "languages",
    "summary": "summary",
    "objective": "summary",
    "career objective": "summary",
    "interests": "interests",
    "hobbies": "interests",
    "achievements": "achievements",
    "certifications": "achievements",
}

class CandidateParser:
    """
    Central orchestrator for candidate-level parsing and validation.
    """

    def __init__(self):
        # Initialize document parsers
        self.pdf_parser = PDFParser()
        self.docx_parser = DocxParser()

    def _normalize_heading(self, line: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", line.strip().lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _build_sections_block_based(self, cleaned_text: str) -> Dict[str, List[str]]:
        lines = cleaned_text.splitlines()
        sections: Dict[str, List[str]] = {}
        current: str | None = None
        current_lines: List[str] = []
        blocks: Dict[str, List[List[str]]] = {}

        def flush_line_block() -> None:
            nonlocal current_lines
            if current and current_lines:
                blocks.setdefault(current, []).append(current_lines)
                current_lines = []

        for raw in lines:
            line = raw.strip()
            head = self._normalize_heading(line)
            section = _HEADING_ALIASES.get(head)
            if section:
                flush_line_block()
                current = section
                continue
            if not current:
                continue
            if not line:
                flush_line_block()
                continue
            current_lines.append(line)
        flush_line_block()

        for sec, sec_blocks in blocks.items():
            if sec in {"projects", "education", "experience", "achievements"}:
                sections[sec] = [" ".join(block).strip() for block in sec_blocks if block]
            else:
                merged: List[str] = []
                for block in sec_blocks:
                    merged.extend(block)
                sections[sec] = merged
        return {k: v for k, v in sections.items() if v}

    def _build_sections_heuristic(self, cleaned_text: str) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {k: [] for k in _ALLOWED_SECTIONS}
        for raw in cleaned_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if any(lang in low for lang in _COMMON_LANGS):
                sections["languages"].append(line)
                continue
            if any(k in low for k in ("b.tech", "bachelor", "master", "diploma", "cgpa", "gpa", "percentage", "university", "institute", "school")):
                sections["education"].append(line)
                continue
            if any(k in low for k in ("project", "system", "application", "developed", "implemented", "created", "designed")) or _DATE_RANGE_RE.search(low):
                sections["projects"].append(line)
                continue
            if any(k in low for k in ("intern", "experience", "role", "worked", "company", "employment")):
                sections["experience"].append(line)
                continue
            if any(k in low for k in ("achievement", "award", "certificate", "certification", "hackathon")):
                sections["achievements"].append(line)
                continue
            if any(tok in low for tok in _TECH_TOKENS):
                sections["skills"].append(line)
                continue
            if len(line.split()) >= 8:
                sections["summary"].append(line)
            else:
                sections["interests"].append(line)
        return {k: v for k, v in sections.items() if v}

    def _validate_and_score_candidate(self, resume_obj) -> Tuple[int, Dict[str, bool]]:
        checks: Dict[str, bool] = {}
        sections = set((resume_obj.sections or {}).keys())
        checks["allowed_sections_only"] = sections.issubset(_ALLOWED_SECTIONS)
        checks["no_unknown_sections"] = all(not k.startswith("unknown_") for k in sections)
        checks["skills_dedup"] = len(resume_obj.skills) == len(set(resume_obj.skills))
        checks["skills_no_sentence"] = all(len(str(s).split()) <= 3 and not any(c in str(s) for c in ".!?") for s in resume_obj.skills)
        checks["languages_valid"] = all(str(lang_code).lower() in _COMMON_LANGS for lang_code in resume_obj.languages)
        checks["project_name_not_date"] = all(not _DATE_ONLY_RE.fullmatch(str(p.get("name", "")).strip().lower()) for p in resume_obj.project_details)
        checks["project_desc_no_date"] = all(not _DATE_RANGE_RE.search(str(p.get("description", "")).lower()) for p in resume_obj.project_details)

        score = sum(10 for v in checks.values() if v)
        score += min(10, len(resume_obj.project_details) * 2)
        score += min(10, len(resume_obj.education_details) * 2)
        score += min(10, len(resume_obj.skills) // 2)
        return score, checks

    def _valid_skills(self, resume_obj) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for skill in list(getattr(resume_obj, "skills", []) or []):
            s = str(skill).lower().strip()
            if not s or s in seen:
                continue
            if s not in _TECH_TOKENS:
                continue
            if len(s.split()) > 3:
                continue
            out.append(s)
            seen.add(s)
        return out

    def _valid_projects(self, resume_obj) -> List[Dict[str, object]]:
        valid: List[Dict[str, object]] = []
        for item in list(getattr(resume_obj, "project_details", []) or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).strip()
            duration = str(item.get("duration", "")).strip()
            if not name or _DATE_ONLY_RE.fullmatch(name.lower()):
                continue
            if not desc:
                continue
            if not duration and _DATE_RANGE_RE.search(desc.lower()):
                continue
            valid.append({
                "name": name,
                "duration": duration,
                "description": desc,
            })
        return valid

    def _valid_experience(self, resume_obj) -> List[str]:
        out: List[str] = []
        for line in list(getattr(resume_obj, "experience", []) or []):
            line_val = str(line).strip()
            low = line_val.lower()
            if not line_val:
                continue
            if any(k in low for k in ("hackathon", "club", "activity", "activities")):
                continue
            if any(k in low for k in ("intern", "developer", "engineer", "analyst", "trainee", "role", "worked", "experience")):
                out.append(line_val)
        return out

    def _merge_best_candidate_components(self, candidates):
        # Base candidate = highest quality total.
        _, base_resume, _ = max(candidates, key=lambda x: x[2])

        # Choose best components independently.
        best_skills = max((self._valid_skills(r) for _, r, _ in candidates), key=len, default=[])
        best_projects = max((self._valid_projects(r) for _, r, _ in candidates), key=len, default=[])
        best_experience = max((self._valid_experience(r) for _, r, _ in candidates), key=len, default=[])

        base_resume.skills = best_skills
        base_resume.project_details = best_projects
        base_resume.projects = [
            " ".join([str(x) for x in [p.get("name"), p.get("duration"), p.get("description")] if x]).strip()
            for p in best_projects
        ]
        base_resume.experience = best_experience
        # Screening pipeline should not rely on low-signal sections.
        base_resume.languages = []
        base_resume.interests = []
        base_resume.summary_lines = []
        for key in ["languages", "interests", "summary"]:
            if key in base_resume.sections:
                base_resume.sections.pop(key, None)
        return base_resume

    def _run_multi_pass_parsing(self, cleaned_text: str):
        strategies: List[Tuple[str, Dict[str, List[str]]]] = []
        strategies.append(("structured", detect_sections(cleaned_text)))
        strategies.append(("block_based", self._build_sections_block_based(cleaned_text)))
        strategies.append(("heuristic", self._build_sections_heuristic(cleaned_text)))
        
        # Extract entities once for all strategies
        from ml_engine.extraction import extract_entities
        entities = extract_entities(cleaned_text)
        
        candidates = []
        for name, sections in strategies:
            resume = build_ats_structure(cleaned_text, sections, entities)
            score, checks = self._validate_and_score_candidate(resume)
            logger.info("Parser strategy=%s score=%s checks=%s", name, score, checks)
            candidates.append((name, resume, score))

        if not candidates:
            return build_ats_structure(cleaned_text, {}, entities)
        return self._merge_best_candidate_components(candidates)

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

    def to_optimized_dict(self, resume_obj) -> dict:
        """
        Convert full ResumeSchema to ML-ready feature dictionary.
        
        Pipeline:
        1. Merge skills from all sources
        2. Enrich skills from project descriptions (exact keywords only)
        3. Validate features
        4. Return flat feature dict for ML
        
        Output contains ONLY features (no scores, no decisions).
        """
        minimal_education: Dict[str, Any] = {}
        if getattr(resume_obj, "education_details", None):
            first = (resume_obj.education_details or [{}])[0]
            if isinstance(first, dict):
                minimal_education = {
                    "degree_type": str(first.get("degree_type", "")).strip(),
                    "is_it_candidate": int(first.get("is_it_candidate", 0) or 0),
                    "score": float(first.get("score", 0) or 0),
                }

        projects_minimal = []
        for item in (resume_obj.project_details or []):
            if not isinstance(item, dict):
                continue
            projects_minimal.append({
                "name": str(item.get("name", "")).strip(),
                "duration": str(item.get("duration", "").strip()),
                "description": str(item.get("description", "")).strip(),
            })

        identity = {
            "name": resume_obj.name,
            "email": resume_obj.email,
            "phone": resume_obj.phone,
            "location": resume_obj.location,
        }

        temp_persona = {
            "skills": list(resume_obj.skills),
            "email": identity.get("email"),
            "phone": identity.get("phone")
        }
        temp_sections = {k: list(v) for k, v in resume_obj.sections.items()}
        merged_skills, typo_count = merge_skills(temp_persona, temp_sections)

        enrichment_keywords = ("deep learning", "computer vision")
        for pr in projects_minimal:
            desc = str(pr.get("description", "")).lower()
            for keyword in enrichment_keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", desc) and keyword not in merged_skills:
                    merged_skills.append(keyword)

        wl_fw = _wl("frameworks.txt")
        wl_db = _wl("databases.txt")
        wl_prog = _wl("programming_languages.txt")
        
        # Pull advanced features from the pre-computed feature extractor
        full_features = getattr(resume_obj, "features", {})
        exp_years = float(full_features.get("experience_years_estimate", 0.0) or 0.0)

        has_email = bool(identity.get("email"))
        has_phone = bool(identity.get("phone"))
        has_valid_contact = int(has_email and has_phone)

        degree_type_raw = str(minimal_education.get("degree_type", "")).strip().lower()
        degree_type_map = {"diploma": 0, "bachelor": 1, "master": 2, "unknown": -1}
        degree_type_numeric = degree_type_map.get(degree_type_raw, -1)

        features = {
            "skills_count": len(merged_skills),
            "programming_languages_count": sum(1 for s in merged_skills if s in wl_prog),
            "framework_count": sum(1 for s in merged_skills if s in wl_fw),
            "database_count": sum(1 for s in merged_skills if s in wl_db),
            "years_of_experience": exp_years,
            "projects_count": len(projects_minimal),
            "has_projects": int(bool(projects_minimal)),
            "has_experience": int(bool(resume_obj.experience)),
            "has_internship": int(any("intern" in str(line).lower() for line in resume_obj.experience)),
            "degree_type": degree_type_numeric,
            "is_it_candidate": int(minimal_education.get("is_it_candidate", 0) or 0),
            "score": float(minimal_education.get("score", 0) or 0),
            "has_valid_contact": has_valid_contact,
            
            # --- Power Features ---
            "typo_count": typo_count,
            "ats_total_penalty_score": float(full_features.get("ats_total_penalty_score", 0)),
            "overall_profile_strength": float(full_features.get("overall_profile_strength", 0)),
            "quantified_impact_count": float(full_features.get("quantified_impact_count", 0)),
            "online_presence_count": float(full_features.get("online_presence_count", 0)),
        }

        if not _validate_features(features, merged_skills):
            logger.warning("Feature validation failed for resume. Skipping.")
            return {}

        if not _passes_hard_rules(features):
            logger.warning("Hard rules failed for resume. Skipping.")
            return {}

        result = {
            "identity": identity,
            "normalized_resume": {
                "skills": merged_skills,
                "education": minimal_education,
                "experience": list(resume_obj.experience),
                "projects": projects_minimal,
            },
            "features": features,
        }

        return result

    def parse(self, file_path: Union[str, Path], timeout_seconds: float = 30.0) -> dict:
        """
        Execute resume processing pipeline safely and return ML-ready data.
        """
        file_path = Path(file_path).expanduser()

        try:
            file_path = file_path.resolve(strict=True)
        except FileNotFoundError:
            msg = f"Resume file not found: {file_path}"
            logger.error(msg)
            raise ResumeEngineError(msg)

        logger.info(f"Starting ML Engine pipeline for: {file_path.name}")

        self._validate_file(file_path)

        parser = self._select_parser(file_path)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(parser.parse, str(file_path))
                raw_text = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            msg = f"Pipeline Timeout: Parsing {file_path.name} exceeded {timeout_seconds} seconds."
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

        cleaned_text = clean_text(raw_text)

        MAX_TEXT_LENGTH = 200000
        if len(cleaned_text) > MAX_TEXT_LENGTH:
            logger.warning(f"Truncating internal processing length for huge resume: {file_path.name}")
            cleaned_text = cleaned_text[:MAX_TEXT_LENGTH]

        resume_object = self._run_multi_pass_parsing(cleaned_text)
        resume_object.raw_text = cleaned_text

        entities = extract_entities(cleaned_text)
        if entities.get("name"):
            resume_object.name = entities["name"]
        if entities.get("email"):
            resume_object.email = entities["email"]
        if entities.get("phone"):
            resume_object.phone = entities["phone"]
        if entities.get("location"):
            resume_object.location = fix_location(entities["location"])

        try:
            features = extract_features(resume_object)
        except Exception as exc:
            msg = f"Feature extraction crash for {file_path.name}"
            logger.exception(msg)
            raise ResumeEngineError(msg) from exc

        resume_object.features = features

        if not hasattr(resume_object, "quality") or resume_object.quality is None:
            resume_object.quality = {}
        resume_object.quality["typos"] = count_typos(cleaned_text)

        logger.info(f"Successfully finished pipeline processing for {file_path.name}")

        return self.to_optimized_dict(resume_object)


def _validate_features(features: dict, skills: list[str]) -> bool:
    """
    Validate features for ML readiness.
    
    Checks:
    - skills_count == len(skills)
    - no negative counts
    - no missing values
    - all fields present
    """
    required_fields = [
        "skills_count", "programming_languages_count", "framework_count",
        "database_count", "projects_count", "has_projects", "has_experience",
        "has_internship", "degree_type", "is_it_candidate", "score", "has_valid_contact"
    ]
    
    for field in required_fields:
        if field not in features:
            return False
        if features[field] is None:
            return False
    
    if features["skills_count"] != len(skills):
        return False
    
    count_fields = ["skills_count", "programming_languages_count", "framework_count",
                    "database_count", "projects_count"]
    for field in count_fields:
        if features[field] < 0:
            return False
    
    binary_fields = ["has_projects", "has_experience", "has_internship",
                     "has_valid_contact", "is_it_candidate"]
    for field in binary_fields:
        if features[field] not in (0, 1):
            return False
    
    if features["degree_type"] not in (-1, 0, 1, 2):
        return False
    
    return True


def _passes_hard_rules(features: dict) -> bool:
    """
    Apply hard rejection rules before ML.
    
    Reject if:
    - has_valid_contact == 0
    - skills_count == 0
    - projects_count == 0 AND has_experience == 0
    """
    if features.get("has_valid_contact", 0) == 0:
        return False
    if features.get("skills_count", 0) == 0:
        return False
    if features.get("projects_count", 0) == 0 and features.get("has_experience", 0) == 0:
        return False
    return True
