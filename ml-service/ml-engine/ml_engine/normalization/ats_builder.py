"""
ATS normalization engine - Main builder.

This module orchestrates the normalization process by delegating
to specialized builder modules. It should be concise and readable.

For detailed extraction logic, see:
- skill_extractor.py
- project_extractor.py
- education_builder.py
- experience_builder.py
- identity_builder.py
"""

import logging
from typing import Any, Dict, List

from ml_engine.schemas import ResumeSchema
from ml_engine.normalization.skill_extractor import extract_skills
from ml_engine.normalization.project_extractor import extract_project_details, clean_project_name
from ml_engine.normalization.identity_builder import extract_identity
from ml_engine.normalization.skills_builder import build_skills, extract_project_skills
from ml_engine.normalization.education_builder import build_education
from ml_engine.normalization.experience_builder import build_experience
from ml_engine.normalization.project_builder import build_projects
from ml_engine.features import extract_features

logger = logging.getLogger(__name__)

# Re-export commonly used functions for backward compatibility
from ml_engine.normalization.ats_builder_utils import (
    _clean_lines,
    clean_experience,
    clean_project_name,
    _as_plain_dicts,
    _canonicalize_sections,
    _coerce_sections_map,
    _clean_achievement_lines,
    _clean_summary_lines,
    _clean_interest_lines,
    recompute_section_flags,
    fix_location,
)


def build_ats_structure(
    raw_text: str,
    sections: Dict[Any, Any] | Any,
    entities: Dict[str, Any] | None = None
) -> ResumeSchema:
    """
    Convert detected sections into ATS structured resume.

    Args:
        raw_text: Raw resume text
        sections: Detected sections dictionary
        entities: Extracted entities (name, email, phone, location)

    Returns:
        ResumeSchema object with all normalized fields
    """
    if not isinstance(sections, dict):
        logger.warning(f"Invalid sections type provided: {type(sections)}. Resetting to empty dict.")
        sections_map: Dict[str, List[str]] = {}
    else:
        sections_map = _canonicalize_sections(_coerce_sections_map(sections))

    entities_dict: Dict[str, Any] = entities if entities is not None else {}

    resume = ResumeSchema()
    resume.raw_text = raw_text
    resume.sections = sections_map

    # IDENTITY
    identity = extract_identity(sections_map, entities_dict, raw_text)
    resume.name = identity.get("name")
    resume.email = identity.get("email")
    resume.phone = identity.get("phone")
    resume.location = identity.get("location")

    # PROJECTS (extract first for skill enrichment)
    project_lines = _extract_project_lines_from_sections(sections_map)
    if project_lines and "projects" not in sections_map:
        sections_map["projects"] = list(project_lines)
    
    if project_lines:
        resume.project_details = _as_plain_dicts(extract_project_details(project_lines))
        resume.projects = _extract_projects(project_lines)
        if not resume.project_details and "projects" in sections_map:
            sections_map.pop("projects", None)

    # SKILLS (with project enrichment)
    project_skills = extract_project_skills([
        p.get("description", "") for p in resume.project_details
    ])
    resume.skills = build_skills(sections_map, raw_text, project_skills)

    # EDUCATION
    resume.education = []
    if "education" in sections_map:
        resume.education = _clean_lines(sections_map["education"])
        resume.education_details = _as_plain_dicts(build_education(sections_map))

    # EXPERIENCE
    resume.experience = []
    if "experience" in sections_map:
        resume.experience = build_experience(sections_map)

    # ACHIEVEMENTS
    resume.achievements = []
    if "achievements" in sections_map:
        resume.achievements = _clean_achievement_lines(_clean_lines(sections_map["achievements"]))

    # SUMMARY
    resume.summary_lines = []
    if "summary" in sections_map:
        resume.summary_lines = _clean_summary_lines(_clean_lines(sections_map["summary"]))

    # INTERESTS
    resume.interests = []
    if "interests" in sections_map:
        resume.interests = _clean_interest_lines(_clean_lines(sections_map["interests"]))

    # LANGUAGES
    resume.languages = []
    if "languages" in sections_map:
        resume.languages = _clean_lines(sections_map["languages"])

    # COMPUTE FEATURES
    resume.features = extract_features(resume)

    # Recompute section flags
    resume_data = {
        "normalized_resume": {
            "skills": resume.skills,
            "education": resume.education_details,
            "experience": resume.experience,
            "projects": resume.project_details,
            "achievements": resume.achievements,
            "languages": resume.languages,
            "interests": resume.interests,
            "summary": resume.summary_lines,
        },
        "features": resume.features,
    }
    resume_data = recompute_section_flags(resume_data)
    resume.features = resume_data["features"]

    return resume


def _extract_project_lines_from_sections(sections_map: Dict[str, List[str]]) -> List[str]:
    """Extract project-related lines from sections."""
    from ml_engine.normalization.project_extractor import _PROJECT_NOISE_RE, PROJECT_DATE_RE
    
    if "projects" in sections_map:
        return [
            ln for ln in sections_map["projects"]
            if not _PROJECT_NOISE_RE.search(ln.lower())
        ]
    
    candidates: List[str] = []
    for sec_name, lines in sections_map.items():
        if sec_name in {"education", "languages", "summary", "interests", "skills"}:
            continue
        if sec_name not in {"experience", "achievements"}:
            continue
        for line in _clean_lines(lines):
            ll = line.lower()
            if _PROJECT_NOISE_RE.search(ll):
                continue
            if any(k in ll for k in ("project", "system", "application")):
                candidates.append(line)
                continue
            if any(v in ll for v in ("developed", "implemented", "created", "built", "designed")):
                candidates.append(line)
                continue
    
    # Scan other sections for misplaced projects
    for sec_name, lines in sections_map.items():
        if sec_name in {"projects", "education", "experience", "achievements"}:
            continue
        
        sec_lines = _clean_lines(lines)
        project_like_lines = []
        
        for i, line in enumerate(sec_lines):
            ll = line.lower()
            if _PROJECT_NOISE_RE.search(ll):
                continue
            if ll in {"english", "hindi", "gujarati", "tamil", "telugu", "kannada", "marathi"}:
                continue
            if line.strip().isupper() and len(line.split()) <= 2:
                continue
            
            is_project_like = False
            if any(k in ll for k in ("portfolio", "project", "web clone", "game", "app", "application", "system", "platform", "website")):
                is_project_like = True
            if any(v in ll for v in ("developed", "created", "built", "designed", "implemented", "made", "using", "used")):
                is_project_like = True
            
            tech_count = sum(1 for t in ("html", "css", "javascript", "js", "node", "mongo", "sql", "react", "php", "python", "firebase", "api", "fetch") if t in ll)
            if tech_count >= 2:
                is_project_like = True
            
            if ":" in line and len(line.split()) >= 2:
                parts = line.split(":", 1)
                if len(parts[0].split()) <= 4:
                    is_project_like = True
            
            if not is_project_like and i + 1 < len(sec_lines):
                next_line = sec_lines[i + 1].lower()
                next_has_action = any(v in next_line for v in ("developed", "created", "built", "designed", "implemented", "made", "using", "used"))
                next_tech_count = sum(1 for t in ("html", "css", "javascript", "js", "node", "mongo", "sql", "react", "php", "python", "firebase", "api", "fetch") if t in next_line)
                if next_has_action or next_tech_count >= 2:
                    if len(line.split()) <= 4 and not line.strip().isupper():
                        is_project_like = True
            
            if is_project_like:
                project_like_lines.append(line)
        
        if len(project_like_lines) >= 3:
            candidates.extend(project_like_lines)
    
    return candidates


def _extract_projects(text: str | List[str]) -> List[str]:
    """Extract project strings from text."""
    return [
        " ".join(
            part for part in [
                item.get("name", "").strip(),
                item.get("duration", "").strip(),
                item.get("description", "").strip(),
            ]
            if part
        ).strip()
        for item in extract_project_details(text)
    ]
