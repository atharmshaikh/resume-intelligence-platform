"""
Rule-based ATS scoring system.

Produces interpretable resume quality scores.
"""
import logging
from typing import Dict, Any
from ml_engine.quality import typo_score

logger = logging.getLogger(__name__)

def _safe(features: dict, key: str, default=0):
    v = features.get(key, default)
    if v is None:
        return default
    return v

def score_resume(features: dict, raw_text: str) -> Dict[str, float]:
    try:
        return _score_resume_impl(features, raw_text)
    except Exception as exc:
        logger.exception("ATS Scoring crashed. Returning baseline 0.")
        return {
            "ats_score": 0.0,
            "contact_score": 0.0,
            "structure_score": 0.0,
            "content_score": 0.0,
            "length_score": 0.0,
            "typo_score": 0.0,
        }

def _score_resume_impl(features: dict, raw_text: str) -> Dict[str, float]:

    scores = {}

    # -------------------------
    # Contact score
    # -------------------------

    contact_score = (
        _safe(features, "has_name")
        + _safe(features, "has_email")
        + _safe(features, "has_phone")
        + _safe(features, "has_location")
    ) / 4 * 100

    # -------------------------
    # Structure score
    # -------------------------

    structure_points = (
        _safe(features, "has_skills_section")
        + _safe(features, "has_education_section")
        + _safe(features, "has_projects_section")
    )   

    structure_score = structure_points / 3 * 100

    # -------------------------
    # Content score
    # -------------------------

    skill_score = min(_safe(features, "skills_count") / 15, 1)

    project_score = _safe(features, "has_projects")

    education_score = min(_safe(features, "education_count"), 1)

    content_score = (skill_score + project_score + education_score) / 3 * 100

    # -------------------------
    # Resume length score
    # -------------------------

    word_count = _safe(features, "resume_word_count")

    if word_count < 200:
        length_score = 40
    elif word_count < 350:
        length_score = 75
    elif word_count < 900:
        length_score = 100
    elif word_count < 1400:
        length_score = 85
    else:
        length_score = 60

    # -------------------------
    # Typo score
    # -------------------------

    typo_quality = typo_score(raw_text)

    # -------------------------
    # Final ATS score
    # -------------------------

    ats_score = (
        contact_score * 0.2
        + structure_score * 0.25
        + content_score * 0.25
        + length_score * 0.15
        + typo_quality * 0.15
    )

    ats_score = max(0, min(100, ats_score))
    
    scores["ats_score"] = round(ats_score, 2)
    scores["contact_score"] = round(contact_score, 2)
    scores["structure_score"] = round(structure_score, 2)
    scores["content_score"] = round(content_score, 2)
    scores["length_score"] = round(length_score, 2)
    scores["typo_score"] = round(typo_quality, 2)
    
    return scores