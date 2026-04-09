"""
Rule-based ATS scoring system.

Produces interpretable resume quality scores with:
- Hard penalty layer for critical issues
- Dynamic content scoring
- Experience signal integration
- Consistency checks
- Explainability (reasons)
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _safe(features: dict, key: str, default=0):
    """Safely get feature value with default."""
    v = features.get(key, default)
    if v is None:
        return default
    return v


def compute_contact_score(features: dict, identity: dict) -> int:
    """
    Compute contact score based on email and phone presence.

    Logic:
    - if email AND phone → 100
    - if one present → 50
    - else → 0
    """
    has_email = bool(identity.get("email"))
    has_phone = bool(identity.get("phone"))

    if has_email and has_phone:
        return 100
    elif has_email or has_phone:
        return 50
    else:
        return 0


def score_resume(features: dict, raw_text: str, identity: dict | None = None) -> Dict[str, Any]:
    """
    Score resume with enhanced production-grade logic.

    Args:
        features: Feature dictionary from extraction
        raw_text: Raw resume text for length/typo analysis
        identity: Identity dictionary with email/phone

    Returns:
        Dictionary with scores and optional reasons
    """
    try:
        return _score_resume_impl(features, raw_text, identity or {})
    except Exception:
        logger.exception("ATS Scoring crashed. Returning baseline 0.")
        return {
            "ats_score": 0.0,
            "contact_score": 0.0,
            "structure_score": 0.0,
            "content_score": 0.0,
            "length_score": 0.0,
            "typo_score": 0.0,
            "reasons": ["Scoring error occurred"],
        }


def _score_resume_impl(features: dict, raw_text: str, identity: dict) -> Dict[str, Any]:
    """Internal scoring implementation with enhanced logic."""
    from ml_engine.quality import typo_score
    
    logger.info("Scoring resume...")
    
    scores = {}
    reasons: List[str] = []
    penalty_score = 0
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: HARD PENALTY LAYER (Critical Issues)
    # ─────────────────────────────────────────────────────────────
    
    has_email = bool(identity.get("email"))
    has_phone = bool(identity.get("phone"))
    
    if not has_email and not has_phone:
        logger.info("No valid contact found - auto ATS score = 0")
        return {
            "ats_score": 0.0,
            "contact_score": 0.0,
            "structure_score": 0.0,
            "content_score": 0.0,
            "length_score": 0.0,
            "typo_score": 0.0,
            "reasons": ["Missing contact information (email and phone)"],
        }
    
    # Skills penalty
    skills_count = _safe(features, "skills_count", 0)
    if skills_count == 0:
        penalty_score += 40
        reasons.append("No skills detected (-40)")
        logger.info("No skills found - applying -40 penalty")
    
    # Projects penalty
    has_projects = _safe(features, "has_projects", 0)
    projects_count = _safe(features, "projects_count", 0)
    if has_projects == 0:
        penalty_score += 30
        reasons.append("No projects found (-30)")
        logger.info("No projects found - applying -30 penalty")
    
    # Education score penalty
    education_score_raw = float(_safe(features, "score", 0) or 0)
    if education_score_raw < 5:
        penalty_score += 10
        reasons.append("Weak education score (<5) (-10)")
        logger.info(f"Weak education score ({education_score_raw}) - applying -10 penalty")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: CONTACT SCORE
    # ─────────────────────────────────────────────────────────────
    
    contact_score = float(compute_contact_score(features, identity))
    
    if contact_score == 100:
        reasons.append("Complete contact information")
    elif contact_score == 50:
        reasons.append("Partial contact information")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: STRUCTURE SCORE (Enhanced with experience)
    # ─────────────────────────────────────────────────────────────
    
    has_skills_section = _safe(features, "has_skills_section", 0)
    has_education_section = _safe(features, "has_education_section", 0)
    has_projects_section = _safe(features, "has_projects_section", 0)
    has_experience_section = _safe(features, "has_experience_section", 0)
    
    structure_points = (
        has_skills_section +
        has_education_section +
        has_projects_section +
        has_experience_section
    )
    
    structure_score = (structure_points / 4) * 100
    
    if structure_points >= 3:
        reasons.append("Well-structured resume")
    elif structure_points <= 1:
        reasons.append("Poor resume structure")
    
    logger.info(f"Structure score: {structure_score:.1f} (sections: {structure_points}/4)")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: CONTENT SCORE (Dynamic, Enhanced)
    # ─────────────────────────────────────────────────────────────
    
    # Skill score with dynamic tiers
    if skills_count == 0:
        skill_score = 0
    elif skills_count <= 3:
        skill_score = 30
        reasons.append("Weak skills section (≤3 skills)")
    elif skills_count <= 8:
        skill_score = 60
        reasons.append("Average skills section")
    else:
        skill_score = 100
        reasons.append("Strong skills section (9+ skills)")
    
    logger.info(f"Skill score: {skill_score} (count: {skills_count})")
    
    # Project score with tiers
    if projects_count == 0:
        project_score = 0
    elif projects_count == 1:
        project_score = 60
        reasons.append("Limited project experience (1 project)")
    elif projects_count == 2:
        project_score = 80
        reasons.append("Good project experience (2 projects)")
    else:
        project_score = 100
        reasons.append(f"Strong project portfolio ({projects_count}+ projects)")
    
    logger.info(f"Project score: {project_score} (count: {projects_count})")
    
    # Education score
    if education_score_raw > 10:
        education_score = (education_score_raw / 100.0) * 100
    else:
        education_score = (education_score_raw / 10.0) * 100
    education_score = min(education_score, 100.0)
    
    # Experience score (NEW)
    has_experience = _safe(features, "has_experience", 0)
    has_internship = _safe(features, "has_internship", 0)
    
    if has_experience:
        experience_score = 100
        reasons.append("Professional experience found")
    elif has_internship:
        experience_score = 70
        reasons.append("Internship experience found")
    else:
        experience_score = 0
        reasons.append("No work experience")
    
    logger.info(f"Experience score: {experience_score}")
    
    # Combined content score with experience integration
    content_score = (
        (skill_score * 0.40) +
        (project_score * 0.30) +
        (education_score * 0.20) +
        (experience_score * 0.10)
    )
    
    # Experience bonus
    if has_experience:
        content_score = min(content_score + 10, 100)
    
    logger.info(f"Content score: {content_score:.1f}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: LENGTH SCORE (Enhanced with penalties)
    # ─────────────────────────────────────────────────────────────
    
    word_count = len(raw_text.split())
    
    if word_count < 120:
        length_score = 20
        reasons.append("Resume too short (<120 words)")
    elif word_count < 200:
        length_score = 40
    elif word_count < 350:
        length_score = 75
    elif word_count < 900:
        length_score = 100
    elif word_count < 1400:
        length_score = 85
    elif word_count > 2000:
        length_score = 50
        reasons.append("Resume too long (>2000 words)")
    else:
        length_score = 60
    
    logger.info(f"Length score: {length_score} (words: {word_count})")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 6: TYPO SCORE
    # ─────────────────────────────────────────────────────────────
    
    typo_quality = typo_score(raw_text)
    
    if typo_quality < 50:
        penalty_score += 10
        reasons.append("Poor writing quality (typos) (-10)")
        logger.info(f"Low typo score ({typo_quality}) - applying -10 penalty")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 7: CONSISTENCY CHECK
    # ─────────────────────────────────────────────────────────────
    
    # High skills but no projects → suspicious
    if skills_count >= 10 and projects_count == 0:
        penalty_score += 10
        reasons.append("Inconsistent: many skills but no projects (-10)")
        logger.info("Consistency check: high skills but no projects - applying -10 penalty")
    
    # Projects but no skills → suspicious
    if projects_count >= 2 and skills_count <= 3:
        penalty_score += 10
        reasons.append("Inconsistent: projects but few skills (-10)")
        logger.info("Consistency check: projects but few skills - applying -10 penalty")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 8: FINAL ATS SCORE CALCULATION
    # ─────────────────────────────────────────────────────────────
    
    ats_score = (
        contact_score * 0.20 +
        structure_score * 0.20 +
        content_score * 0.35 +
        length_score * 0.15 +
        typo_quality * 0.10
    )
    
    # Apply penalties
    ats_score = ats_score - penalty_score
    
    # Ensure bounds
    ats_score = max(0, min(100, ats_score))
    
    # ─────────────────────────────────────────────────────────────
    # STEP 9: CALIBRATION (Realistic Distribution)
    # ─────────────────────────────────────────────────────────────
    
    # Weak resumes should be 20-50
    if ats_score < 30 and penalty_score > 20:
        ats_score = min(ats_score, 35)
    
    # Strong resumes should be 75-90 (not 100)
    if ats_score > 90:
        ats_score = 90
    
    logger.info(f"Final ATS score: {ats_score:.1f} (penalties: {penalty_score})")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 10: BUILD RESULT
    # ─────────────────────────────────────────────────────────────
    
    scores["ats_score"] = round(ats_score, 2)
    scores["contact_score"] = round(contact_score, 2)
    scores["structure_score"] = round(structure_score, 2)
    scores["content_score"] = round(content_score, 2)
    scores["length_score"] = round(length_score, 2)
    scores["typo_score"] = round(typo_quality, 2)
    scores["reasons"] = reasons
    
    logger.info(f"Scoring complete. Reasons: {len(reasons)}")
    
    return scores
