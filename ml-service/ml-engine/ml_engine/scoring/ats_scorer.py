"""
Rule-based ATS scoring system.

Produces interpretable resume quality scores.
"""
from ml_engine.quality.typo_checker import typo_score

def score_resume(features: dict, raw_text: str) -> dict:

    scores = {}

    # -------------------------
    # Contact score
    # -------------------------

    contact_score = (
        features["has_name"]
        + features["has_email"]
        + features["has_phone"]
        + features["has_location"]
    ) / 4 * 100

    # -------------------------
    # Structure score
    # -------------------------

    structure_points = (
        features["has_skills_section"]
        + features["has_education_section"]
        + features["has_projects_section"]
    )

    structure_score = structure_points / 3 * 100

    # -------------------------
    # Content score
    # -------------------------

    skill_score = min(features["skills_count"] / 10, 1)

    project_score = features["has_projects"]

    education_score = min(features["education_count"], 1)

    content_score = (skill_score + project_score + education_score) / 3 * 100

    # -------------------------
    # Resume length score
    # -------------------------

    word_count = features["resume_word_count"]

    if word_count < 200:
        length_score = 50
    elif word_count < 400:
        length_score = 80
    elif word_count < 900:
        length_score = 100
    else:
        length_score = 70

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

    scores["ats_score"] = round(ats_score, 2)
    scores["contact_score"] = round(contact_score, 2)
    scores["structure_score"] = round(structure_score, 2)
    scores["content_score"] = round(content_score, 2)
    scores["length_score"] = round(length_score, 2)
    scores["typo_score"] = round(typo_quality, 2)
    
    return scores