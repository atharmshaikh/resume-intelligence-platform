"""
Feature extraction module.

Converts parsed resume data into numerical features
that can later be used for machine learning models.
"""
import re
from ml_engine.schemas.resume_schema import ResumeSchema


def extract_features(resume: ResumeSchema) -> dict:
    """
    Generate ATS-ready numerical features from ResumeSchema.
    """

    features = {}

    # -----------------------------
    # Contact information features
    # -----------------------------

    features["has_name"] = int(resume.name is not None)
    features["has_email"] = int(resume.email is not None)
    features["has_phone"] = int(resume.phone is not None)
    features["has_location"] = int(resume.location is not None)

    # -----------------------------
    # Resume structure features
    # -----------------------------

    features["section_count"] = len(resume.sections)

    features["has_skills_section"] = int("skills" in resume.sections)
    features["has_education_section"] = int("education" in resume.sections)
    features["has_projects_section"] = int("projects" in resume.sections)
    features["has_experience_section"] = int("experience" in resume.sections)

    # -----------------------------
    # Skill strength
    # -----------------------------

    features["skills_count"] = len(resume.skills)

    # -----------------------------
    # Education
    # -----------------------------

    education_entries = 0

    for line in resume.education:
        if "bachelor" in line.lower() or "master" in line.lower() or "bca" in line.lower() or "mca" in line.lower():
            education_entries += 1

    features["education_count"] = education_entries

    # -----------------------------
    # Experience
    # -----------------------------

    features["experience_count"] = len(resume.experience)

    # -----------------------------
    # Resume length
    # -----------------------------

    word_count = len(re.findall(r"\b\w+\b", resume.raw_text))
    features["resume_word_count"] = word_count

    # -----------------------------
    # Projects
    # -----------------------------

    features["has_projects"] = int("projects" in resume.sections)

    return features