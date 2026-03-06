"""
Feature extraction module.

Converts parsed resume data into numerical features
that can later be used for machine learning models.
"""


import re
from ml_engine.schemas.resume_schema import ResumeSchema
from ml_engine.extraction.keyword_loader import load_wordlist

WORD_PATTERN = re.compile(r"\b\w+\b")

# Safe wordlist loader to avoid pipeline crash
def _safe_load_wordlist(filename: str):
    try:
        return load_wordlist(filename)
    except FileNotFoundError:
        return set()

# -----------------------------
# Feature keyword wordlists
# -----------------------------

DEGREE_KEYWORDS = set(_safe_load_wordlist("academic_terms.txt"))

HIGH_VALUE_SKILLS = set(_safe_load_wordlist("high_value_skills.txt"))
MID_VALUE_SKILLS = set(_safe_load_wordlist("mid_value_skills.txt"))
COMMON_LANGUAGES = set(_safe_load_wordlist("common_languages.txt"))

def extract_features(resume: ResumeSchema) -> dict:
    """
    Generate ATS-ready numerical features from ResumeSchema.
    """

    features = {}

    skills = resume.skills or []
    education = resume.education or []
    experience = resume.experience or []
    projects = resume.projects or []
    achievements = resume.achievements or []
    languages = resume.languages or []
    sections = resume.sections or {}
    raw_text = resume.raw_text or ""

    # Safety guard: limit extremely large text to prevent heavy regex processing
    raw_text = raw_text[:100000]

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

    features["section_count"] = len(sections)

    features["has_skills_section"] = int("skills" in sections)
    features["has_education_section"] = int("education" in sections)
    features["has_projects_section"] = int("projects" in sections)
    features["has_experience_section"] = int("experience" in sections)

    # -----------------------------
    # Skill strength
    # -----------------------------

    features["skills_count"] = len(skills)

    # -----------------------------
    # Weighted skill strength
    # -----------------------------

    skill_weight_score = 0.0

    for skill in skills:

        if not isinstance(skill, str):
            continue

        s = skill.lower().strip()
        s = s.replace("-", " ")

        if s in HIGH_VALUE_SKILLS:
            skill_weight_score += 2.0

        elif s in MID_VALUE_SKILLS:
            skill_weight_score += 1.0

        else:
            skill_weight_score += 0.5

    features["skill_weight_score"] = round(skill_weight_score, 2)

    # -----------------------------
    # Education
    # -----------------------------

    education_entries = 0

    for line in education:
        if not isinstance(line, str):
            continue

        text = line.lower()

        if any(keyword in text for keyword in DEGREE_KEYWORDS):
            education_entries += 1

    features["education_count"] = education_entries

    # -----------------------------
    # Experience
    # -----------------------------

    features["experience_count"] = len(experience)

    # -----------------------------
    # Resume length
    # -----------------------------

    word_count = len(re.findall(r"\b\w+\b", raw_text))
    features["resume_word_count"] = word_count

    # -----------------------------
    # Developer profile links
    # -----------------------------

    text_lower = raw_text.lower()

    features["has_github"] = int("github.com" in text_lower)
    features["has_linkedin"] = int("linkedin.com" in text_lower)    

    # -----------------------------
    # Projects
    # -----------------------------

    features["has_projects"] = int("projects" in sections)
    features["projects_count"] = len(projects)

    # -----------------------------
    # Achievements
    # -----------------------------

    features["achievement_count"] = len(achievements)

    features["has_achievements"] = int(features["achievement_count"] > 0)

    # -----------------------------
    # Language diversity
    # -----------------------------

    extra_languages = 0

    for lang in languages:

        if not isinstance(lang, str):
            continue

        if lang.lower() not in COMMON_LANGUAGES:
            extra_languages += 1

    features["extra_language_count"] = extra_languages

    return features