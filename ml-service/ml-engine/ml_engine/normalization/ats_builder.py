"""
ATS normalization engine.

Transforms extracted sections into structured ATS fields.
Designed to be robust across resume layouts.
"""

import re

from ml_engine.schemas.resume_schema import ResumeSchema
from ml_engine.config.ats_config import (
    BULLET_CHARACTERS,
    SKILL_SECTION_HEADERS,
    SKILL_STOPWORDS,
)


def _normalize_bullets(text: str) -> str:
    """
    Replace bullet characters with newline for easier splitting.
    """

    for bullet in BULLET_CHARACTERS:
        text = text.replace(bullet, "\n")

    return text


def _clean_lines(text: str):

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        lines.append(line)

    return lines


def _extract_skills(text: str):

    text = _normalize_bullets(text)

    tokens = re.split(r"\n|,|\|", text)

    skills = []

    for token in tokens:

        skill = token.strip()

        if not skill:
            continue

        skill_lower = skill.lower()

        if skill_lower in SKILL_STOPWORDS:
            continue

        if skill_lower in SKILL_SECTION_HEADERS:
            continue

        if len(skill) < 2:
            continue

        skills.append(skill)

    return skills


def build_ats_structure(raw_text: str, sections: dict) -> ResumeSchema:
    """
    Convert detected sections into ATS structured resume.
    """

    resume = ResumeSchema()

    resume.raw_text = raw_text
    resume.sections = sections

    # -----------------------
    # SKILLS
    # -----------------------

    if "skills" in sections:

        resume.skills = _extract_skills(sections["skills"])

    # -----------------------
    # EDUCATION
    # -----------------------

    if "education" in sections:

        resume.education = _clean_lines(sections["education"])

    # -----------------------
    # EXPERIENCE
    # -----------------------

    if "experience" in sections:

        resume.experience = _clean_lines(sections["experience"])

    return resume