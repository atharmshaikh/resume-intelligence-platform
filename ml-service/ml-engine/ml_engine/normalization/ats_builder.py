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
MAX_SECTION_LINES = 800
MAX_SKILLS = 200

def _normalize_bullets(text: str) -> str:
    """
    Normalize bullet characters and separators.
    """

    for bullet in BULLET_CHARACTERS:
        text = text.replace(bullet, "\n")

    # Normalize common separators
    text = text.replace("•", "\n")
    text = text.replace("|", "\n")
    text = text.replace(";", "\n")

    return text

def _clean_lines(text):
    """
    Clean and normalize section lines.
    """
    if isinstance(text, list):
        return [line.strip() for line in text if line.strip()]
    
    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if len(line) > 1000:
            continue
        lines.append(line)

        if len(lines) >= MAX_SECTION_LINES:
            break

    return lines

def _extract_skills(text: str):
    """
    Extract normalized skill tokens.
    """
    # Convert list sections to text
    if isinstance(text, list):
        text = "\n".join(text)

    text = _normalize_bullets(text)

    tokens = re.split(r"\n|,|\||/", text)

    skills = []
    seen = set()

    for token in tokens:

        skill = token.strip()

        # Ignore years/dates
        if re.search(r"\d{4}", skill):
            continue

        # Ignore common location tokens
        LOCATION_STOPWORDS = {"anand", "bharuch", "gujarat", "india"}

        if skill.lower() in LOCATION_STOPWORDS:
            continue

        # Normalize common skill variants
        skill = skill.replace("My SQL", "MySQL")
        skill = skill.replace("CPP", "C++")
        skill = skill.replace("JS", "JavaScript")

        # Ignore pure numbers
        if skill.isdigit():
            continue

        # Ignore dates
        if re.search(r"\d{4}", skill):
            continue

        # Ignore pure locations
        if re.match(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$", skill) and len(skill) > 10:
            continue

        if not skill:
            continue

        skill_lower = skill.lower()

        if skill_lower in SKILL_STOPWORDS:
            continue

        if skill_lower in SKILL_SECTION_HEADERS:
            continue

        HEADER_STOPWORDS = {"professional", "experience", "education", "projects"}

        if skill_lower in HEADER_STOPWORDS: 
            continue

        if len(skill) < 2:
            continue

        # Skip long sentences (likely descriptions)
        if len(skill.split()) > 4:
            continue

        if skill_lower in seen:
            continue

        # Normalize common skill tokens
        NORMALIZATION = {
            "cpp": "C++",
            "c++": "C++",
            "mysql": "MySQL",
            "html": "HTML",
            "css": "CSS",
        }

        skill_norm = NORMALIZATION.get(skill_lower, skill)

        skills.append(skill_norm)   
        seen.add(skill_lower)

        if len(skills) >= MAX_SKILLS:
            break

    return skills

def _extract_projects(text: str):
    """
    Extract project descriptions.
    """
    if isinstance(text, list):
        text = "\n".join(text)

    lines = _clean_lines(text)

    projects = []

    buffer = []

    for line in lines:

        if len(line) < 3:
            continue

        if line.endswith(":") or line.startswith("-"):
            continue

        # Treat short lines as project titles
        if len(line) < 60 and not line.startswith("-"):
            if buffer:
                projects.append(" ".join(buffer))
                buffer = []
            buffer.append(line)
        else:
            buffer.append(line)

    if buffer:
        projects.append(" ".join(buffer))

    return projects

def build_ats_structure(raw_text: str, sections: dict) -> ResumeSchema:
    """
    Convert detected sections into ATS structured resume.
    """

    resume = ResumeSchema()

    resume.raw_text = raw_text
    resume.sections = sections

    resume.skills = []
    resume.education = []
    resume.experience = []
    resume.projects = []

    # -----------------------
    # SKILLS
    # -----------------------

    if "skills" in sections:
        resume.skills = _extract_skills(sections["skills"])
    for alt in ("technical_skills", "core_skills"):

        if not resume.skills and alt in sections:
            resume.skills = _extract_skills(sections[alt])
    
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

    # -----------------------
    # PROJECTS
    # -----------------------

    if "projects" in sections:
        resume.projects = _extract_projects(sections["projects"])
    
    # -----------------------
    # ACHIEVEMENTS
    # -----------------------

    if "achievements" in sections:
        resume.achievements = _clean_lines(sections["achievements"])

    # -----------------------
    # LANGUAGES
    # -----------------------

    if "languages" in sections:
        resume.languages = _clean_lines(sections["languages"])

    return resume