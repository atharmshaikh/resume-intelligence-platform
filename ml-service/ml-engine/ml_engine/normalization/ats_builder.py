"""
ATS normalization engine.

Transforms extracted sections into structured ATS fields.
Designed to be robust across resume layouts.
"""

import logging
import re
from ml_engine.schemas import ResumeSchema
from ml_engine.extraction import load_wordlist

logger = logging.getLogger(__name__)

# -------------------------------
# Utility loaders
# -------------------------------

def load_normalization_map(filename: str) -> dict[str, str]:
    """
    Load normalization mappings like:
    cpp:C++
    js:JavaScript
    """
    mapping: dict[str, str] = {}

    for line in load_wordlist(filename):

        if ":" not in line:
            continue

        src, dst = line.split(":", 1)

        mapping[src.strip().lower()] = dst.strip()

    return mapping

# -------------------------------
# Wordlists
# -------------------------------

LOCATION_STOPWORDS = set(load_wordlist("locations.txt"))
PROGRAMMING_LANGUAGES = set(load_wordlist("programming_languages.txt"))
FRAMEWORKS = set(load_wordlist("frameworks.txt"))
DATABASES = set(load_wordlist("databases.txt"))
TOOLS = set(load_wordlist("tools.txt"))
TECH_TERMS = set(load_wordlist("tech_terms.txt"))

COMMON_HEADERS = set(load_wordlist("common_headers.txt"))
RESUME_TERMS = set(load_wordlist("resume_terms.txt"))

SKILL_SECTION_NAMES = set(load_wordlist("section_keywords.txt"))

SKILL_NORMALIZATION = load_normalization_map("skill_normalization.txt")

# Combined skill whitelist
SKILL_WHITELIST = {
    s.lower()
    for s in (
        PROGRAMMING_LANGUAGES
        | FRAMEWORKS
        | DATABASES
        | TOOLS
        | TECH_TERMS
    )
}

# -------------------------------
# Constants
# -------------------------------

BULLET_CHARACTERS = {
    "•", "●", "▪", "◦", "►", "▸", "■", "□", "◆", "◇",
    "-", "*"
}


MAX_SECTION_LINES = 800
MAX_SKILLS = 200

# -------------------------------
# Text utilities
# -------------------------------

def _normalize_bullets(text: str) -> str:

    for bullet in BULLET_CHARACTERS:
        text = text.replace(bullet, "\n")

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

        # remove URLs
        if "http" in line.lower():
            continue

        if "github.com" in line.lower():
            continue

        if "linkedin.com" in line.lower():
            continue

        if len(line) > 1000:
            continue

        lines.append(line)

        if len(lines) >= MAX_SECTION_LINES:
            break

    return lines

# -------------------------------
# Skill extraction
# -------------------------------

def _extract_skills(text: str):

    # Convert list sections to text
    if isinstance(text, list):
        text = "\n".join(text)

    text = _normalize_bullets(text)

    tokens = re.split(r"[,\n|/;&]", text)

    skills = []
    seen = set()

    for token in tokens:

        skill = token.strip()

        if not skill:
            continue

        skill = skill.replace("(", "").replace(")", "")
        skill = re.sub(r"[–—−]", "-", skill)
        # replace dash only when used as separator
        skill = re.sub(r"\s-\s", " ", skill)

        skill_lower = skill.lower().strip()
        
        # Normalize first
        skill_norm = SKILL_NORMALIZATION.get(skill_lower, skill)
        skill_norm_lower = skill_norm.lower()

        # Ignore years/dates
        if re.search(r"\d{4}", skill):
            continue

        # ignore locations
        if skill.lower() in LOCATION_STOPWORDS:
            continue
        
        # ignore headers
        if skill_lower in COMMON_HEADERS:
            continue

        if skill_lower in RESUME_TERMS:
            continue

        # Remove degree words incorrectly captured as skills
        if skill_lower in {"bachelor", "btech", "b.tech", "diploma", "degree"}:
            continue

        # ignore numbers
        if skill.isdigit():
            continue

        # ignore long sentences
        if len(skill.split()) > 2:
            continue

        # whitelist validation
        if skill_norm_lower not in SKILL_WHITELIST:
            if not any(part in SKILL_WHITELIST for part in skill_norm_lower.split()):
                continue

        if skill_norm_lower in seen:
            continue

        skills.append(skill_norm)

        seen.add(skill_norm_lower)

        if len(skills) >= MAX_SKILLS:
            break

    return skills

# -------------------------------
# Project extraction
# -------------------------------

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

        # skip section headers
        if line.strip().lower() in COMMON_HEADERS:
            continue

        if line.endswith(":") or line.startswith("-"):
            continue

        # Treat short lines as project titles
        if 5 < len(line) < 60 and line.count(" ") <= 6 and not line.startswith("-"):
            if buffer:
                projects.append(" ".join(buffer))
                buffer = []
            buffer.append(line)
        else:
            buffer.append(line)

    if buffer:
        projects.append(" ".join(buffer))

    return projects

# -------------------------------
# ATS structure builder
# -------------------------------

def build_ats_structure(raw_text: str, sections: dict) -> ResumeSchema:
    """
    Convert detected sections into ATS structured resume.
    """

    if not isinstance(sections, dict):
        logger.warning(f"Invalid sections type provided: {type(sections)}. Resetting to empty dict.")
        sections = {}

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

    # If no skills found, look for alternate section names (only then)
    if not resume.skills:
        for section_name, content in sections.items():
            if section_name.lower() in SKILL_SECTION_NAMES:
                resume.skills = _extract_skills(content)
                break
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
    
    # Infer skills from project descriptions if skills section is weak
    if resume.projects:
        project_text = " ".join(resume.projects).lower()

        existing_skills = {s.lower() for s in resume.skills}

        for skill in SKILL_WHITELIST:

            if skill in existing_skills:
                continue

            if re.search(rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])", project_text):

                resume.skills.append(skill)

                existing_skills.add(skill)

    # -----------------------
    # ACHIEVEMENTS
    # -----------------------

    if "achievements" in sections:
        resume.achievements = _clean_lines(sections["achievements"])

    # -----------------------
    # LANGUAGES
    # -----------------------

    if "languages" in sections:
        lang_lines = _clean_lines(sections["languages"])

        languages = []
        for line in lang_lines:
            parts = re.split(r"[,\s/]+", line)
            languages.extend(p for p in parts if p)

        resume.languages = languages

    # -----------------------
    # UNKNOWN SECTION (AI CLASSIFIER FALLBACK)
    # -----------------------
    for sec_name, content in list(sections.items()):
        if sec_name.startswith("unknown_"):
            content_str = " ".join(content).lower()
            
            # Heuristic 1: Experience (dates + length)
            if bool(re.search(r"20\d{2}|19\d{2}", content_str)) and len(content) > 2:
                if not resume.experience:
                    resume.experience = _clean_lines(content)
                continue
                
            # Heuristic 2: Skills (dense tech keywords)
            tech_hits = sum(1 for skill in SKILL_WHITELIST if skill in content_str)
            if tech_hits >= 3 and not resume.skills:
                resume.skills = _extract_skills(content)
                continue
                
            # Heuristic 3: Projects
            if ("project" in sec_name or "system" in content_str) and not resume.projects:
                resume.projects = _extract_projects(content)
                continue

    return resume