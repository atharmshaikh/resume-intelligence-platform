"""
ATS configuration for resume normalization.

This module defines:
• section keywords   – canonical section names + every real-world variant
• skill taxonomy     – CS/IT focused skill classification
• stop words         – strings to exclude from skill extraction
• normalization rules

Designed to handle:
  BE / B.Tech CS / IT / ECE / EE
  BCA / MCA
  BSC IT / MSC IT
  Any messy or non-standard resume layout
"""

# ---------------------------------------------------------------------------
# SECTION HEADINGS
# Covers every common phrasing found in Indian CS/IT resumes.
# Each entry maps to a canonical section key used by the pipeline.
# ---------------------------------------------------------------------------

from typing import Dict, List, Set

SECTION_KEYWORDS: Dict[str, List[str]] = {

    "career_objective": [
        "career objective",
        "objective",
        "professional objective",
        "career summary",
        "profile",
        "professional profile",
        "summary",
        "professional summary",
        "about me",
        "about",
        "introduction",
        "overview",
        "personal statement",
    ],

    "education": [
        "education",
        "educational background",
        "educational qualifications",
        "educational details",
        "academic background",
        "academic qualifications",
        "academic details",
        "qualifications",
        "education and training",
        "academic record",
        "scholastic record",
        "degrees",
    ],

    "skills": [
        "skills",
        "technical skills",
        "technical expertise",
        "core competencies",
        "core skills",
        "key skills",
        "it skills",
        "tech skills",
        "areas of expertise",
        "competencies",
        "expertise",
        "technologies",
        "technologies known",
        "tools and technologies",
        "programming skills",
        "software skills",
        "technical proficiencies",
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "employment",
        "professional background",
        "work history",
        "career history",
        "internships",
        "internship experience",
        "industrial experience",
        "industrial training",
        "training",
        "job experience",
        "relevant experience",
        "practical experience",
    ],

    "projects": [
        "projects",
        "academic projects",
        "academic project",
        "personal projects",
        "major projects",
        "mini projects",
        "technical projects",
        "project work",
        "project experience",
        "key projects",
        "notable projects",
        "software projects",
        "live projects",
        "recent projects",
    ],

    "achievements": [
        "achievements",
        "achivements",
        "awards",
        "awards and achievements",
        "honors",
        "honours",
        "accomplishments",
        "recognition",
        "certifications",
        "certificates",
        "certification",
        "certificate",
        "courses",
        "online courses",
        "workshops",
        "seminars",
        "extra curricular activities",
        "extracurricular activities",
        "co curricular activities",
        "activities",
        "competitions",
        "hackathons",
    ],

    "languages": [
        "languages",
        "known languages",
        "languages known",
        "language proficiency",
        "language skills",
    ],

    "interests": [
        "interests",
        "hobbies",
        "hobbies and interests",
        "personal interests",
    ],

    "declaration": [
        "declaration",
        "i hereby declare",
    ],
}

# ---------------------------------------------------------------------------
# SKILL CATEGORY HEADERS
# ---------------------------------------------------------------------------

SKILL_SECTION_HEADERS: Set[str] = {
    "technical skills",
    "soft skills",
    "languages",
}

# ---------------------------------------------------------------------------
# SKILL TAXONOMY for CS / BCA / MCA resumes
# ---------------------------------------------------------------------------

TECH_SKILLS: Set[str] = {
    "python", "java", "c", "c++", "c#", "javascript",
    "typescript", "php", "dart", "kotlin", "swift",
    "sql", "mysql", "postgresql", "mongodb", "sqlite",
    "html", "css", "react", "angular", "vue", "nodejs",
    "django", "flask", "fastapi", "spring", "laravel",
    "git", "docker", "linux", "aws", "azure",
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "data science", "excel", "powerpoint",
}

SOFT_SKILLS: Set[str] = {
    "communication", "team coordination", "time management",
    "leadership", "problem solving", "adaptability",
    "critical thinking", "presentation", "teamwork",
}

# ---------------------------------------------------------------------------
# STOP WORDS for skill extraction
# ---------------------------------------------------------------------------

SKILL_STOPWORDS: Set[str] = {
    "technical skills", "soft skills", "skills", "technologies",
    "areas of expertise", "tools and technologies",
}

# ---------------------------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------------------------

BULLET_CHARACTERS: List[str] = ["•", "-", "*", "▪", "◦", "▸", "►"]