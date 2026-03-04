"""
ATS configuration for resume normalization.

This module defines:
• section keywords
• skill taxonomy
• stop words
• normalization rules

Focused for Computer Science / BCA / MCA resumes.
"""

# -------------------------------
# SECTION HEADINGS
# -------------------------------

SECTION_KEYWORDS = {
    "career_objective": [
        "career objective",
        "objective",
        "professional objective",
    ],
    "projects": [
        "projects",
        "academic project",
        "academic projects",
    ],
    "education": [
        "education",
        "academic background",
        "qualifications",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
    ],
    "experience": [
        "experience",
        "work experience",
        "employment history",
    ],
    "achievements": [
        "achievements",
        "achivements",
        "awards",
    ],
    "languages": [
        "languages",
    ],
    "declaration": [
        "declaration"
    ]
}

# -------------------------------
# SKILL CATEGORY HEADERS
# -------------------------------

SKILL_SECTION_HEADERS = {
    "technical skills",
    "soft skills"
}

# -------------------------------
# SKILL TAXONOMY (CS / BCA / MCA)
# -------------------------------

TECH_SKILLS = {
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "html",
    "css",
    "javascript",
    "excel",
    "powerpoint",
    "ms word",
    "data entry",
    "database",
}

SOFT_SKILLS = {
    "communication",
    "team coordination",
    "time management",
    "leadership",
    "problem solving",
}

# -------------------------------
# STOP WORDS
# -------------------------------

SKILL_STOPWORDS = {
    "technical skills",
    "soft skills",
    "skills",
}

# -------------------------------
# TEXT NORMALIZATION
# -------------------------------

BULLET_CHARACTERS = ["•", "-", "*", "▪"]