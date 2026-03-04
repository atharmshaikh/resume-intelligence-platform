"""
Central configuration for the ML resume engine.
Keeping all constants here prevents hardcoding values across modules.
"""

# Supported resume file formats
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

# Maximum allowed file size (MB)
MAX_FILE_SIZE_MB = 10

# Section headings used for resume parsing
SECTION_PATTERNS = {
    "skills": [
        "skills",
        "technical skills",
        "core competencies"
    ],
    "education": [
        "education",
        "academic background",
        "qualifications"
    ],
    "experience": [
        "experience",
        "work experience",
        "employment history"
    ],
    "projects": [
        "projects",
        "academic project",
        "academic projects"
    ],
    "languages": [
        "languages"
    ],
    "achievements": [
        "achievements",
        "achivements"
    ]
}