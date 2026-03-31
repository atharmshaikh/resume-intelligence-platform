"""
Central configuration for the ML resume engine.
Keeping all constants here prevents hardcoding values across modules.
"""

from typing import Set, Dict, List

# Supported resume file formats
SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".docx"}

# Maximum allowed file size (MB)
MAX_FILE_SIZE_MB: int = 10

# Section headings used for resume parsing
SECTION_PATTERNS: Dict[str, List[str]] = {
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