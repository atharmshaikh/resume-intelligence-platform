"""
Detect major resume sections based on known headings.
"""

import re
from typing import Dict, List
from ml_engine.config.ats_config import SECTION_KEYWORDS

SECTION_PATTERNS = {
    section: [
        re.compile(rf"\b{re.escape(keyword)}\b")
        for keyword in keywords
    ]
    for section, keywords in SECTION_KEYWORDS.items()
}

MAX_SECTION_LINES = 500
MAX_HEADER_LENGTH = 60
HEADER_CLEAN_PATTERN = re.compile(r"[^a-z\s]")

def detect_sections(text: str) -> Dict[str, List[str]]:
    """
    Detect resume sections using keyword-based heading detection.

    Returns
    -------
    Dict[str, str]
        Mapping of section name → section content.
    """
    sections: Dict[str, List[str]] = {}

    lines = text.splitlines()

    current_section = None
    buffer = []

    # Iterate line-by-line to detect section headers
    for line in lines:

        normalized = HEADER_CLEAN_PATTERN.sub(" ", line.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        matched_section = None
        
        # Skip long lines that are unlikely to be headers
        if len(normalized) > MAX_HEADER_LENGTH:
            matched_section = None
        else:

            for section, patterns in SECTION_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(normalized):
                        matched_section = section
                        break
                if matched_section:
                    break

        if matched_section:

            if current_section and buffer:
                content = [
                    line.strip()
                    for line in buffer
                    if line.strip()
                ]
                if content and current_section not in sections:
                    sections[current_section] = content

            current_section = matched_section
            buffer = []
            continue

        if current_section:
            if len(buffer) < MAX_SECTION_LINES and len(line) < 2000:
                buffer.append(line)

    if current_section and buffer:
        sections[current_section] = [
            line.strip()
            for line in buffer
            if line.strip()
        ]

    sections = {
        key: value
        for key, value in sections.items()
        if value
    }

    return sections