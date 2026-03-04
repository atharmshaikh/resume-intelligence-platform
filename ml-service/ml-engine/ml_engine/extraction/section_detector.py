"""
Detect major resume sections based on known headings.
"""

import re
from typing import Dict

from ml_engine.config.ats_config import SECTION_KEYWORDS

def detect_sections(text: str) -> Dict[str, str]:

    sections = {}

    lines = text.split("\n")

    current_section = None
    buffer = []

    for line in lines:

        normalized = line.lower().strip()

        matched_section = None

        for section, keywords in SECTION_KEYWORDS.items():
            for keyword in keywords:
                if re.fullmatch(rf"{keyword}", normalized):
                    matched_section = section
                    break
            if matched_section:
                break

        if matched_section:

            if current_section and buffer:
                sections[current_section] = "\n".join(buffer).strip()

            current_section = matched_section
            buffer = []
            continue

        if current_section:
            buffer.append(line)

    if current_section and buffer:
        sections[current_section] = "\n".join(buffer).strip()

    return sections