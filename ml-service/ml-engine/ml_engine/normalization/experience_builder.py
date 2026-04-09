"""
Experience normalization builder.

Extracts and normalizes work experience from resume sections.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

_EXPERIENCE_NOISE_RE = re.compile(
    r"\b(hackathon|club|committee|volunteer|event|competition|co[- ]?curricular|extra[- ]?curricular|activity|activities)\b",
    re.IGNORECASE,
)


def build_experience(sections: Dict[str, List[str]]) -> List[str]:
    """
    Build normalized experience list from sections.
    
    Args:
        sections: Detected resume sections
        
    Returns:
        List of cleaned experience lines
    """
    logger.info("Starting experience extraction")
    
    if "experience" not in sections:
        logger.info("No experience section found")
        return []
    
    experience_lines = sections["experience"]
    if not experience_lines:
        logger.info("Experience section is empty")
        return []
    
    cleaned = clean_experience(experience_lines)
    logger.info(f"Extracted {len(cleaned)} experience entries")
    
    return cleaned


def clean_experience(experience_list: List[str]) -> List[str]:
    """Clean experience lines."""
    cleaned: List[str] = []
    seen = set()
    
    for line in experience_list:
        entry = line.strip()
        if not entry:
            continue
        low = entry.lower()
        if _EXPERIENCE_NOISE_RE.search(low):
            continue
        if not any(k in low for k in ("intern", "developer", "engineer")):
            continue
        if len(entry.split()) > 12:
            continue
        if any(p in entry for p in ".!?"):
            continue
        role = re.sub(r"\s+", " ", low).strip(" ,.-")
        if not role:
            continue
        if role in seen:
            continue
        cleaned.append(role)
        seen.add(role)
    
    return cleaned


def has_internship(experience: List[str]) -> bool:
    """Check if experience includes internship."""
    for line in experience:
        if "intern" in line.lower():
            return True
    return False


def estimate_experience_years(experience: List[str]) -> float:
    """Estimate years of experience from experience lines."""
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")
    years_found = set()
    
    for line in experience:
        matches = year_pattern.findall(line)
        for year in matches:
            years_found.add(int(year))
    
    if len(years_found) >= 2:
        return max(years_found) - min(years_found)
    
    return 0.0
