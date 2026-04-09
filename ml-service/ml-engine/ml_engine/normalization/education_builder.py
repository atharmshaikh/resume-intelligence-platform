"""
Education normalization builder.

Extracts and normalizes education information from resume sections.
"""

import logging
import re
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)

# Education-related patterns
_CGPA_KEYED_RE = re.compile(
    r"(?:cgpa|gpa|cpi)\s*[:\-]?\s*([+-]?\d+(?:[.,]\d+)?)\s*(?:/\s*(10|100)|%)?",
    re.IGNORECASE,
)

INSTITUTION_HINTS = (
    "institute", "institution", "college", "university", "school",
    "polytechnic", "academy", "campus",
)
DEGREE_HINTS = (
    "b.tech", "btech", "b.e", "be ", "bachelor", "diploma", "master",
    "m.tech", "mca", "bca", "bsc", "msc", "course:", "cgpa", "percentage",
)


class EducationDetail(TypedDict):
    """Education detail structure."""
    degree_type: str
    is_it_candidate: int
    score: float


def build_education(sections: Dict[str, List[str]]) -> List[EducationDetail]:
    """
    Build normalized education list from sections.
    
    Args:
        sections: Detected resume sections
        
    Returns:
        List of education dictionaries
    """
    logger.info("Starting education extraction")
    
    if "education" not in sections:
        logger.info("No education section found")
        return []
    
    education_lines = sections["education"]
    if not education_lines:
        logger.info("Education section is empty")
        return []
    
    education_details = _extract_education_details(education_lines)
    logger.info(f"Extracted {len(education_details)} education entries")
    
    return education_details


def _extract_education_details(lines: List[str]) -> List[EducationDetail]:
    """Extract education details from lines."""
    blob = " ".join(lines)
    lowered = blob.lower()
    
    degree_type = "unknown"
    if any(k in lowered for k in ("master", "m.tech", "mca", "msc", "m.sc")):
        degree_type = "master"
    elif any(k in lowered for k in ("bachelor", "b.tech", "btech", "b.e", "be ", "engineering")):
        degree_type = "bachelor"
    elif "diploma" in lowered:
        degree_type = "diploma"
    
    is_it_candidate = int(
        any(
            k in lowered
            for k in (
                "information technology",
                "computer engineering",
                "computer science",
                "information science",
                "it",
                "cs",
                "cse",
            )
        )
    )
    
    score = 0.0
    cgpa_val, scale = _extract_cgpa_from_blob(blob)
    if cgpa_val:
        try:
            val = float(cgpa_val)
            # Normalize to 100-point scale for ML consistency
            if scale == 10 or (val <= 10.0 and not scale):
                score = val * 10.0
            else:
                score = val
        except ValueError:
            score = 0.0
    
    return [{
        "degree_type": degree_type,
        "is_it_candidate": is_it_candidate,
        "score": round(score, 2),
    }]


def _extract_cgpa_from_blob(blob: str) -> tuple:
    """Extract CGPA from text blob."""
    for match in _CGPA_KEYED_RE.finditer(blob):
        value = match.group(1).replace(",", ".")
        scale_group = match.group(2)
        try:
            val = float(value)
        except ValueError:
            continue
        if val < 0:
            val = abs(val)
            value = str(val)
        if val == 0:
            continue
        if scale_group:
            scale = int(scale_group)
            if (scale == 10 and val <= 10) or (scale == 100 and val <= 100):
                return value, scale
            continue
        if "%" in match.group(0):
            if val <= 100:
                return value, 100
            continue
        if val <= 10:
            return value, 10
    
    return "", None


def get_primary_education(education: List[EducationDetail]) -> EducationDetail:
    """Get primary (first/most recent) education entry."""
    if not education:
        return {
            "degree_type": "unknown",
            "is_it_candidate": 0,
            "score": 0.0,
        }
    
    return education[0]


def is_it_candidate(education: List[EducationDetail]) -> bool:
    """Check if candidate is from IT field."""
    if not education:
        return False
    
    return bool(education[0].get("is_it_candidate", 0))


def get_degree_type(education: List[EducationDetail]) -> str:
    """Get degree type from education."""
    if not education:
        return "unknown"
    
    return education[0].get("degree_type", "unknown")


def get_cgpa(education: List[EducationDetail]) -> float:
    """Get CGPA/score from education."""
    if not education:
        return 0.0
    
    return float(education[0].get("score", 0.0))
