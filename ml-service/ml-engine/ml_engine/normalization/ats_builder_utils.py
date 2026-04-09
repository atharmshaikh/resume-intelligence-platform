"""
ATS normalization utilities.

Shared low-level utilities for normalization.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ml_engine.extraction import load_wordlist

logger = logging.getLogger(__name__)

# Common headers
COMMON_HEADERS = set(load_wordlist("common_headers.txt") or [])

# Experience noise patterns
_EXPERIENCE_NOISE_RE = re.compile(
    r"\b(hackathon|club|committee|volunteer|event|competition|co[- ]?curricular|extra[- ]?curricular|activity|activities)\b",
    re.IGNORECASE,
)

# Location stopwords
LOCATION_STOPWORDS = set(load_wordlist("locations.txt") or [])

# Email/URL/Phone patterns
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?:https?://|www\.|github\.com|linkedin)", re.IGNORECASE)
_ORG_LOCATION_RE = re.compile(
    r"\b(software|technologies|technology|solutions|systems|services|infotech|company|pvt|private|ltd|llp|inc|corp|labs|studio)\b",
    re.IGNORECASE,
)


def _clean_lines(text: str | List[str]) -> List[str]:
    """Clean and normalize text lines."""
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
    return lines


def clean_experience(experience_list: List[str]) -> List[str]:
    """Clean experience lines."""
    cleaned: List[str] = []
    seen: set = set()
    
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


def clean_project_name(name: str) -> str:
    """Clean project name."""
    from ml_engine.normalization.project_extractor import clean_project_name as _clean
    return _clean(name)


def _as_plain_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    """Convert typed dicts to plain dicts."""
    return [dict(item) if hasattr(item, '_asdict') else item for item in items]


def _canonicalize_sections(sections: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Canonicalize section names."""
    aliases = {
        "career_objective": "summary",
        "objective": "summary",
        "profile": "summary",
        "certifications": "achievements",
        "certification": "achievements",
    }
    
    result: Dict[str, List[str]] = {}
    for name, content in sections.items():
        canonical = aliases.get(name, name)
        if content:
            result[canonical] = content
    return result


def _coerce_sections_map(sections: Dict[Any, Any]) -> Dict[str, List[str]]:
    """Coerce sections to proper type."""
    result: Dict[str, List[str]] = {}
    for key, value in sections.items():
        if isinstance(key, str):
            if isinstance(value, list):
                result[key] = [str(v) for v in value if v]
            elif isinstance(value, str):
                result[key] = [value] if value else []
    return result


def _clean_achievement_lines(lines: List[str]) -> List[str]:
    """Clean achievement lines."""
    return [line.strip() for line in lines if line.strip()]


def _clean_summary_lines(lines: List[str]) -> List[str]:
    """Clean summary lines."""
    return [line.strip() for line in lines if line.strip()]


def _clean_interest_lines(lines: List[str]) -> List[str]:
    """Clean interest lines."""
    return [line.strip() for line in lines if line.strip()]


def fix_location(location: str) -> Optional[str]:
    """Clean and validate location string."""
    loc = str(location or "").strip()
    if not loc:
        return None
    if "(" in loc or ")" in loc:
        return None
    low = loc.lower()
    if _EMAIL_RE.search(loc) or _URL_RE.search(loc):
        return None
    if _ORG_LOCATION_RE.search(low):
        return None
    if len(loc.split()) > 6:
        return None
    if loc.lower() in LOCATION_STOPWORDS:
        return None
    return loc


def recompute_section_flags(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute section flags in feature data."""
    if not isinstance(data, dict):
        return data
    
    normalized = data.get("normalized_resume", {})
    features = data.get("features", {})
    
    if not isinstance(normalized, dict) or not isinstance(features, dict):
        return data
    
    section_to_flag = {
        "skills": "has_skills_section",
        "education": "has_education_section",
        "experience": "has_experience_section",
        "projects": "has_projects_section",
        "achievements": "has_achievements_section",
        "languages": "has_languages_section",
        "interests": "has_interests_section",
        "summary": "has_objective_section",
    }
    
    def _has_content(section_name: str) -> bool:
        value = normalized.get(section_name, [])
        if isinstance(value, list):
            return any(str(item).strip() for item in value)
        if isinstance(value, dict):
            return any(str(item).strip() for item in value.values())
        return bool(str(value).strip())
    
    for section_name, flag_name in section_to_flag.items():
        present = _has_content(section_name)
        features[flag_name] = 1 if present else 0
    
    features["section_count"] = sum(int(features.get(flag, 0)) for flag in section_to_flag.values())
    key_sections = (
        int(features.get("has_skills_section", 0))
        + int(features.get("has_education_section", 0))
        + int(features.get("has_experience_section", 0))
        + int(features.get("has_projects_section", 0))
    )
    features["section_completeness_score"] = round(float(key_sections) / 4.0, 2)
    
    data["normalized_resume"] = normalized
    data["features"] = features
    return data
