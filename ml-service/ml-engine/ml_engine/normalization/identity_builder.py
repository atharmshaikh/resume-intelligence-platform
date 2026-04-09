"""
Identity normalization builder.

Extracts and normalizes candidate identity information:
- Name
- Email
- Phone
- Location
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ml_engine.extraction import load_wordlist

logger = logging.getLogger(__name__)

# Regex patterns
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-]{8,13}\d)")
_URL_RE = re.compile(r"(?:https?://|www\.|github\.com|linkedin)", re.IGNORECASE)
_ORG_LOCATION_RE = re.compile(
    r"\b(software|technologies|technology|solutions|systems|services|infotech|company|pvt|private|ltd|llp|inc|corp|labs|studio)\b",
    re.IGNORECASE,
)

# Location stopwords
LOCATION_STOPWORDS: set[str] = set(load_wordlist("locations.txt") or [])


def extract_identity(sections: Dict[str, List[str]], entities: Dict[str, Any], raw_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract identity information from sections and entities.

    Args:
        sections: Detected resume sections
        entities: Extracted entities (name, email, phone, location)
        raw_text: Optional raw text for fallback name extraction

    Returns:
        Dictionary with normalized identity fields
    """
    logger.info("Starting identity extraction")

    identity = {
        "name": entities.get("name"),
        "email": entities.get("email"),
        "phone": entities.get("phone"),
        "location": entities.get("location"),
    }

    # Fallback: try to extract from sections if entities missing
    if not identity["name"]:
        identity["name"] = _extract_name_from_sections(sections)
    
    # Final fallback: extract from raw text first few lines
    if not identity["name"] and raw_text:
        identity["name"] = _extract_name_from_raw_text(raw_text)

    if not identity["email"]:
        identity["email"] = _extract_email_from_sections(sections)

    if not identity["phone"]:
        identity["phone"] = _extract_phone_from_sections(sections)

    if not identity["location"]:
        identity["location"] = _extract_location_from_sections(sections)

    logger.info(f"Identity extracted: name={identity['name']}, email={identity['email']}")
    return identity


def _extract_name_from_sections(sections: Dict[str, List[str]]) -> Optional[str]:
    """Extract name from first few lines of resume sections."""
    # Only check first section (should be header/summary)
    first_section = None
    for section_name, lines in sections.items():
        if section_name in ('summary', 'career_objective', 'profile'):
            first_section = lines
            break
    
    if not first_section:
        # Fallback to first available section
        for section_name, lines in sections.items():
            first_section = lines
            break
    
    if not first_section:
        return None
    
    for line in first_section[:3]:
        line = line.strip()
        if not line or len(line) > 50:
            continue
        if '@' in line or any(c.isdigit() for c in line):
            continue
        # Must be 2-4 words, all alphabetic (with allowed punctuation)
        tokens = line.split()
        if not (2 <= len(tokens) <= 4):
            continue
        # All tokens must be alphabetic (allowing apostrophes, hyphens)
        if not all(t.replace("'", "").replace("-", "").replace("`", "").isalpha() for t in tokens):
            continue
        # Skip if looks like a section header or bullet point
        if line.startswith('-') or line.startswith('•'):
            continue
        # Skip common non-name patterns
        lower = line.lower()
        if any(kw in lower for kw in ['participated', 'bachelor', 'master', 'degree', 'course', 'project', 'home', 'service', 'provider', 'team', 'coll']):
            continue
        logger.debug(f"Found potential name in section: {line}")
        return line.strip()
    
    return None


def _extract_name_from_raw_text(raw_text: str) -> Optional[str]:
    """Extract name from first few lines of raw text."""
    lines = raw_text.splitlines()
    for line in lines[:5]:
        line = line.strip()
        if not line or len(line) > 50:
            continue
        if '@' in line or any(c.isdigit() for c in line):
            continue
        # Must be 2-4 words, all alphabetic
        tokens = line.split()
        if not (2 <= len(tokens) <= 4):
            continue
        if not all(t.replace("'", "").replace("-", "").replace("`", "").isalpha() for t in tokens):
            continue
        # Skip bullets
        if line.startswith('-') or line.startswith('•'):
            continue
        # Skip common non-name patterns
        lower = line.lower()
        if any(kw in lower for kw in ['participated', 'bachelor', 'master', 'degree', 'course', 'project', 'home', 'service', 'provider', 'team', 'coll']):
            continue
        logger.debug(f"Found potential name in raw text: {line}")
        return line.strip()
    return None


def _extract_email_from_sections(sections: Dict[str, List[str]]) -> Optional[str]:
    """Extract email from all sections."""
    for section_name, lines in sections.items():
        for line in lines:
            match = _EMAIL_RE.search(line)
            if match:
                logger.debug(f"Found email in {section_name}: {match.group()}")
                return match.group()
    return None


def _extract_phone_from_sections(sections: Dict[str, List[str]]) -> Optional[str]:
    """Extract phone from all sections."""
    for section_name, lines in sections.items():
        for line in lines:
            match = _PHONE_RE.search(line)
            if match:
                logger.debug(f"Found phone in {section_name}: {match.group()}")
                return match.group()
    return None


def _extract_location_from_sections(sections: Dict[str, List[str]]) -> Optional[str]:
    """Extract location from all sections."""
    for section_name, lines in sections.items():
        for line in lines:
            loc = _clean_location(line)
            if loc:
                logger.debug(f"Found location in {section_name}: {loc}")
                return loc
    return None


def _clean_location(location: str) -> Optional[str]:
    """
    Clean and validate location string.
    
    Returns None if location is invalid (email, URL, org name, etc.)
    """
    loc = str(location or "").strip()
    if not loc:
        return None
    
    # Reject if contains parentheses
    if "(" in loc or ")" in loc:
        return None
    
    low = loc.lower()
    
    # Reject if contains email or URL
    if _EMAIL_RE.search(loc) or _URL_RE.search(loc):
        return None
    
    # Reject if looks like organization name
    if _ORG_LOCATION_RE.search(low):
        return None
    
    # Reject if too long
    if len(loc.split()) > 6:
        return None
    
    # Reject if in stopwords
    if loc.lower() in LOCATION_STOPWORDS:
        return None
    
    return loc
