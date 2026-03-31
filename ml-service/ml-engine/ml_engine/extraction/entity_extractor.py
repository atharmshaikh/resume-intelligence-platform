"""
Entity extraction from resume text.

Extracts:
  - name     (first non-contact, non-header line from the top)
  - email
  - phone    (Indian + international formats)
  - location (city, state, country patterns)

All extractors are designed to be robust against noisy PDF text.
"""

import re
from itertools import islice
from typing import Dict, List, Optional
from .keyword_loader import load_wordlist  # type: ignore[import]

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?:\+?(?:91|1|44|61|971|65)\s*[-.\s]?)?"   # optional country code
    r"(?:\(?[0-9]{3,5}\)?[-.\s]?)?"               # optional area code
    r"[0-9]{7,10}"                                 # main digits
)

# Words that appear in the first few lines but are NOT names
_ROLE_WORDS: frozenset = frozenset({
    # Job titles / roles
    "developer", "engineer", "designer", "manager", "analyst",
    "architect", "consultant", "specialist", "lead", "head",
    "director", "officer", "executive", "coordinator",
    # Degree types
    "bachelor", "b.tech", "btech", "b.e", "be",
    "bca", "b.c.a", "bsc", "b.sc",
    "master", "m.tech", "mtech", "mca", "m.c.a", "msc", "m.sc",
    "mba", "phd", "diploma", "pgdca",
    # Institution words
    "institute", "university", "college", "polytechnic",
    "school", "academy",
    # Common resume noise
    "resume", "curriculum", "vitae", "cv",
    "profile", "summary", "objective",
    # Stack roles
    "fullstack", "full-stack", "frontend", "front-end",
    "backend", "back-end", "devops", "intern",
    "student", "fresher", "graduate",
    # Indian IT company names sometimes appear
    "technologies", "solutions", "systems", "services",
})

# ---------------------------------------------------------------------------
# Wordlists
# ---------------------------------------------------------------------------

def _safe_load(filename: str) -> List[str]:
    try:
        return load_wordlist(filename)
    except Exception:
        return []

COMMON_HEADERS   : List[str] = _safe_load("common_headers.txt")
LOCATION_KEYWORDS: List[str] = _safe_load("locations.txt")

_HEADER_SET: frozenset = frozenset(h.lower() for h in COMMON_HEADERS)

# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def extract_email(text: str) -> Optional[str]:
    """Return the first valid email address found in text."""
    match = EMAIL_PATTERN.search(text)
    if not match:
        return None
    email = match.group().lower().strip(".,;:'\"")
    # Remove trailing domain fragments that sneak through
    email = re.sub(r"([a-z]{2,})(?:[.,;:]+)$", r"\1", email)
    return email


def extract_phone(text: str) -> Optional[str]:
    """Extract the first phone-like string with 10–13 digits."""
    for match in PHONE_PATTERN.finditer(text):
        raw   = match.group()
        digits = re.sub(r"[^\d]", "", raw)

        # Reject year-like 4-digit sequences
        if len(digits) == 4 and digits[:2] in {"19", "20"}:  # type: ignore[index]
            continue

        if 10 <= len(digits) <= 13:
            # Re-add + prefix if original had it
            return ("+" + digits) if raw.strip().startswith("+") else digits

    return None


def extract_name(text: str) -> Optional[str]:
    """
    Heuristically identify the candidate's name from the first 8 lines.

    Rules:
    - Must be 2–4 alpha tokens (first + last / first + middle + last)
    - Must not contain digits
    - Must match [A-Za-z .'`-]+ only
    - Must not be a known header, role word, or degree phrase
    - Prefers ALL-CAPS or Title Case lines (typical name formatting)
    """
    all_lines: List[str] = text.splitlines()

    for line in islice(all_lines, 8):
        line = line.strip()

        if not line:
            continue

        # Must be reasonably short
        if len(line) > 50:
            continue

        lower = line.lower()

        # Reject known headers
        if lower in _HEADER_SET:
            continue

        # Reject lines containing contact info
        if "@" in line or any(c.isdigit() for c in line):
            continue

        # Reject lines with URLs
        if "http" in lower or ".com" in lower or ".in" in lower:
            continue

        # Reject lines with known role / degree words
        tokens = lower.split()
        if any(tok in _ROLE_WORDS for tok in tokens):
            continue

        # Must be 2–4 word alpha tokens only
        if not (2 <= len(tokens) <= 4):
            continue

        if not re.fullmatch(r"[A-Za-z .'\-`]+", line):
            continue

        return line.title()

    return None


def extract_location(text: str) -> Optional[str]:
    """
    Find a location string (City, State, Country pattern) within
    the first 50 lines of the resume.
    """
    all_lines: List[str] = text.split("\n")

    for line in islice(all_lines, 50):
        line_lower = line.lower()

        for keyword in LOCATION_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", line_lower):
                clean = line

                # Strip emails and phone digits
                clean = re.sub(r"\S+@\S+", "", clean)
                clean = re.sub(r"\+?\d[\d\s\-]{7,}", "", clean)
                clean = clean.strip()

                # Must be a short, plausible location string
                if clean and len(clean) <= 80 and len(clean.split()) <= 7:
                    return clean

    return None


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> Dict[str, Optional[str]]:
    """Extract name, email, phone, and location from raw resume text."""
    return {
        "name":     extract_name(text),
        "email":    extract_email(text),
        "phone":    extract_phone(text),
        "location": extract_location(text),
    }