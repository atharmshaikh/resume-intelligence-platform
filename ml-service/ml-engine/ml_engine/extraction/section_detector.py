"""
Resume section detector.

Strategy
--------
A line qualifies as a section header if and only if:

  1. It is short (≤ 60 chars, ≤ 5 words after normalisation).
  2. After lower-casing, stripping punctuation and collapsing whitespace,
     it exactly matches a known keyword or alias.
  3. It does NOT end in sentence-continuation punctuation (, ; ( ) ).

Two-line headers like "PROFESSIONAL\\nEXPERIENCE" are safely merged before
scanning — but only when the combined form is itself a known keyword.

Handles any messy resume: all-caps, Title Case, mixed case, trailing colons,
extra spaces.  No regex backtracking on the hot path.
"""

import re
from typing import Dict, List, Optional
import logging

from ml_engine.config import SECTION_KEYWORDS  # type: ignore[import]
from ml_engine.utils.exceptions import ExtractionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HEADER_CHARS = 60
MAX_HEADER_WORDS = 5
MAX_SECTION_LINES = 500

# Characters to strip when normalising a candidate header line.
# Keep only alphanumeric and space (handles colons, dashes, bullets, /, &, +).
_STRIP_RE = re.compile(r"[^a-z0-9\s]")

# ---------------------------------------------------------------------------
# Build canonical lookup: normalised keyword → section key (done once)
# ---------------------------------------------------------------------------

KEYWORD_TO_SECTION: Dict[str, str] = {}

for _section_key, _kws in SECTION_KEYWORDS.items():
    for _kw in _kws:
        _normed = re.sub(r"\s+", " ", _kw.strip().lower())
        KEYWORD_TO_SECTION[_normed] = _section_key

# Extended aliases (variants not in ats_config but common in wild resumes)
_EXTRA_ALIASES: Dict[str, str] = {
    # skills
    "my skills":                   "skills",
    "skill set":                   "skills",
    "skillset":                    "skills",
    # experience
    "work":                        "experience",
    "jobs":                        "experience",
    "career":                      "experience",
    # education
    "academic":                    "education",
    "schooling":                   "education",
    # projects
    "my projects":                 "projects",
    # achievements
    "extra curricular":            "achievements",
    "co curricular":               "achievements",
    "courses and certifications":  "achievements",
    "certifications and awards":   "achievements",
    # objective
    "career goal":                 "career_objective",
    "career goals":                "career_objective",
    "personal profile":            "career_objective",
    # languages
    "language":                    "languages",
}

KEYWORD_TO_SECTION.update(_EXTRA_ALIASES)

# All tokens we recognise (used in the merge pass)
ALL_SECTION_TOKENS: set = set(KEYWORD_TO_SECTION.keys())


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise(raw: str) -> str:
    """Lower-case, remove non-alphanumeric, collapse whitespace, strip colon."""
    n = raw.strip().lower()
    n = _STRIP_RE.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ---------------------------------------------------------------------------
# Header classifier
# ---------------------------------------------------------------------------

def _match_header(line: str) -> Optional[str]:
    """
    Return the canonical section key if *line* is a section header,
    otherwise None.

    Robust to:
    - ALL CAPS ("EDUCATION")
    - Title Case ("Education")
    - trailing colon ("Education:")
    - leading bullets ("• Skills")
    - extra spaces
    """
    stripped = line.strip()

    if not stripped:
        return None

    # Hard length guard – avoids processing long description lines
    if len(stripped) > MAX_HEADER_CHARS:
        return None

    # Lines ending in continuation punctuation are not headers
    if stripped[-1] in {",", ";"}:
        return None

    normalised = _normalise(stripped)

    if not normalised:
        return None

    if len(normalised.split()) > MAX_HEADER_WORDS:
        return None

    # Trailing colon normalisation ("education" matches "education:")
    lookup = normalised.rstrip(":")

    return KEYWORD_TO_SECTION.get(lookup)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sections(text: str) -> Dict[str, List[str]]:
    """
    Detect resume sections from cleaned resume text.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of canonical section name → list of non-empty content lines.
    """
    try:
        return _detect_sections_impl(text)
    except Exception as exc:
        msg = "Critical failure during section detection."
        logger.exception(msg)
        raise ExtractionError(msg) from exc

def _detect_sections_impl(text: str) -> Dict[str, List[str]]:
    if not text or not isinstance(text, str):
        return {}

    lines: List[str] = text.splitlines()

    # ------------------------------------------------------------------
    # Pass 1 – merge two-line split headers.
    #
    # PDFs often produce:
    #   "PROFESSIONAL "
    #   "EXPERIENCE"
    # These must be merged before keyword lookup.
    #
    # Rule: merge only when BOTH lines are short single-word alpha tokens
    # AND their combined form is in ALL_SECTION_TOKENS.
    # This prevents merging real content lines.
    # ------------------------------------------------------------------
    merged: List[str] = []
    i = 0
    while i < len(lines):
        current = lines[i].strip()
        lower_c = _normalise(current)

        # Already a known keyword on its own → no merge needed
        if lower_c in ALL_SECTION_TOKENS:
            merged.append(current)
            i += 1
            continue

        # Try merging with next line
        if (
            i + 1 < len(lines)
            and 1 <= len(current.split()) <= 2
            and len(current) <= 30
            and current.replace(" ", "").isalpha()
        ):
            nxt = lines[i + 1].strip()
            if (
                nxt
                and 1 <= len(nxt.split()) <= 2
                and len(nxt) <= 30
                and nxt.replace(" ", "").isalpha()
            ):
                candidate = _normalise(current + " " + nxt)
                if candidate in ALL_SECTION_TOKENS:
                    merged.append(current + " " + nxt)
                    i += 2
                    continue

        merged.append(current)
        i += 1

    # ------------------------------------------------------------------
    # Pass 2 – walk lines, detect headers, accumulate section content
    # ------------------------------------------------------------------
    sections: Dict[str, List[str]] = {}
    current_section: Optional[str] = None
    buffer: List[str] = []

    for line in merged:
        key = _match_header(line)

        # AI Heuristic: Unknown Section Detection
        if key is None:
            _clean = line.strip()
            if 3 < len(_clean) <= 40 and len(_clean.split()) <= 4:
                if _clean[-1] not in {'.', ',', ';', '-', '(', ')'}:
                    if _clean.isupper() or (_clean.istitle() and len(_clean.split()) <= 2):
                        # Avoid classifying generic single words as sections easily
                        if len(_clean.split()) > 1 or len(_clean) >= 5:
                            key = f"unknown_{_clean.lower().replace(' ', '_')}"

        if key is not None:
            # Avoid resetting the same section twice in a row
            if key == current_section:
                continue

            # Flush previous section
            if current_section is not None and buffer:
                content = [ln.strip() for ln in buffer if ln.strip()]
                if content:
                    sections.setdefault(current_section, []).extend(content)

            current_section = key  # type: ignore[assignment]
            buffer = []

        else:
            if current_section is not None:
                if len(buffer) < MAX_SECTION_LINES and len(line) < 2000:
                    buffer.append(line)

    # Flush last section
    if current_section is not None and buffer:
        content = [ln.strip() for ln in buffer if ln.strip()]
        if content:
            sections.setdefault(current_section, []).extend(content)

    return {k: v for k, v in sections.items() if v}