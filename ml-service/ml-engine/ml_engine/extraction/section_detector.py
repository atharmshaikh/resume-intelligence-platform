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
# 2-Column Layout Preprocessor
# ---------------------------------------------------------------------------

def _detect_2column_layout(lines: List[str]) -> bool:
    """
    Detect if text appears to be in 2-column format.
    
    Heuristics:
    - Lines with large whitespace gaps in the middle (10+ spaces)
    - Both left and right sides have independent meaningful content
    - Pattern consistent across many lines
    
    Conservative detection to avoid false positives.
    Key check: midpoint should NOT split words (real 2-column has clean breaks).
    """
    if len(lines) < 10:
        return False
    
    gap_count = 0
    mixed_content = 0
    long_lines = 0
    word_split_count = 0  # Track if midpoint splits words
    
    for line in lines[:100]:  # Sample first 100 lines
        if len(line) < 80:  # Only consider very long lines
            continue
        
        long_lines += 1
        
        midpoint = len(line) // 2
        
        # KEY CHECK: If midpoint splits a word, this is NOT 2-column layout
        # Real 2-column resumes have whitespace at column boundaries
        mid_chars = line[midpoint-2:midpoint+2]
        if mid_chars.isalpha() or (line[midpoint-1].isalpha() and line[midpoint].isalpha()):
            word_split_count += 1
            continue  # This line doesn't have a clean column break
        
        # Check for very large whitespace gap (10+ spaces) in middle third
        mid_start = len(line) // 3
        mid_end = 2 * len(line) // 3
        middle_section = line[mid_start:mid_end]
        
        if re.search(r'\s{10,}', middle_section):
            gap_count += 1
        
        # Check if line has truly independent content on both sides
        left = line[:midpoint].strip()
        right = line[midpoint:].strip()
        
        # Both sides must have substantial independent content
        if left and right and len(left) > 20 and len(right) > 20:
            left_words = len(left.split())
            right_words = len(right.split())
            # Both sides should look like complete phrases/sentences
            if left_words >= 4 and right_words >= 4:
                # Additional check: right side should start with capital or bullet
                if right[0].isupper() or right[0] in '-•●▪':
                    mixed_content += 1
    
    # If midpoint splits words in >30% of long lines, NOT a 2-column layout
    if long_lines >= 5 and word_split_count / long_lines > 0.3:
        return False
    
    # Require very strong evidence: >60% of long lines show clear 2-column pattern
    if long_lines >= 8:
        ratio = (gap_count + mixed_content) / long_lines
        return ratio > 0.6
    
    return False


def _split_columns(lines: List[str]) -> str:
    """
    Split 2-column text into linear single-column format.

    Strategy:
    1. Find real whitespace gap (≥6 spaces) in each line
    2. Split at gap position (not midpoint)
    3. Accumulate left and right column content separately
    4. Return left_block + "\\n\\n" + right_block
    5. Fallback: if no gaps found, return original text
    """
    left_parts: List[str] = []
    right_parts: List[str] = []
    gap_found_count = 0
    total_nonempty = 0

    for line in lines:
        if not line.strip():
            continue

        total_nonempty += 1

        # Find real column break: large whitespace gap (≥6 spaces)
        gap_match = re.search(r'\s{6,}', line)

        if gap_match:
            gap_found_count += 1
            # Split at the gap position
            left_part = line[:gap_match.start()].strip()
            right_part = line[gap_match.end():].strip()

            if left_part:
                left_parts.append(left_part)
            if right_part:
                right_parts.append(right_part)
        else:
            # No gap found - keep full line in left column
            # DO NOT split blindly at midpoint
            left_parts.append(line.strip())

    # Safety check: if we found gaps in <20% of lines, fallback to original
    if total_nonempty > 0 and gap_found_count / total_nonempty < 0.2:
        return "\n".join(lines)

    # Reconstruct as single-column text
    left_block = "\n".join(left_parts)
    right_block = "\n".join(right_parts)

    # Combine: left column first, then right column
    if left_block and right_block:
        result = left_block + "\n\n" + right_block
        # Safety check: ensure we didn't lose content
        if len(result) < len("\n".join(lines)) * 0.8:
            return "\n".join(lines)  # Fallback to original
        return result
    elif left_block:
        return left_block
    elif right_block:
        return right_block
    return "\n".join(lines)


def preprocess_2column(text: str) -> str:
    """
    Preprocess resume text to handle 2-column layouts.
    
    If 2-column pattern detected:
    - Split into left and right columns
    - Reconstruct as linear text
    
    Otherwise:
    - Return original text unchanged
    """
    if not text or not isinstance(text, str):
        return text
    
    lines = text.splitlines()
    
    if not _detect_2column_layout(lines):
        return text
    
    logger.info("Detected 2-column layout, preprocessing...")
    return _split_columns(lines)

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
    "other projects":              "projects",
    # achievements
    "extra curricular":            "achievements",
    "co curricular":               "achievements",
    "courses and certifications":  "achievements",
    "certifications and awards":   "achievements",
    # projects
    "projects":                    "projects",
    "PROJECTS":                    "projects",
    "academic projects":           "projects",
    "personal projects":            "projects",
    # skills
    "skills":                      "skills",
    "SKILLS":                      "skills",
    # experience
    "experience":                  "experience",
    "EXPERIENCE":                  "experience",
    # education
    "education":                    "education",
    "EDUCATION":                    "education",
    # objective
    "career goal":                 "career_objective",
    "career goals":                "career_objective",
    "personal profile":            "career_objective",
    # languages
    "language":                    "languages",
    "LANGUAGES":                   "languages",
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

    # HIGH-CONFIDENCE RECOVERY: If it's a single word, all-caps, and is a known keyword
    # this bypasses some of the more restrictive whitespace/punctuation checks.
    if stripped.isupper() and len(stripped.split()) == 1:
        norm_single = _normalise(stripped)
        if norm_single in ALL_SECTION_TOKENS:
            return KEYWORD_TO_SECTION[norm_single]

    if len(normalised.split()) > MAX_HEADER_WORDS:
        return None

    # Trailing colon normalisation ("education" matches "education:")
    lookup = normalised.rstrip(":")

    return KEYWORD_TO_SECTION.get(lookup)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sections(text: str, debug: bool = False) -> Dict[str, List[str]]:
    """
    Detect resume sections from cleaned resume text.
    
    Args:
        text: Cleaned resume text
        debug: Enable debug logging for section detection decisions

    Returns
    -------
    Dict[str, List[str]]
        Mapping of canonical section name → list of non-empty content lines.
    """
    try:
        logger.info("Starting section detection..." + (" (debug mode)" if debug else ""))
        
        # Preprocess 2-column layouts before section detection
        text = preprocess_2column(text)
        
        if debug:
            logger.debug(f"Text length after preprocessing: {len(text)} chars")
        
        sections = _detect_sections_impl(text, debug=debug)
        
        logger.info(f"Sections detected: {list(sections.keys())}")
        
        return sections
        
    except Exception as exc:
        msg = "Critical failure during section detection."
        logger.exception(msg)
        raise ExtractionError(msg) from exc


def detect_sections_with_confidence(text: str, 
                                     debug: bool = False) -> Dict[str, Dict]:
    """
    Detect sections with confidence scores.
    
    Args:
        text: Cleaned resume text
        debug: Enable debug logging
        
    Returns:
        Dictionary of section_name -> {content: [...], confidence: float}
    """
    sections = detect_sections(text, debug=debug)
    
    result = {}
    
    for section_name, content in sections.items():
        # Calculate confidence based on section characteristics
        confidence = _calculate_section_confidence(section_name, content)
        
        result[section_name] = {
            "content": content,
            "confidence": confidence,
        }
        
        if debug:
            logger.debug(f"  {section_name}: confidence={confidence:.2f}, lines={len(content)}")
    
    return result


def _calculate_section_confidence(section_name: str, content: List[str]) -> float:
    """
    Calculate confidence score for a detected section.
    
    Factors:
    - Content length (more content = higher confidence)
    - Section name clarity (known section = higher confidence)
    - Content consistency (uniform formatting = higher confidence)
    
    Returns:
        Confidence score between 0.5 and 1.0
    """
    confidence = 0.7  # Base confidence
    
    # Factor 1: Content length
    if len(content) >= 10:
        confidence += 0.15
    elif len(content) >= 5:
        confidence += 0.1
    elif len(content) >= 2:
        confidence += 0.05
    
    # Factor 2: Section name clarity
    known_sections = {'skills', 'education', 'experience', 'projects', 
                      'achievements', 'languages', 'interests', 'summary'}
    if section_name.lower() in known_sections:
        confidence += 0.1
    
    # Factor 3: Content consistency (check if lines have similar length)
    if len(content) >= 3:
        lengths = [len(line) for line in content[:10]]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((length_val - avg_length) ** 2 for length_val in lengths) / len(lengths)
        
        # Low variance = consistent formatting = higher confidence
        if variance < 100:
            confidence += 0.05
    
    return min(1.0, confidence)

def _detect_sections_impl(text: str, debug: bool = False) -> Dict[str, List[str]]:
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
                    if debug:
                        logger.debug(f"Merged header: '{current}' + '{nxt}' → '{candidate}'")
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
                            candidate_key = _clean.lower().replace(' ', '_')
                            # If this "unknown" section name actually maps to a known canonical key, use it!
                            if candidate_key in ALL_SECTION_TOKENS:
                                key = KEYWORD_TO_SECTION[candidate_key]
                                if debug:
                                    logger.debug(f"Detected unknown section mapping: '{_clean}' → '{key}'")

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
