"""
Entity Extractor - Resume Information Extraction

Extracts structured entities from resume text:
- Name (candidate's full name)
- Email (validated email addresses)
- Phone (Indian and international formats)
- Location (city, state, country)

Features:
- Multi-layer extraction (direct + context + fallback)
- Context-aware filtering to reduce false positives
- Support for ALL CAPS and mixed case names
- Indian location normalization
- Comprehensive logging for debugging
- Safe error handling with fallback logic
"""

import re
from itertools import islice
from typing import Any, Dict, Optional
import logging

from .keyword_loader import load_wordlist
from .extraction_utils import (
    extract_emails, validate_email, validate_name, validate_location,
    clean_line, normalize_text,
)
from ml_engine.utils.exceptions import ExtractionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Words that indicate a line is NOT a name
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
    # Company/organization words
    "technologies", "solutions", "systems", "services",
    "ltd", "limited", "pvt", "private", "inc", "corp",
})

# Location stopwords (words that appear in addresses but aren't locations)
_LOCATION_STOPWORDS: frozenset = frozenset({
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln",
    "floor", "flat", "no", "near", "opp", "behind",
    "address", "contact", "phone", "mobile", "email",
})

# ---------------------------------------------------------------------------
# Wordlists (Lazy Loaded)
# ---------------------------------------------------------------------------

def _safe_load(filename: str) -> frozenset:
    """Safely load wordlist with error handling."""
    try:
        return load_wordlist(filename, required=False)
    except Exception as exc:
        logger.warning(f"Failed to load wordlist '{filename}': {exc}")
        return frozenset()


_COMMON_HEADERS: frozenset = _safe_load("common_headers.txt")
_LOCATION_KEYWORDS: frozenset = _safe_load("locations.txt")
_INDIAN_CITIES: frozenset = _safe_load("locations.txt")  # Use locations.txt as fallback

_HEADER_SET: frozenset = frozenset(h.lower() for h in _COMMON_HEADERS)

# ---------------------------------------------------------------------------
# Email Extraction
# ---------------------------------------------------------------------------

def extract_email(text: str) -> Optional[str]:
    """
    Extract the first valid email address from text.
    
    Features:
    - Validates email format
    - Removes trailing punctuation
    - Filters out false positives
    
    Args:
        text: Resume text
        
    Returns:
        First valid email or None
    """
    logger.debug("Starting email extraction...")
    
    emails = extract_emails(text)
    
    if not emails:
        logger.debug("No email patterns found")
        return None
    
    # Validate and return first valid email
    for email in emails:
        if validate_email(email):
            logger.info(f"Extracted email: {email}")
            return email
    
    logger.debug("No valid emails found after validation")
    return None


# ---------------------------------------------------------------------------
# Phone Extraction
# ---------------------------------------------------------------------------

def extract_phone(text: str) -> Optional[str]:
    """
    Extract the first valid phone number from text.
    
    Features:
    - Supports Indian (+91) and international formats
    - Handles spaces, dashes, dots in phone numbers
    - Validates digit count (10-13 digits)
    - Rejects year-like patterns
    
    Args:
        text: Resume text
        
    Returns:
        First valid phone or None
    """
    logger.debug("Starting phone extraction...")
    
    # Pattern that allows spaces/dashes within the number
    pattern = r'(?:\+?\d[\d\s\-.]{8,15}\d)'
    matches = re.findall(pattern, text)
    
    valid_phones = []
    for match in matches:
        # Remove all non-digit characters
        digits = re.sub(r'[^\d]', '', match)
        
        # Reject year-like 4-digit sequences
        if len(digits) == 4 and digits[:2] in {'19', '20'}:
            continue
        
        # Validate digit count (10-13 for most phones)
        if 10 <= len(digits) <= 13:
            # Re-add + prefix if original had it
            phone = ("+" + digits) if match.strip().startswith("+") else digits
            valid_phones.append(phone)
    
    if not valid_phones:
        logger.debug("No valid phone patterns found")
        return None
    
    phone = valid_phones[0]
    logger.info(f"Extracted phone: {phone}")
    return phone


# ---------------------------------------------------------------------------
# Name Extraction
# ---------------------------------------------------------------------------

def extract_name(text: str) -> Optional[str]:
    """
    Extract candidate name from resume text.
    
    Strategy:
    1. Check first 3 lines for ALL CAPS name (highest priority)
    2. Scan first 8 lines for name-like patterns
    3. Accept ALL CAPS, Title Case, or mixed case
    4. Filter out headers, roles, contact info
    5. Validate format (2-4 alpha tokens)
    
    Args:
        text: Resume text
        
    Returns:
        Candidate name or None
    """
    logger.debug("Starting name extraction...")
    
    lines = text.splitlines()
    
    # Strategy 1: Check first 3 lines for ALL CAPS name (common resume format)
    for i, line in enumerate(islice(lines, 3)):
        line = clean_line(line)
        
        if not line:
            continue
        
        # ALL CAPS with 2-4 words is very likely a name
        if line.isupper() and 2 <= len(line.split()) <= 4:
            # Validate: only alpha and allowed punctuation
            if re.fullmatch(r"[A-Z .'\-`]+", line):
                # Skip if contains role words
                lower = line.lower()
                tokens = lower.split()
                if not any(tok in _ROLE_WORDS for tok in tokens):
                    name = re.sub(r'\s+', ' ', line).strip()
                    logger.info(f"Extracted name (ALL CAPS line {i}): {name}")
                    return name
    
    # Strategy 2: Scan first 8 lines for name-like patterns
    for i, line in enumerate(islice(lines, 8)):
        line = clean_line(line)
        
        if not line:
            continue
        
        # Skip lines that are too long
        if len(line) > 50:
            continue
        
        lower = line.lower()
        
        # Skip known headers
        if lower in _HEADER_SET:
            logger.debug(f"Line {i}: Skipped (known header)")
            continue
        
        # Skip lines with contact info
        if "@" in line or any(c.isdigit() for c in line):
            logger.debug(f"Line {i}: Skipped (contains contact info)")
            continue
        
        # Skip lines with URLs
        if "http" in lower or ".com" in lower or ".in" in lower:
            logger.debug(f"Line {i}: Skipped (contains URL)")
            continue
        
        # Tokenize and check for role words
        tokens = lower.split()
        if any(tok in _ROLE_WORDS for tok in tokens):
            logger.debug(f"Line {i}: Skipped (contains role word: {tokens})")
            continue
        
        # Skip lines that look like project descriptions
        if any(kw in lower for kw in ['project', 'developed', 'built', 'created', 'using']):
            logger.debug(f"Line {i}: Skipped (looks like project)")
            continue
        
        # Must be 2-4 alpha tokens
        if not (2 <= len(tokens) <= 4):
            logger.debug(f"Line {i}: Skipped (wrong token count: {len(tokens)})")
            continue
        
        # Must match name pattern
        if not re.fullmatch(r"[A-Za-z .'\-`]+", line):
            logger.debug(f"Line {i}: Skipped (invalid characters)")
            continue
        
        # Validate name format
        if validate_name(line):
            # Preserve original casing, normalize spacing
            name = re.sub(r'\s+', ' ', line).strip()
            logger.info(f"Extracted name (line {i}): {name}")
            return name
    
    # Strategy 3: Fallback - search for name patterns in full text
    logger.debug("Fallback: Searching full text for name patterns...")
    name = _extract_name_fallback(text)
    
    if name:
        logger.info(f"Extracted name (fallback): {name}")
    
    return name


def _extract_name_fallback(text: str) -> Optional[str]:
    """
    Fallback name extraction from full text.
    
    Looks for patterns like:
    - "Name: John Doe"
    - "I am John Doe"
    - ALL CAPS lines with 2-4 words
    """
    # Pattern 1: Labeled name
    labeled_pattern = r'(?:name|candidate|applicant)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    match = re.search(labeled_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: "I am" or "I'm" pattern
    i_am_pattern = r"(?:I am|I'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
    match = re.search(i_am_pattern, text)
    if match:
        return match.group(1).strip()
    
    # Pattern 3: ALL CAPS lines (2-4 words)
    for line in text.splitlines()[:20]:
        line = clean_line(line)
        
        if line.isupper() and 2 <= len(line.split()) <= 4:
            # Skip if contains non-alpha
            if not re.fullmatch(r"[A-Z .'\-`]+", line):
                continue
            
            # Skip if too short (likely header)
            if len(line) < 5:
                continue
            
            return line.title()
    
    return None


# ---------------------------------------------------------------------------
# Location Extraction
# ---------------------------------------------------------------------------

def extract_location(text: str) -> Optional[str]:
    """
    Extract location from resume text.
    
    Features:
    - Supports Indian cities and international locations
    - Normalizes spacing and punctuation
    - Filters false positives
    
    Args:
        text: Resume text
        
    Returns:
        Location string or None
    """
    logger.debug("Starting location extraction...")
    
    lines = text.splitlines()
    
    # Strategy 1: Scan first 50 lines for location keywords
    for i, line in enumerate(islice(lines, 50)):
        line_lower = line.lower()
        
        for keyword in _LOCATION_KEYWORDS:
            if re.search(rf'\b{re.escape(keyword)}\b', line_lower):
                location = _clean_location(line)
                
                if location and validate_location(location):
                    logger.info(f"Extracted location: {location}")
                    return location
    
    # Strategy 2: Search for Indian cities
    logger.debug("Fallback: Searching for Indian cities...")
    for i, line in enumerate(islice(lines, 50)):
        line_lower = line.lower()
        
        for city in _INDIAN_CITIES:
            pattern = rf'\b{re.escape(city)}\b'
            if re.search(pattern, line_lower):
                # Return properly capitalized city name
                location = city.title()
                logger.info(f"Extracted location (city): {location}")
                return location
    
    logger.debug("No location found")
    return None


def _clean_location(line: str) -> Optional[str]:
    """
    Clean and normalize location string.
    
    Removes:
    - Email addresses
    - Phone numbers
    - URLs
    - Excessive whitespace
    """
    clean = line.strip()
    
    # Strip emails
    clean = re.sub(r'\S+@\S+', '', clean)
    
    # Strip phone numbers
    clean = re.sub(r'\+?\d[\d\s\-]{7,}', '', clean)
    
    # Strip URLs
    clean = re.sub(r'(?:https?://|www\.)\S+', '', clean)
    
    # Normalize whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    # Remove trailing punctuation
    clean = clean.rstrip(',.;:')
    
    # Must be reasonable length
    if not clean or len(clean) > 80 or len(clean.split()) > 7:
        return None
    
    return clean


# ---------------------------------------------------------------------------
# Unified Entry Point
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> Dict[str, Optional[str]]:
    """
    Extract all entities from resume text.

    Args:
        text: Raw resume text

    Returns:
        Dictionary with keys: name, email, phone, location

    Raises:
        ExtractionError: If extraction fails critically
    """
    logger.info("Starting entity extraction...")

    try:
        # Note: We don't normalize text here because name extraction
        # relies on case information (ALL CAPS detection)
        
        entities = {
            "name": extract_name(text),
            "email": extract_email(text),
            "phone": extract_phone(text),
            "location": extract_location(text),
        }

        # Log extraction summary
        logger.info("Entity extraction complete:")
        logger.info(f"  - Name: {entities['name']}")
        logger.info(f"  - Email: {entities['email']}")
        logger.info(f"  - Phone: {entities['phone']}")
        logger.info(f"  - Location: {entities['location']}")

        return entities
        
    except Exception as exc:
        logger.exception(f"Entity extraction failed: {exc}")
        raise ExtractionError(f"Failed to extract entities: {exc}") from exc


def extract_entities_detailed(text: str) -> Dict[str, Any]:
    """
    Extract entities with additional metadata.

    Returns:
        Dictionary with entity values and confidence scores
    """
    logger.info("Starting detailed entity extraction...")
    
    normalized = normalize_text(text)
    
    result = {
        "entities": extract_entities(normalized),
        "metadata": {
            "text_length": len(normalized),
            "line_count": len(normalized.splitlines()),
        }
    }
    
    # Calculate confidence scores
    entities = result["entities"]
    confidence = 0
    
    if entities["name"]:
        confidence += 25
    if entities["email"]:
        confidence += 25
    if entities["phone"]:
        confidence += 25
    if entities["location"]:
        confidence += 25
    
    result["metadata"]["extraction_confidence"] = confidence
    
    return result
