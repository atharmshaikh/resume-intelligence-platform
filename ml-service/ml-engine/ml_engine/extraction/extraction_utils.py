"""
Extraction Utilities - Shared Helpers

Common utilities for all extraction modules:
- Text normalization
- Regex helpers
- Token filtering
- Noise detection
- Validation helpers
"""

import logging
import re
from typing import List, Set, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for extraction.
    
    - Convert to lowercase
    - Normalize whitespace
    - Remove excessive newlines
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_line(line: str) -> str:
    """
    Clean a single line for processing.
    
    - Strip whitespace
    - Normalize internal spaces
    """
    line = line.strip()
    line = re.sub(r' {2,}', ' ', line)
    return line


def normalize_spacing(text: str) -> str:
    """
    Normalize spacing throughout document.
    
    Preserves paragraph breaks but cleans inconsistent spacing.
    """
    lines = text.splitlines()
    normalized = []
    
    prev_was_empty = False
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            if not prev_was_empty:
                normalized.append('')
                prev_was_empty = True
        else:
            normalized.append(stripped)
            prev_was_empty = False
    
    return '\n'.join(normalized)


# ---------------------------------------------------------------------------
# Regex Helpers
# ---------------------------------------------------------------------------

def extract_emails(text: str) -> List[str]:
    """Extract all email addresses from text."""
    pattern = r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.lower().strip('.,;:\'"') for m in matches]


def extract_phones(text: str) -> List[str]:
    """Extract all phone numbers from text."""
    pattern = r'(?:\+?(?:91|1|44|61|971|65)\s*[-.\s]?)?(?:\(?[0-9]{3,5}\)?[-.\s]?)?[0-9]{7,10}'
    matches = re.findall(pattern, text)
    
    valid_phones = []
    for match in matches:
        digits = re.sub(r'[^\d]', '', match)
        
        # Reject year-like 4-digit sequences
        if len(digits) == 4 and digits[:2] in {'19', '20'}:
            continue
        
        if 10 <= len(digits) <= 13:
            # Re-add + prefix if original had it
            phone = ("+" + digits) if match.strip().startswith("+") else digits
            valid_phones.append(phone)
    
    return valid_phones


def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    pattern = r'(?:https?://|www\.)\S+'
    return re.findall(pattern, text, re.IGNORECASE)


def extract_years(text: str) -> List[str]:
    """Extract year patterns from text."""
    pattern = r'\b(?:19|20)\d{2}\b'
    return re.findall(pattern, text)


# ---------------------------------------------------------------------------
# Token Filtering
# ---------------------------------------------------------------------------

def is_valid_token(token: str, min_length: int = 2) -> bool:
    """
    Check if token is valid for extraction.
    
    - Minimum length
    - Alphabetic or alphanumeric
    - Not a number
    """
    if len(token) < min_length:
        return False
    
    if token.isdigit():
        return False
    
    # Must contain at least one letter
    if not any(c.isalpha() for c in token):
        return False
    
    return True


def filter_tokens(tokens: List[str], 
                  stopwords: Optional[Set[str]] = None,
                  min_length: int = 2) -> List[str]:
    """
    Filter tokens by validity and stopwords.
    
    Args:
        tokens: List of tokens to filter
        stopwords: Set of words to exclude
        min_length: Minimum token length
        
    Returns:
        Filtered list of valid tokens
    """
    if stopwords is None:
        stopwords = set()
    
    filtered = []
    for token in tokens:
        token_lower = token.lower().strip()
        
        if token_lower in stopwords:
            continue
        
        if not is_valid_token(token, min_length):
            continue
        
        filtered.append(token_lower)
    
    return filtered


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Split text into tokens.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        
    Returns:
        List of tokens
    """
    if lowercase:
        text = text.lower()
    
    # Split on non-alphanumeric
    tokens = re.findall(r'[a-zA-Z0-9]+', text)
    return tokens


# ---------------------------------------------------------------------------
# Noise Detection
# ---------------------------------------------------------------------------

def is_noise_line(line: str) -> bool:
    """
    Check if line is likely noise.
    
    Noise indicators:
    - Too short (< 3 chars)
    - Too long (> 200 chars)
    - All caps single word
    - Contains only symbols
    """
    stripped = line.strip()
    
    if len(stripped) < 3:
        return True
    
    if len(stripped) > 200:
        return True
    
    # All caps single word (likely icon or artifact)
    if stripped.isupper() and len(stripped.split()) == 1 and len(stripped) <= 5:
        return True
    
    # Only symbols
    if re.fullmatch(r'[^a-zA-Z0-9]+', stripped):
        return True
    
    return False


def is_contact_line(line: str) -> bool:
    """
    Check if line contains contact information.
    
    Indicators:
    - Email pattern
    - Phone pattern
    - URL pattern
    """
    if '@' in line:
        return True
    
    if re.search(r'(?:https?://|www\.)', line, re.IGNORECASE):
        return True
    
    # Phone-like pattern
    if re.search(r'\d{7,}', line):
        return True
    
    return False


def is_header_line(line: str, known_headers: Optional[Set[str]] = None) -> bool:
    """
    Check if line is a section header.
    
    Args:
        line: Line to check
        known_headers: Set of known header strings
        
    Returns:
        True if line appears to be a header
    """
    stripped = line.strip()
    
    if not stripped or len(stripped) > 60:
        return False
    
    lower = stripped.lower()
    
    # Check against known headers
    if known_headers and lower in known_headers:
        return True
    
    # All caps, short
    if stripped.isupper() and len(stripped.split()) <= 4:
        return True
    
    # Title case, short, no ending punctuation
    if stripped.istitle() and len(stripped.split()) <= 3:
        if not stripped.endswith(('.', ',', ';', ':')):
            return True
    
    return False


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number format (Indian context)."""
    digits = re.sub(r'[^\d]', '', phone)
    
    # Indian numbers: 10-13 digits
    if phone.startswith('+91'):
        return len(digits) == 13
    
    # International: 10-15 digits
    return 10 <= len(digits) <= 15


def validate_name(name: str, min_words: int = 2, max_words: int = 4) -> bool:
    """
    Validate name format.
    
    Checks:
    - Word count within range
    - Contains only valid characters
    - No digits
    """
    words = name.split()
    
    if not (min_words <= len(words) <= max_words):
        return False
    
    # Must be alphabetic with allowed punctuation
    if not re.fullmatch(r"[A-Za-z .'\-`]+", name):
        return False
    
    # No digits
    if any(c.isdigit() for c in name):
        return False
    
    return True


def validate_location(location: str, max_words: int = 7) -> bool:
    """
    Validate location format.
    
    Checks:
    - Reasonable length
    - Word count limit
    - Contains alphabetic characters
    """
    if not location or len(location) > 80:
        return False
    
    if len(location.split()) > max_words:
        return False
    
    # Must contain at least one alphabetic character
    if not any(c.isalpha() for c in location):
        return False
    
    return True


# ---------------------------------------------------------------------------
# Context Helpers
# ---------------------------------------------------------------------------

def get_context_window(text: str, position: int, window_size: int = 100) -> Tuple[str, str]:
    """
    Get context window around a position.
    
    Args:
        text: Full text
        position: Position of interest
        window_size: Size of context window
        
    Returns:
        Tuple of (before_context, after_context)
    """
    start = max(0, position - window_size)
    end = min(len(text), position + window_size)
    
    before = text[start:position]
    after = text[position:end]
    
    return before, after


def find_in_context(text: str, pattern: str, context_size: int = 50) -> List[Tuple[str, str, str]]:
    """
    Find all occurrences of pattern with context.
    
    Args:
        text: Text to search
        pattern: Pattern to find
        context_size: Size of context to extract
        
    Returns:
        List of (before, match, after) tuples
    """
    results = []
    
    for match in re.finditer(pattern, text, re.IGNORECASE):
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)
        
        before = text[start:match.start()]
        matched = match.group()
        after = text[match.end():end]
        
        results.append((before, matched, after))
    
    return results
