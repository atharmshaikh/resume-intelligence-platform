"""
Text cleaning utilities for resume parsing.

Centralizes all text normalization logic to avoid duplication.
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Unicode cleanup patterns
_UNICODE_FIXES = {
    '\uf0e0': '',  # Email icon
    '\uf095': '',  # Phone icon
    '\uf080': '',  # Section icon
    '\uf19d': '',  # Other icons
    '\u2022': '•',  # Normalize bullets
    '\u2013': '-',  # En dash
    '\u2014': '-',  # Em dash
    '\u00a0': ' ',  # Non-breaking space
}

# Noise patterns
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
_EMAIL_PATTERN = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
PHONE_PATTERN = re.compile(r'(?:\+?\d[\d\s\-]{8,13}\d)')

# Bullet patterns
_BULLET_CHARS = {'•', '●', '▪', '◦', '►', '▸', '■', '□', '◆', '◇', '-', '*', ''}


def clean_text(text: str) -> str:
    """
    Clean and normalize resume text.
    
    Args:
        text: Raw text from PDF/DOCX parser
        
    Returns:
        Cleaned text with normalized unicode, spacing, and formatting
    """
    if not text:
        return ""
    
    logger.debug(f"Cleaning text: {len(text)} characters")
    
    # Step 1: Unicode normalization
    text = _fix_unicode(text)
    
    # Step 2: Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Step 3: Normalize bullets
    text = _normalize_bullets(text)
    
    # Step 4: Clean excessive whitespace
    text = _clean_whitespace(text)
    
    # Step 5: Remove noise (optional - keep for most resumes)
    # text = _remove_noise(text)
    
    logger.debug(f"Cleaned text: {len(text)} characters")
    return text


def _fix_unicode(text: str) -> str:
    """Fix common unicode issues in PDF text."""
    for unicode_char, replacement in _UNICODE_FIXES.items():
        text = text.replace(unicode_char, replacement)
    return text


def _normalize_bullets(text: str) -> str:
    """Normalize various bullet characters to standard format."""
    # Replace various bullet characters with standard bullet
    for bullet in _BULLET_CHARS:
        if bullet != '•':  # Keep standard bullet
            text = text.replace(bullet, '•')
    return text


def _clean_whitespace(text: str) -> str:
    """Clean excessive whitespace while preserving structure."""
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Strip leading/trailing whitespace
        line = line.strip()
        
        # Collapse multiple spaces into single space
        line = re.sub(r' {2,}', ' ', line)
        
        # Keep non-empty lines
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def _remove_noise(text: str) -> str:
    """
    Remove noise from text.
    
    Note: Use carefully - may remove valid content.
    Currently disabled by default.
    """
    # Remove URLs (keep email/phone)
    text = _URL_PATTERN.sub('', text)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


def split_into_lines(text: str) -> List[str]:
    """
    Split text into clean lines.
    
    Args:
        text: Raw or cleaned text
        
    Returns:
        List of non-empty lines
    """
    lines = text.splitlines()
    return [line.strip() for line in lines if line.strip()]


def is_bullet_line(line: str) -> bool:
    """Check if line starts with a bullet character."""
    stripped = line.strip()
    return stripped and stripped[0] in _BULLET_CHARS


def is_heading_line(line: str) -> bool:
    """
    Check if line looks like a section heading.
    
    Heuristics:
    - Short line (< 60 chars)
    - All caps or title case
    - No ending punctuation
    """
    stripped = line.strip()
    
    if not stripped or len(stripped) > 60:
        return False
    
    if stripped.endswith(('.', ',', ';', ':')):
        return False
    
    # All caps (common for section headers)
    if stripped.isupper() and len(stripped.split()) <= 4:
        return True
    
    # Title case, short
    if stripped.istitle() and len(stripped.split()) <= 3:
        return True
    
    return False


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
            # Keep single blank lines between sections
            if not prev_was_empty:
                normalized.append('')
                prev_was_empty = True
        else:
            normalized.append(stripped)
            prev_was_empty = False
    
    return '\n'.join(normalized)
