"""
Keyword Loader - Wordlist Management

Loads and caches keyword lists used by extraction modules.

Features:
- Lazy loading (load on first access)
- File existence validation
- Safe fallback (returns empty set if file missing)
- Optional preloading of critical wordlists
- Load-time logging for debugging
"""

import logging
from pathlib import Path
from typing import Optional, Set, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
WORDLIST_DIR = BASE_DIR / "wordlists"

# Critical wordlists to preload at startup
CRITICAL_WORDLISTS = [
    "common_headers.txt",
    "locations.txt",
    "section_keywords.txt",
]

# ---------------------------------------------------------------------------
# Cache and Statistics
# ---------------------------------------------------------------------------

@dataclass
class WordlistStats:
    """Statistics for wordlist loading."""
    total_loaded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failed_loads: int = 0
    load_times: Dict[str, float] = field(default_factory=dict)


_stats = WordlistStats()
_cache: Dict[str, frozenset] = {}
_preloaded: Set[str] = set()


def get_stats() -> WordlistStats:
    """Get wordlist loading statistics."""
    return _stats


def clear_cache() -> None:
    """Clear the wordlist cache."""
    global _cache
    _cache.clear()
    logger.info("Wordlist cache cleared")


# ---------------------------------------------------------------------------
# Core Loading Functions
# ---------------------------------------------------------------------------

def load_wordlist(filename: str, 
                  required: bool = False,
                  normalize: bool = True) -> frozenset:
    """
    Load newline-separated wordlist from wordlists directory.
    
    Features:
    - Lazy loading with caching
    - Comment support (lines starting with #)
    - Blank line handling
    - Safe fallback if file missing
    
    Args:
        filename: Name of wordlist file
        required: If True, raise error on missing file (default: False)
        normalize: Convert to lowercase (default: True)
        
    Returns:
        Frozen set of words (empty set if file missing and not required)
        
    Raises:
        FileNotFoundError: If file missing and required=True
    """
    # Check cache first
    cache_key = f"{filename}:{normalize}"
    
    if cache_key in _cache:
        _stats.cache_hits += 1
        logger.debug(f"Cache hit for '{filename}'")
        return _cache[cache_key]
    
    _stats.cache_misses += 1
    _stats.total_loaded += 1
    
    # Build path
    path = WORDLIST_DIR / filename
    
    # Check existence
    if not path.exists():
        _stats.failed_loads += 1
        
        if required:
            msg = f"Required wordlist not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            logger.warning(f"Wordlist not found (using empty set): {path}")
            return frozenset()
    
    # Load file
    logger.debug(f"Loading wordlist: {path}")
    
    words: Set[str] = set()
    
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                word = line.strip()
                
                # Skip empty lines and comments
                if not word or word.startswith("#"):
                    continue
                
                # Normalize if requested
                if normalize:
                    word = word.lower()
                
                words.add(word)
        
        frozen = frozenset(words)
        _cache[cache_key] = frozen
        
        logger.debug(f"Loaded {len(words)} words from '{filename}'")
        
        return frozen
        
    except Exception as exc:
        _stats.failed_loads += 1
        logger.error(f"Failed to load wordlist '{filename}': {exc}")
        
        if required:
            raise
        return frozenset()


def preload_wordlists(wordlists: Optional[list] = None) -> Dict[str, frozenset]:
    """
    Preload critical wordlists into cache.
    
    Args:
        wordlists: List of wordlist filenames to preload
                   (uses CRITICAL_WORDLISTS if None)
                   
    Returns:
        Dictionary of loaded wordlists
    """
    if wordlists is None:
        wordlists = CRITICAL_WORDLISTS
    
    logger.info(f"Preloading {len(wordlists)} critical wordlists...")
    
    loaded = {}
    
    for filename in wordlists:
        try:
            words = load_wordlist(filename, required=False)
            loaded[filename] = words
            _preloaded.add(filename)
            logger.debug(f"  ✓ Preloaded '{filename}' ({len(words)} words)")
        except Exception as exc:
            logger.warning(f"  ✗ Failed to preload '{filename}': {exc}")
    
    logger.info(f"Preloading complete: {len(loaded)}/{len(wordlists)} successful")
    
    return loaded


def is_preloaded(filename: str) -> bool:
    """Check if a wordlist has been preloaded."""
    return filename in _preloaded


def get_loaded_wordlists() -> Set[str]:
    """Get set of all loaded wordlist filenames."""
    return set(_cache.keys())


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def load_headers() -> frozenset:
    """Load common headers wordlist."""
    return load_wordlist("common_headers.txt")


def load_locations() -> frozenset:
    """Load locations wordlist."""
    return load_wordlist("locations.txt")


def load_section_keywords() -> frozenset:
    """Load section keywords wordlist."""
    return load_wordlist("section_keywords.txt")


def load_skills() -> frozenset:
    """Load skills wordlist."""
    return load_wordlist("skills.txt")


def load_programming_languages() -> frozenset:
    """Load programming languages wordlist."""
    return load_wordlist("programming_languages.txt")


def load_frameworks() -> frozenset:
    """Load frameworks wordlist."""
    return load_wordlist("frameworks.txt")


def load_databases() -> frozenset:
    """Load databases wordlist."""
    return load_wordlist("databases.txt")


def load_tools() -> frozenset:
    """Load tools wordlist."""
    return load_wordlist("tools.txt")


def load_tech_terms() -> frozenset:
    """Load technical terms wordlist."""
    return load_wordlist("tech_terms.txt")


def load_common_languages() -> frozenset:
    """Load common languages wordlist."""
    return load_wordlist("common_languages.txt")


def load_resume_terms() -> frozenset:
    """Load resume terms wordlist."""
    return load_wordlist("resume_terms.txt")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize() -> None:
    """
    Initialize keyword loader with preloaded wordlists.
    
    Call this at application startup for faster first-run performance.
    """
    logger.info("Initializing keyword loader...")
    preload_wordlists()
    logger.info("Keyword loader initialized")


# Auto-initialize on import (optional - can be disabled)
# initialize()
