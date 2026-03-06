"""
Load keyword lists used by extraction modules.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORDLIST_DIR = BASE_DIR / "wordlists"

_cache: dict[str, frozenset[str]] = {}


def load_wordlist(filename: str) -> frozenset[str]:
    """
    Load newline-separated wordlist.

    Supports:
    - comments (#)
    - blank lines
    """

    if filename in _cache:
        return _cache[filename]

    path = WORDLIST_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Wordlist not found: {path}")

    words: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:

            word = line.strip().lower()

            if not word or word.startswith("#"):
                continue

            words.add(word)

    frozen = frozenset(words)

    _cache[filename] = frozen

    return frozen