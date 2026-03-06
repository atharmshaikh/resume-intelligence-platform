"""
Simple typo detection module.
Counts spelling errors in resume text.
"""

import re
from spellchecker import SpellChecker
from ml_engine.extraction.keyword_loader import load_wordlist

TECH_WHITELIST = load_wordlist("tech_terms.txt")
RESUME_WORDS = load_wordlist("resume_terms.txt")
ACADEMIC_TERMS = load_wordlist("academic_terms.txt")
COMMON_DOMAINS = load_wordlist("domains.txt")
LOCATION_WHITELIST = load_wordlist("locations.txt")
INSTITUTE_WHITELIST = load_wordlist("institutes.txt")

WHITELIST = (
    TECH_WHITELIST
    | RESUME_WORDS
    | ACADEMIC_TERMS
    | COMMON_DOMAINS
    | LOCATION_WHITELIST
    | INSTITUTE_WHITELIST
)

EMAIL_PATTERN = re.compile(r"\S+@\S+")
URL_PATTERN = re.compile(r"https?://\S+")

MAX_SPELLCHECK_WORDS = 2000
MAX_TEXT_LENGTH = 100000

_spell = None

def _get_spellchecker():
    global _spell
    if _spell is None:
        _spell = SpellChecker(distance=1)
    return _spell

def count_typos(text: str):

    text = text[:MAX_TEXT_LENGTH]

    text = EMAIL_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    words = [w.lower() for w in words]  
    
    filtered_words = []

    for word in words:

        if word in WHITELIST:
            continue

        if word.isupper():
            continue

        if len(word) < 3:
            continue

        # Skip proper nouns (likely names)
        if word[0].isupper():
            continue
        # Skip merged technical tokens (OCR artifacts)
        if len(word) > 20:
            continue
        # Skip typical institute tokens
        if word.endswith("university") or word.endswith("institute"):
            continue

        filtered_words.append(word)

    spell = _get_spellchecker()
    filtered_words = filtered_words[:MAX_SPELLCHECK_WORDS]
    misspelled = set(spell.unknown(set(filtered_words))) 

    typo_count = len(misspelled)
    total_words = len(filtered_words)
    ratio = typo_count / total_words if total_words else 0

    return {
    "typo_count": typo_count,
    "total_words": total_words,
    "typo_ratio": round(ratio, 4),
    "typo_words": list(misspelled)
}


def typo_score(text: str):

    result = count_typos(text)

    total_words = result["total_words"]

    if total_words == 0:
        return 100

    typo_ratio = result["typo_ratio"]

    score = max(50, 100 - (typo_ratio * 200))
    return round(score, 2)