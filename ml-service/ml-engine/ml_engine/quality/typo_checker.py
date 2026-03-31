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
RESUME_ENTITIES = load_wordlist("resume_entities.txt")
SECURITY_TERMS = load_wordlist("security_terms.txt")
OCR_TERMS = load_wordlist("ocr_terms.txt")
INDIAN_EDU_TERMS = load_wordlist("common_indian_terms.txt")
NAME_WHITELIST = load_wordlist("names_whitelist.txt")
COMMON_WORDS = load_wordlist("common_words.txt")

WHITELIST = (
    TECH_WHITELIST
    | RESUME_WORDS
    | ACADEMIC_TERMS
    | COMMON_DOMAINS
    | LOCATION_WHITELIST
    | INSTITUTE_WHITELIST
    | RESUME_ENTITIES
    | SECURITY_TERMS
    | OCR_TERMS
    | INDIAN_EDU_TERMS
    | NAME_WHITELIST
    | COMMON_WORDS
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

def _is_probable_merged_word(word: str) -> bool:
    """
    Detect merged OCR tokens such as:
    'vehicleregistration' -> vehicle + registration
    """

    length = len(word)

    if length < 12:
        return False

    # try splitting word into two valid words
    for i in range(2, length - 2):

        left = word[:i]
        right = word[i:]

        if left in COMMON_WORDS and right in COMMON_WORDS:
            return True

    return False

def count_typos(text: str):

    text = text[:MAX_TEXT_LENGTH]

    text = EMAIL_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)

    WORD_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")
    words = WORD_PATTERN.findall(text)

    filtered_words = []

    for word in words:

        word = word.lower()

        if word in WHITELIST:
            continue

        if word.isupper():
            continue

        if len(word) <= 4:
            continue

        # Skip proper nouns (likely names)
        if word[0].isupper():
            continue

        # Skip merged technical tokens (OCR artifacts)
        if len(word) > 20:
            continue

        # Skip OCR merged tokens
        if _is_probable_merged_word(word):  
            continue

        # Skip typical institute tokens
        if word.endswith("university") or word.endswith("institute"):
            continue

        filtered_words.append(word)

    spell = _get_spellchecker()
    filtered_words = filtered_words[:MAX_SPELLCHECK_WORDS]
    misspelled = set(spell.unknown(filtered_words[:MAX_SPELLCHECK_WORDS]))

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