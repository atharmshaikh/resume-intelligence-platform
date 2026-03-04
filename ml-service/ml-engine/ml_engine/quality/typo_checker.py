"""
Simple typo detection module.
Counts spelling errors in resume text.
"""

from spellchecker import SpellChecker
import re

spell = SpellChecker(distance=1)

EMAIL_PATTERN = re.compile(r"\S+@\S+")
URL_PATTERN = re.compile(r"https?://\S+")

TECH_WHITELIST = {
    "python","excel","powerpoint","sql","linux",
    "tensorflow","pandas","numpy","java","javascript",
    "react","node","django","flask","aws","docker",
    "kubernetes","git","github","gitlab","fastapi",
    "postgresql","mongodb","mysql","redis","spark"
}

RESUME_WORDS = {
    "resume","project","skills","education","experience",
    "university","management","system","development",
    "engineering","technology","computer","science",
    "analysis","data","software","application",
    "technical","professional","objective","career","workflow"
}

ACADEMIC_TERMS = {
    "cgpa","gpa","bca","bsc","mca","msc",
    "gseb","cbse","icse","iit","iim",
    "university","college","institute"
}

COMMON_DOMAINS = {
    "gmail","yahoo","outlook","hotmail"
}
LOCATION_WHITELIST = {
    "anand",
    "khambhat",
    "limdi",
    "gujarat",
    "india"
}

INSTITUTE_WHITELIST = {
    "gshseb",
    "gseb",
    "cbse",
    "icse",
    "sp"
}
def count_typos(text: str):

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    
    filtered_words = []

    for w in words:

        word = w.lower()

        if word in TECH_WHITELIST:
            continue

        if word in RESUME_WORDS:
            continue

        if word in ACADEMIC_TERMS:
            continue

        if word in COMMON_DOMAINS:
            continue

        if word in LOCATION_WHITELIST:
            continue

        if word in INSTITUTE_WHITELIST:
            continue    

        if EMAIL_PATTERN.search(word):
            continue

        if URL_PATTERN.search(word):
            continue

        if len(word) < 3:
           continue

        filtered_words.append(word)

    misspelled = set(spell.unknown(filtered_words))

    typo_count = len(misspelled)
    total_words = len(filtered_words)

    return {
    "typo_count": typo_count,
    "total_words": total_words,
    "typo_ratio": typo_count / total_words if total_words else 0,
    "typo_words": list(misspelled)
}


def typo_score(text: str):

    result = count_typos(text)

    typo_count = result["typo_count"]
    total_words = result["total_words"]

    if total_words == 0:
        return 100

    typo_ratio = result["typo_ratio"]

    score = max(60, 100 - (typo_ratio * 250))
    return round(score, 2)