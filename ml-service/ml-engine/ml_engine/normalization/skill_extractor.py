"""
Skill extraction utilities.

Low-level skill extraction from raw text.
"""

import logging
import re
from typing import List, Set, Tuple

from ml_engine.extraction import load_wordlist

logger = logging.getLogger(__name__)

# Load skill wordlists
SKILL_WHITELIST = {
    s.lower()
    for s in (
        load_wordlist("programming_languages.txt")
        | load_wordlist("frameworks.txt")
        | load_wordlist("databases.txt")
        | load_wordlist("tools.txt")
        | load_wordlist("tech_terms.txt")
    )
}

TECH_KEYWORDS = {
    s.lower().strip()
    for s in SKILL_WHITELIST
    if s and str(s).strip()
}
TECH_KEYWORDS.update({"machine learning", "deep learning", "computer vision", "scnn"})

# Skill normalization mapping
_CANONICAL_ALIASES = {
    "js": "javascript",
    "java script": "javascript",
    "node js": "nodejs",
    "node.js": "nodejs",
    "mongo db": "mongodb",
    "express js": "express",
    "angular js": "angular",
    "angularjs": "angular",
    "react js": "react",
    "reactjs": "react",
    "my sql": "mysql",
}

_SKILL_NORMALIZATION = {}
for line in load_wordlist("skill_normalization.txt") or []:
    if ":" in line:
        src, dst = line.split(":", 1)
        _SKILL_NORMALIZATION[src.strip().lower()] = dst.strip()


def _normalize_skill_token(token: str) -> str:
    """Normalize a skill token to canonical form."""
    normalized = str(token or "").lower().strip()
    normalized = re.sub(r"\s+", " ", normalized.replace(".", " ")).strip()
    normalized = _CANONICAL_ALIASES.get(normalized, normalized)
    normalized = (_SKILL_NORMALIZATION.get(normalized, normalized) or "").strip().lower()
    normalized = _CANONICAL_ALIASES.get(normalized, normalized)
    return str(normalized)


def extract_skills(text: str | List[str]) -> Tuple[List[str], int]:
    """
    Extract skills from text.
    
    Args:
        text: Raw text or list of lines
        
    Returns:
        Tuple of (List of normalized skills, Typo count)
    """
    if isinstance(text, list):
        text = "\n".join(text)
    
    text = _normalize_bullets(text)
    tokens = re.split(r"[,\n|;&]", text)
    
    skills: List[str] = []
    seen: Set[str] = set()
    
    for token in tokens:
        skill = token.strip()
        if not skill:
            continue
        
        # Handle "Category: skill1" format aggressively regardless of category name
        if ":" in skill:
            skill = skill.split(":", 1)[1].strip()
        
        skill = skill.replace("(", "").replace(")", "")
        skill = re.sub(r"[–—−]", "-", skill)
        skill = re.sub(r"\s-\s", " ", skill)
        skill = skill.replace("&", "/")
        
    # 1. Cleaning & Delimiter homogenization
    text = text.replace("/", ",")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    
    # 2. Tokenization
    parts = re.split(r"[,\n|;&•\-*▸►◦▪]", text)
    skills = []
    seen = set()
    typo_count = 0
    
    from difflib import get_close_matches
    whitelist_list = list(SKILL_WHITELIST)

    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Handle "category: skill1, skill2"
        if ":" in part:
            subparts = part.split(":")
            for sp in subparts:
                sp_skills, sp_typos = extract_skills(sp)
                for s in sp_skills:
                    if s not in seen:
                        skills.append(s)
                        seen.add(s)
                typo_count += sp_typos
            continue
            
        skill_lower = part.lower()
        # Full normalization for whitelist check
        skill_norm = re.sub(r"[^a-z0-9#+ ]", "", skill_lower).strip()
        
        if len(skill_norm) > 1:
            if part.isdigit():
                continue
            
            # Ignore years/dates (even if nested in text)
            if re.search(r"\d{4}", part) or re.search(r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}", skill_lower):
                continue
            
            # Additional noise filter
            if any(noise in skill_lower for noise in ("soft skills", "teamwork", "communication", "problem-solving")):
                continue
            
            # 1. Exact Match
            if skill_norm in SKILL_WHITELIST:
                if skill_norm not in seen:
                    skills.append(skill_norm)
                    seen.add(skill_norm)
                continue
            
            # 2. Fuzzy Match (Typo Detection)
            if len(skill_norm) >= 4:
                matches = get_close_matches(skill_norm, whitelist_list, n=1, cutoff=0.88)
                if matches:
                    corrected = matches[0]
                    if corrected not in seen:
                        skills.append(corrected)
                        seen.add(corrected)
                        typo_count += 1
                continue
            
    return skills, typo_count


def _normalize_bullets(text: str) -> str:
    """Replace bullet characters with newlines."""
    for bullet in {"•", "●", "▪", "◦", "►", "▸", "■", "□", "◆", "◇", "-", "*"}:
        text = text.replace(bullet, "\n")
    return text
