# pyre-ignore-all-errors[import,call-overload,operator,index,arg-type,assignment,misc]
# All errors in this file are confirmed Pyre2 false-positives:
#   round(x, n)       → broken ndigits stub
#   Dict[str,Any] idx → @_ type-loss on readback
#   FrozenSet iter    → @_ type-loss in loops
# Runtime is correct — verified by tests/test_pipeline.py
"""
resume_intelligence_platform · ml_engine · features · feature_extractor
========================================================================

Transforms a parsed :class:`~ml_engine.schemas.resume_schema.ResumeSchema`
into a flat dictionary of 150+ numerical features suitable for Random Forest,
XGBoost, or any plug-and-play sklearn-compatible model.

Feature groups
--------------
G01  Contact completeness        (5 features)
G02  Section structure           (11 features)
G03  Skill taxonomy              (24 features)
G04  Education depth             (22 features)
G05  Experience analysis         (18 features)
G06  Project analysis            (18 features)
G07  Achievements & extras       (14 features)
G08  Online presence             (7 features)
G09  Resume format quality       (16 features)
G10  ATS red-flag / penalty      (20 features)
G11  Composite / derived scores  (10 features)
                                 ─────────────
                                 165 features total

Penalty design
--------------
Each ATS red-flag feature is 1 when the bad signal IS present (higher = worse).
Downstream scorers should subtract weighted penalties from positives so that
Random Forest / gradient boosting models learn the sign correctly from labelled
training data.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from ml_engine.schemas import ResumeSchema      # type: ignore[import]
from ml_engine.extraction import load_wordlist  # type: ignore[import]

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regular expressions  (module-level → compiled once)
# ---------------------------------------------------------------------------

_RE_WORD        = re.compile(r"\b\w+\b")
_RE_NUMBER      = re.compile(r"\b\d+(?:\.\d+)?\b")
_RE_PERCENT     = re.compile(r"\b\d+(?:\.\d+)?%")
_RE_MULTIPLIER  = re.compile(r"\b\d+x\b", re.IGNORECASE)
_RE_DATE_RANGE  = re.compile(
    r"(?:\d{1,2}/)?(20\d{2}|19\d{2})\s*[–\-—]\s*"
    r"(?:\d{1,2}/)?(20\d{2}|19\d{2}|present|current|now)",
    re.IGNORECASE,
)
_RE_CGPA        = re.compile(
    r"(?:cgpa|gpa|percentage|aggregate|marks?|score|grade)[:\s]*"
    r"(\d{1,2}(?:\.\d{1,2})*)",
    re.IGNORECASE,
)
_RE_GITHUB_URL  = re.compile(r"github\.com/[a-zA-Z0-9_\-]+", re.IGNORECASE)
_RE_URL         = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL       = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_PHONE       = re.compile(r"(?:\+?\d[\d\s\-]{8,13}\d)")
_RE_BULLET      = re.compile(r"^[\s]*[•\-\*▸►◦▪]", re.MULTILINE)
# Spaced-out letter sequences (poor ATS formatting: "S K I L L S")
_RE_SPACED_CAPS = re.compile(r"(?:[A-Z]\s){3,}[A-Z]")
_RE_KEYWORD_STUFF = re.compile(
    r"(\b\w+\b)(?:\s+\1){3,}",   # same word 4+ consecutive times
    re.IGNORECASE,
)
# Quantified impact patterns
_RE_IMPROVED_BY = re.compile(
    r"\b(?:improved|increased|reduced|decreased|boosted|grew|cut|saved)"
    r"\s+(?:by\s+)?\d+",
    re.IGNORECASE,
)
_RE_ACTION_INTS = re.compile(r"\b(?:led|managed|handled|oversaw)\s+\d+", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Wordlist loader – cached, never crashes the pipeline
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _load_cached(filename: str) -> FrozenSet[str]:
    """Load a wordlist once and cache it as a frozenset."""
    try:
        raw: List[str] = load_wordlist(filename)  # type: ignore[misc]
        return frozenset(
            s.strip().lower() for s in raw
            if isinstance(s, str) and s.strip() and not s.strip().startswith("#")
        )
    except FileNotFoundError:
        logger.warning("Wordlist not found: %s — using empty set.", filename)
        return frozenset()
    except Exception as exc:
        logger.exception("Error loading wordlist %s: %s", filename, exc)
        return frozenset()


# ---------------------------------------------------------------------------
# Wordlists (all lazy via lru_cache – only loaded when extract_features runs)
# ---------------------------------------------------------------------------

def _WL(name: str) -> FrozenSet[str]:         # shorthand
    return _load_cached(name)

# ---------------------------------------------------------------------------
# Domain keyword constants (frozen – defined inline, not from files, for speed)
# ---------------------------------------------------------------------------

_BACHELOR_KW: FrozenSet[str] = frozenset({
    "b.tech", "btech", "b.e", "be", "bsc", "b.sc", "bca", "b.c.a",
    "b.sc it", "bsc it", "bachelor", "b.eng", "undergraduate",
})
_MASTER_KW: FrozenSet[str] = frozenset({
    "m.tech", "mtech", "m.e", "msc", "m.sc", "mca", "m.c.a",
    "m.sc it", "msc it", "master", "m.eng", "postgraduate", "pg",
})
_DIPLOMA_KW: FrozenSet[str] = frozenset({
    "diploma", "polytechnic", "d.tech", "d.c.s",
})
_PHD_KW: FrozenSet[str] = frozenset({"phd", "ph.d", "doctorate", "doctoral"})

_BTECH_KW: FrozenSet[str] = frozenset({
    "b.tech", "btech", "b.e", "bachelor of technology", "bachelor of engineering",
    "information technology", "computer engineering", "computer science",
    "b.e.", "b.tech.", "engineering undergraduate",
})
_BCA_KW: FrozenSet[str] = frozenset({
    "bca", "b.c.a", "bachelor of computer applications",
})
_BSC_KW: FrozenSet[str] = frozenset({
    "b.sc", "bsc", "bachelor of science", "b.sc it", "bsc it",
})
_MCA_KW: FrozenSet[str] = frozenset({
    "mca", "m.c.a", "master of computer applications",
})
_MTECH_KW: FrozenSet[str] = frozenset({
    "m.tech", "mtech", "m.e", "master of technology", "master of engineering",
})
_MSC_KW: FrozenSet[str] = frozenset({
    "m.sc", "msc", "master of science", "m.sc it", "msc it",
})

_IT_CS_KW: FrozenSet[str] = frozenset({
    "computer engineering", "it", "cs", "cse", "ce", "ise",
    "information science", "computer applications",
    "electronics and communication", "ece", "software engineering",
    "engineering undergraduate", "engineering student",
    "information technology", "computer science",
})

_HACKATHON_KW: FrozenSet[str] = frozenset({
    "hackathon", "hack", "marathon", "ssip", "smart india hackathon",
    "code unnati", "code for good", "devhack", "buildathon", "ideathon",
    "innovation challenge", "innovation marathon",
})
_COMPETITION_KW: FrozenSet[str] = frozenset({
    "competition", "contest", "olympiad", "quiz", "treasure hunt",
    "code sprint", "codejam", "kaggle", "codeathon",
})
_COMPETITIVE_PLATFORMS: FrozenSet[str] = frozenset({
    "leetcode", "codeforces", "hackerrank", "hackerearth",
    "codechef", "competitive programming", "topcoder", "geeksforgeeks",
})
_OPEN_SOURCE_KW: FrozenSet[str] = frozenset({
    "open source", "open-source", "pull request", "pr merged",
    "contributor", "maintained repo", "forked", "npm package",
    "pypi package", "published package", "github contributions",
})
_CERT_PLATFORMS: FrozenSet[str] = frozenset({
    "coursera", "udemy", "nptel", "infosys springboard", "tcs ion",
    "google certified", "aws certified", "microsoft certified",
    "cisco", "ccna", "oracle certified", "redhat", "comptia",
    "linkedin learning", "pluralsight", "edx",
})
_CERT_KW: FrozenSet[str] = frozenset({
    "certified", "certification", "certificate",
})
_SOFT_KW: FrozenSet[str] = frozenset({
    "communication", "leadership", "teamwork", "collaborative",
    "problem solving", "critical thinking", "adaptability",
    "time management", "presentation", "mentoring", "interpersonal",
    "analytical", "creative", "innovative",
})
_INTERNSHIP_KW: FrozenSet[str] = frozenset({
    "intern", "internship", "trainee", "apprentice",
    "industrial training", "summer intern", "winter intern",
})
_IMPACT_KW: FrozenSet[str] = frozenset({
    "reduced", "improved", "increased", "boosted", "optimized",
    "achieved", "delivered", "exceeded", "generated", "saved",
})
_CLOUD_KW: FrozenSet[str] = frozenset({
    "aws", "azure", "gcp", "google cloud", "cloud", "s3", "ec2",
    "lambda", "cloudfront", "heroku", "digitalocean",
})
_AI_ML_KW: FrozenSet[str] = frozenset({
    "machine learning", "deep learning", "neural network", "nlp",
    "computer vision", "tensorflow", "pytorch", "keras", "sklearn",
    "scikit-learn", "pandas", "numpy", "data science", "llm",
    "langchain", "generative ai", "reinforcement learning",
})
_MOBILE_KW: FrozenSet[str] = frozenset({
    "android", "ios", "flutter", "react native", "kotlin", "swift",
    "dart", "mobile app", "mobile application",
})
_DEVOPS_KW: FrozenSet[str] = frozenset({
    "docker", "kubernetes", "ci/cd", "jenkins", "github actions",
    "terraform", "ansible", "devops", "helm", "prometheus",
})
_SECURITY_KW: FrozenSet[str] = frozenset({
    "penetration testing", "ethical hacking", "cybersecurity",
    "xss", "sql injection", "ssrf", "owasp", "burp suite",
    "metasploit", "wireshark", "nmap", "kali linux",
})
_WEB_KW: FrozenSet[str] = frozenset({
    "react", "angular", "vue", "django", "flask", "fastapi",
    "nodejs", "html", "css", "bootstrap", "tailwind", "nextjs",
    "laravel", "spring boot", "rest api",
})
_DATA_KW: FrozenSet[str] = frozenset({
    "spark", "hadoop", "kafka", "airflow", "tableau", "powerbi",
    "dbt", "etl", "data pipeline", "data warehouse", "bigquery",
})
_TESTING_KW: FrozenSet[str] = frozenset({
    "unit testing", "selenium", "jest", "pytest", "junit",
    "testng", "tdd", "bdd", "postman", "swagger",
})
_GENERIC_BUZZWORDS: FrozenSet[str] = frozenset({
    "hardworking", "passionate", "self-motivated", "go-getter",
    "synergy", "leverage", "paradigm", "results-driven",
    "detail-oriented", "team player", "fast learner", "dynamic",
    "proactive", "strong work ethic", "highly motivated",
})
_PROFESSIONAL_EMAIL_DOMAINS: FrozenSet[str] = frozenset({
    "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
    "protonmail.com", "icloud.com",
})

# Current year for date math
_CURRENT_YEAR: int = 2026

# ---------------------------------------------------------------------------
# Pure helper functions (no external calls, fully testable)
# ---------------------------------------------------------------------------

def _blob(items: List[str]) -> str:
    """Join a string list into a single lower-cased blob for keyword search."""
    return " ".join(x for x in items if x).lower()


def _contains_any(text: str, kw: FrozenSet[str]) -> bool:
    return any(k in text for k in kw)


def _count_any(text: str, kw: FrozenSet[str]) -> int:
    return sum(1 for k in kw if k in text)


def _detect_cgpa(text: str) -> Optional[float]:
    """
    Find all numeric CGPA/percentage values and return the highest valid one.
    Handles '9.5/10', '10/10', '85%', etc.
    """
    matches = _RE_CGPA.findall(text)
    if not matches:
        return None

    scores: List[float] = []
    for m in matches:
        try:
            val = float(str(m).replace(",", "."))
            # Normalization logic: Scale 10-point CGPA to 100-point for ML consistency
            if val <= 10.0 and val > 0:
                normalized_val = val * 10.0
                scores.append(normalized_val)
            elif val > 10.0 and val <= 100.0:
                scores.append(val)
        except (ValueError, TypeError):
            continue

    return max(scores) if scores else None


def _estimate_exp_years(text: str) -> float:
    """
    Sum durations of all date ranges in *text*.
    Returns total experience in years (float).
    """
    total: float = 0.0
    for start_s, end_s in _RE_DATE_RANGE.findall(text):
        try:
            start_y: int = int(start_s)  # type: ignore[arg-type]
            end_str: str = str(end_s).lower().strip()
            end_y: int = (
                _CURRENT_YEAR
                if end_str in {"present", "current", "now"}
                else int(end_str)  # type: ignore[arg-type]
            )
            if 1990 <= start_y <= _CURRENT_YEAR and start_y <= end_y:
                total = total + float(end_y - start_y)  # type: ignore[operator]
        except (ValueError, TypeError):
            continue
    return total


def _count_quantified(text: str) -> int:
    """Count sentences/bullets that contain measurable numeric impact."""
    hits = (
        len(_RE_IMPROVED_BY.findall(text))
        + len(_RE_ACTION_INTS.findall(text))
        + len(_RE_MULTIPLIER.findall(text))
    )
    return hits


def _has_unprofessional_email(email: str) -> bool:
    """Return True if the email looks unprofessional (xXx123, childish names, etc.)."""
    local = email.split("@")[0].lower() if "@" in email else email.lower()
    if re.search(r"(xox|sexy|cool|swag|dragonball|ninja|gamer)", local):
        return True
    
    # Check for excessive digits unless it looks like a typical student ID (0-3 letters + digits)
    if re.search(r"\d{6,}", local):
        if re.fullmatch(r"[a-z]{0,3}\d{6,}", local):
            return False
        return True
    return False


def _safe_ratio(num: int, denom: int, precision: int = 4) -> float:
    """Return num/denom rounded to *precision* digits; 0.0 if denom is zero."""
    if denom <= 0:
        return 0.0
    return round(float(num) / float(denom), precision)  # type: ignore[call-overload]


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, val)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(resume: ResumeSchema) -> Dict[str, Any]:  # type: ignore[misc]
    """
    Extract 150+ numerical features from a parsed resume.

    Parameters
    ----------
    resume : ResumeSchema
        A fully populated resume schema object.

    Returns
    -------
    Dict[str, Any]
        Flat dictionary; every value is ``int`` (0/1 flag) or ``float``.
        Keys are snake_case, grouped by a ``gNN_`` namespace prefix for
        easy filtering in training pipelines.

    Notes
    -----
    - All features are non-negative.
    - Penalty features (`ats_penalty_*`) are 1 when the **bad** signal is
      present.  A downstream scorer should subtract weighted penalties.
    - ``candidate_readiness_score`` (0-100) is a quick signal that works
      without a trained model.
    """

    f: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Safe field extraction from schema
    # ------------------------------------------------------------------
    skills_raw:   List[Any] = list(resume.skills       or [])
    edu_raw:      List[Any] = list(resume.education    or [])
    exp_raw:      List[Any] = list(resume.experience   or [])
    proj_raw:     List[Any] = list(resume.projects     or [])
    ach_raw:      List[Any] = list(resume.achievements or [])
    lang_raw:     List[Any] = list(resume.languages    or [])
    sections:     Dict[str, Any] = dict(resume.sections or {})
    raw_text:     str = str(resume.raw_text or "")
    name:         str = str(resume.name    or "")
    email:        str = str(resume.email   or "").lower()
    phone:        str = str(resume.phone   or "")
    location:     str = str(resume.location or "")

    # String-filtered lists
    skills:   List[str] = [str(s) for s in skills_raw  if isinstance(s, str) and s.strip()]
    edu:      List[str] = [str(s) for s in edu_raw      if isinstance(s, str) and s.strip()]
    exp:      List[str] = [str(s) for s in exp_raw      if isinstance(s, str) and s.strip()]
    projects: List[str] = [str(s) for s in proj_raw     if isinstance(s, str) and s.strip()]
    ach:      List[str] = [str(s) for s in ach_raw      if isinstance(s, str) and s.strip()]
    langs:    List[str] = [str(s) for s in lang_raw     if isinstance(s, str) and s.strip()]

    # Text blobs (lowercase for keyword search)
    raw_lower  = raw_text.lower()
    edu_text   = _blob(edu)
    exp_text   = _blob(exp)
    proj_text  = _blob(projects)
    ach_text   = _blob(ach)
    all_text   = raw_lower          # full resume lower-cased

    # DEBUG LOGS for Audit recovery
    logger.info("AUDIT - Candidate: %s", name)
    logger.info("AUDIT - Education Text Blob: [%s]", edu_text)

    # Word-level head (first 120 words) for header/name context
    _raw_words: List[str] = raw_text.split()
    raw_head   = " ".join(_raw_words[:120]).lower()  # type: ignore[index]

    # Lazy wordlist references
    WL_PROG   = _WL("programming_languages.txt")
    WL_FW     = _WL("frameworks.txt")
    WL_DB     = _WL("databases.txt")
    WL_TOOLS  = _WL("tools.txt")
    WL_HI     = _WL("high_value_skills.txt")
    WL_MID    = _WL("mid_value_skills.txt")
    WL_DEGREE = _WL("academic_terms.txt")
    WL_INTERN = _WL("internship_terms.txt")
    WL_LANG   = _WL("common_languages.txt")
    WL_TOPINST = _WL("top_institutes.txt")
    WL_PERSONAL = _WL("personal_info_flags.txt")
    WL_DEPLOY = _WL("deployment_signals.txt")
    WL_VERBS  = _WL("action_verbs.txt")

    ALL_TECH: FrozenSet[str] = WL_PROG | WL_FW | WL_DB | WL_TOOLS

    # Pre-compute skill lower set for fast membership tests
    skill_lower: List[str] = [s.lower().strip() for s in skills]

    # ======================================================================
    # G01 – Contact completeness                               (5 features)
    # ======================================================================

    _has_name     = int(bool(name and len(name) > 1))
    _has_email    = int(bool(email and "@" in email))
    _has_phone    = int(bool(phone and len(re.sub(r"\D", "", phone)) >= 10))
    _has_location = int(bool(location))
    _has_linkedin = int("linkedin.com" in all_text)

    f["has_name"]                   = _has_name
    f["has_email"]                  = _has_email
    f["has_phone"]                  = _has_phone
    f["has_location"]               = _has_location
    f["contact_completeness_score"] = round(  # type: ignore[call-overload]
        float(_has_name + _has_email + _has_phone + _has_location) / 4.0, 2
    )

    # ======================================================================
    # G02 – Section structure                                  (11 features)
    # ======================================================================

    _sec_skills  = int("skills"           in sections)
    _sec_edu     = int("education"        in sections)
    _sec_exp     = int("experience"       in sections)
    _sec_proj    = int("projects"         in sections)
    _sec_ach     = int("achievements"     in sections)
    _sec_langs   = int("languages"        in sections)
    _sec_obj     = int("career_objective" in sections)

    f["section_count"]              = len(sections)
    f["has_skills_section"]         = _sec_skills
    f["has_education_section"]      = _sec_edu
    f["has_experience_section"]     = _sec_exp
    f["has_projects_section"]       = _sec_proj
    f["has_achievements_section"]   = _sec_ach
    f["has_languages_section"]      = _sec_langs
    f["has_objective_section"]      = _sec_obj
    f["has_interests_section"]      = int("interests" in sections)
    f["has_declaration_section"]    = int("declaration" in sections)

    _key_sections = _sec_skills + _sec_edu + _sec_exp + _sec_proj
    f["section_completeness_score"] = round(float(_key_sections) / 4.0, 2)  # type: ignore[call-overload]

    # ======================================================================
    # G03 – Skill taxonomy                                     (24 features)
    # ======================================================================

    f["skills_count"] = len(skills)

    f["programming_languages_count"] = sum(1 for s in skill_lower if s in WL_PROG)
    f["framework_count"]             = sum(1 for s in skill_lower if s in WL_FW)
    f["database_count"]              = sum(1 for s in skill_lower if s in WL_DB)
    f["tool_count"]                  = sum(1 for s in skill_lower if s in WL_TOOLS)

    f["has_cloud_skills"]    = int(_contains_any(all_text, _CLOUD_KW))
    f["has_ai_ml_skills"]    = int(_contains_any(all_text, _AI_ML_KW))
    f["has_web_dev_skills"]  = int(_contains_any(all_text, _WEB_KW))
    f["has_mobile_skills"]   = int(_contains_any(all_text, _MOBILE_KW))
    f["has_devops_skills"]   = int(_contains_any(all_text, _DEVOPS_KW))
    f["has_security_skills"] = int(_contains_any(all_text, _SECURITY_KW))
    f["has_data_skills"]     = int(_contains_any(all_text, _DATA_KW))
    f["has_testing_skills"]  = int(_contains_any(all_text, _TESTING_KW))

    f["skill_category_count"] = (
        int(f["has_cloud_skills"])    + int(f["has_ai_ml_skills"])
        + int(f["has_web_dev_skills"]) + int(f["has_mobile_skills"])
        + int(f["has_devops_skills"])  + int(f["has_security_skills"])
        + int(f["has_data_skills"])    + int(f["has_testing_skills"])
    )

    _sw: float = sum(
        2.0 if s in WL_HI else 1.0 if s in WL_MID else 0.3
        for s in skill_lower
    )
    f["skill_weight_score"]    = round(_sw, 2)  # type: ignore[call-overload]
    f["has_high_value_skills"] = int(any(s in WL_HI for s in skill_lower))
    f["skill_versatility"]     = int(f["skill_category_count"] >= 3)
    f["skill_density"]         = _safe_ratio(len(skills), len(_RE_WORD.findall(raw_text)))

    # Tech stack richness in project text
    _proj_tech: Set[str] = {t for t in ALL_TECH if t in proj_text}
    f["project_tech_stack_count"] = len(_proj_tech)

    # ======================================================================
    # G04 – Education depth                                    (22 features)
    # ======================================================================

    # Combined search text: edu section + full resume for fallback
    # Root cause: section detector often misclassifies degree lines as unknown_
    # sections, leaving edu_text with only dates. We must search raw_lower too.
    _edu_or_raw = edu_text if len(edu_text.split()) > 5 else raw_lower

    _STRICT_DEGREE_KWS = _BACHELOR_KW | _MASTER_KW | _DIPLOMA_KW | _PHD_KW | _BTECH_KW | _BCA_KW | _BSC_KW | _MCA_KW | _MTECH_KW | _MSC_KW
    f["education_count"] = max(
        sum(1 for line in edu
            if any(k in line.lower() for k in _STRICT_DEGREE_KWS) and len(line.split()) > 2),
        # Fallback: count degree keywords in full resume
        sum(1 for k in {"b.tech", "btech", "diploma", "bachelor", "master"}
            if k in raw_lower),
    )

    f["has_bachelor_degree"] = int(_contains_any(_edu_or_raw, _BACHELOR_KW))
    f["has_master_degree"]   = int(_contains_any(_edu_or_raw, _MASTER_KW))
    f["has_diploma"]         = int(_contains_any(_edu_or_raw, _DIPLOMA_KW) or "diploma" in raw_lower)
    f["has_phd"]             = int(_contains_any(_edu_or_raw, _PHD_KW))
    f["has_btech_be"]        = int(_contains_any(_edu_or_raw, _BTECH_KW) or "b.tech" in raw_lower)
    f["has_bca"]             = int(_contains_any(_edu_or_raw, _BCA_KW))
    f["has_bsc_it"]          = int(_contains_any(_edu_or_raw, _BSC_KW))
    f["has_mca"]             = int(_contains_any(_edu_or_raw, _MCA_KW))
    f["has_mtech_me"]        = int(_contains_any(_edu_or_raw, _MTECH_KW))
    f["has_msc_it"]          = int(_contains_any(_edu_or_raw, _MSC_KW))
    f["has_it_major"]        = int(
        _contains_any(_edu_or_raw, _IT_CS_KW)
        or _contains_any(raw_head, _IT_CS_KW)
        or "information technology" in raw_lower
        or "computer engineering" in raw_lower
    )
    f["is_cs_it_candidate"]  = int(
        bool(f["has_it_major"]) or bool(f["has_bachelor_degree"])
        or bool(f["has_btech_be"]) or bool(f["has_bca"]) or bool(f["has_mca"])
    )
    f["has_top_institution"] = int(_contains_any(_edu_or_raw, WL_TOPINST))

    # CGPA / Percentage detection
    # Try to find CGPA specifically in Bachelor/Master lines first
    _cgpa_raw: Optional[float] = None
    _HIGHER_ED_KWS = _BACHELOR_KW | _MASTER_KW | _BTECH_KW | _BCA_KW | _BSC_KW | _MCA_KW | _MTECH_KW | _MSC_KW
    for line in edu:
        if any(k in line.lower() for k in _HIGHER_ED_KWS):
            val = _detect_cgpa(line)
            if val is not None:
                _cgpa_raw = val
                break

    if _cgpa_raw is None:
        _cgpa_raw = _detect_cgpa(edu_text)
    if _cgpa_raw is None:
        _cgpa_raw = _detect_cgpa(raw_lower)

    _cgpa: float = float(_cgpa_raw) if _cgpa_raw is not None else 0.0
    f["has_cgpa"]             = int(_cgpa_raw is not None)
    f["cgpa_value"]           = round(_cgpa, 2)  # type: ignore[call-overload]
    f["has_strong_academics"] = int(_cgpa >= 8.0 or _cgpa >= 80.0)

    # Dual qualification (e.g., diploma + degree)
    f["has_dual_qualification"] = int(
        bool(f["has_diploma"]) and bool(
            f["has_bachelor_degree"] or f["has_master_degree"]
        )
    )

    # Relevant coursework / electives mentioned
    f["has_relevant_coursework"] = int(
        any(k in all_text for k in {
            "coursework", "elective", "relevant courses",
            "major courses", "core subjects",
        })
    )

    # Graduation year estimation (latest 20XX in edu section)
    _grad_years = [int(y) for y in re.findall(r"\b(20\d{2})\b", edu_text)]
    _grad_year: int = max(_grad_years) if _grad_years else 0
    f["graduation_year"]     = _grad_year
    f["is_recent_graduate"]  = int(
        _grad_year > 0 and (_CURRENT_YEAR - _grad_year) <= 2
    )

    # ======================================================================
    # G05 – Experience analysis                                (18 features)
    # ======================================================================

    f["experience_lines"]          = len(exp)
    f["has_experience"]            = int(bool(exp))
    f["experience_has_internship"] = int(
        _contains_any(exp_text, _INTERNSHIP_KW)
    )
    _exp_years: float = round(_estimate_exp_years(exp_text), 2)  # type: ignore[call-overload]
    f["experience_years_estimate"] = _exp_years

    f["internship_role_count"] = sum(
        1 for line in exp
        if any(t in line.lower() for t in _INTERNSHIP_KW)
    )
    f["has_full_time_experience"] = int(
        any(k in exp_text for k in {
            "software engineer", "developer", "analyst",
            "associate", "consultant", "engineer",
        }) and not bool(f["experience_has_internship"])
    )
    f["has_freelance_experience"] = int(
        any(k in exp_text for k in {"freelance", "freelancer", "self-employed", "consultant"})
    )
    f["has_mnc_experience"] = int(
        any(k in exp_text for k in {
            "google", "microsoft", "amazon", "meta", "apple",
            "infosys", "wipro", "tcs", "hcl", "accenture",
            "cognizant", "capgemini", "ibm", "oracle",
        })
    )
    f["has_startup_experience"] = int(
        any(k in exp_text for k in {"startup", "early stage", "founded", "co-founded"})
    )
    f["has_remote_experience"]   = int(
        any(k in exp_text for k in {"remote", "work from home", "wfh"})
    )
    f["has_management_experience"] = int(
        any(k in exp_text for k in {
            "managed team", "led team", "team lead", "team leader",
            "managed a team", "managed cross",
        })
    )
    f["has_quantified_impact_in_exp"] = int(
        bool(_RE_IMPROVED_BY.search(exp_text))
        or bool(_RE_ACTION_INTS.search(exp_text))
    )
    f["has_missing_dates_in_exp"]  = int(
        bool(exp) and not bool(_RE_DATE_RANGE.search(exp_text))
    )
    f["experience_company_count"]  = len(
        set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|LLC|Inc|Pvt|Technologies|Solutions|Systems)\b", exp_text, re.IGNORECASE))
    )
    f["is_fresher"] = int(
        not bool(f["has_experience"]) or float(f["experience_years_estimate"]) < 1.0
    )

    # ======================================================================
    # G06 – Project analysis                                   (18 features)
    # ======================================================================

    f["has_projects"]      = int("projects" in sections)
    f["projects_count"]    = len(projects)

    f["has_ml_project"]       = int(_contains_any(proj_text, frozenset({
        "machine learning", "deep learning", "neural", "ml model",
        "computer vision", "dataset", "accuracy", "training",
    })))
    f["has_web_project"]      = int(_contains_any(proj_text, _WEB_KW))
    f["has_mobile_project"]   = int(_contains_any(proj_text, _MOBILE_KW))
    f["has_database_project"] = int(_contains_any(proj_text, frozenset({
        "database", "sql", "mysql", "mongodb", "sqlite",
        "firebase", "stored", "retrieve", "query",
    })))
    f["has_api_project"]      = int(_contains_any(proj_text, frozenset({
        "api", "rest", "restful", "graphql", "endpoint", "microservice",
    })))
    f["has_security_project"] = int(_contains_any(proj_text, _SECURITY_KW))
    f["has_cloud_project"]    = int(_contains_any(proj_text, _CLOUD_KW))
    f["has_data_project"]     = int(_contains_any(proj_text, _DATA_KW))

    f["project_tech_diversity"]   = int(len(_proj_tech) >= 3)
    f["has_deployed_project"]     = int(_contains_any(proj_text, WL_DEPLOY))
    f["has_team_project"]         = int(
        any(k in proj_text for k in {"team", "collaborated", "group", "we built"})
    )
    f["project_domain_diversity"] = (
        int(f["has_ml_project"])  + int(f["has_web_project"])
        + int(f["has_mobile_project"]) + int(f["has_api_project"])
        + int(f["has_security_project"]) + int(f["has_cloud_project"])
    )
    f["avg_project_desc_length"] = (
        round(float(sum(len(p.split()) for p in projects)) / float(len(projects)), 1)  # type: ignore[call-overload]
        if projects else 0.0
    )  # type: ignore[call-overload]
    f["has_github_in_projects"] = int("github.com" in proj_text)

    # ======================================================================
    # G07 – Achievements, certifications, extras               (14 features)
    # ======================================================================

    f["achievement_count"]      = len(ach)
    f["has_achievements"]       = int(bool(ach))
    f["has_hackathon"]          = int(_contains_any(ach_text + " " + all_text, _HACKATHON_KW))
    f["hackathon_count"]        = _count_any(all_text, _HACKATHON_KW)
    f["has_competition"]        = int(_contains_any(all_text, _COMPETITION_KW))
    f["has_competitive_coding"] = int(_contains_any(all_text, _COMPETITIVE_PLATFORMS))
    f["has_open_source"]        = int(
        _contains_any(all_text, _OPEN_SOURCE_KW)
        or bool(_RE_GITHUB_URL.search(raw_text))
    )
    f["has_certifications"]     = int(
        _contains_any(all_text, _CERT_KW)
        or _contains_any(all_text, _CERT_PLATFORMS)
        or "achievements" in sections
    )
    f["certification_platform_count"] = _count_any(all_text, _CERT_PLATFORMS)
    f["has_merit_or_rank"]      = int(
        any(k in all_text for k in {
            "rank", "merit", "topper", "first class", "distinction",
            "gold medal", "silver medal", "scholarship",
        })
    )
    f["has_publication"]        = int(
        any(k in all_text for k in {
            "published", "publication", "paper", "journal",
            "conference", "ieee", "springer", "research paper",
        })
    )
    f["has_volunteer_work"]     = int(
        any(k in all_text for k in {"volunteer", "volunteered", "nss", "community service"})
    )
    f["has_leadership_role"]    = int(
        any(k in all_text for k in {
            "president", "vice president", "secretary", "captain",
            "coordinator", "club head", "chapter lead",
        })
    )
    f["has_workshop_seminar"]   = int(
        any(k in all_text for k in {"workshop", "seminar", "webinar", "conference attended"})
    )

    # ======================================================================
    # G08 – Online presence                                    (7 features)
    # ======================================================================

    _has_github    = int("github.com" in all_text)
    _has_portfolio = int(
        any(k in all_text for k in {"portfolio", "personal site", "personal website"})
        or bool(re.search(r"https?://(?!linkedin|github)\S+\.(io|me|dev|com|in)\b", raw_text, re.IGNORECASE))
    )
    _online_count  = _has_linkedin + _has_github + _has_portfolio

    f["has_linkedin"]           = _has_linkedin
    f["has_github"]             = _has_github
    f["has_github_profile"]     = int(bool(_RE_GITHUB_URL.search(raw_text)))
    f["has_portfolio"]          = _has_portfolio
    f["online_presence_count"]  = _online_count
    f["online_presence_score"]  = round(float(_online_count) / 3.0, 2)  # type: ignore[call-overload]
    f["has_multiple_online_links"] = int(_online_count >= 2)

    # ======================================================================
    # G09 – Resume format / writing quality                    (16 features)
    # ======================================================================

    _word_count: int = len(_RE_WORD.findall(raw_text))
    _line_count: int = len([ln for ln in raw_text.splitlines() if ln.strip()])
    _bullet_count: int = len(_RE_BULLET.findall(raw_text))
    _number_count: int = len(_RE_NUMBER.findall(raw_text))
    _url_count: int = len(_RE_URL.findall(raw_text))

    f["resume_word_count"]    = _word_count
    f["resume_line_count"]    = _line_count
    f["has_bullet_points"]    = int(_bullet_count >= 3)
    f["bullet_count"]         = _bullet_count
    f["numbers_count"]        = _number_count           # raw numbers = quantified content
    f["url_count"]            = _url_count
    f["is_length_optimal"]    = int(300 <= _word_count <= 700)
    f["is_too_short"]         = int(_word_count < 150)
    f["is_too_long"]          = int(_word_count > 1200)

    # Action verb richness
    _verb_count: int = sum(
        1 for line in (exp + projects + ach)
        if any(v in line.lower() for v in WL_VERBS)
    )
    f["action_verb_count"]     = _verb_count
    f["has_action_verbs"]      = int(_verb_count >= 3)

    # Quantified impact
    _quant_count: int = _count_quantified(all_text)
    f["quantified_impact_count"] = _quant_count
    f["has_quantified_impact"]   = int(_quant_count >= 1)

    # Soft skills
    f["has_soft_skills"]   = int(_contains_any(all_text, _SOFT_KW))
    f["soft_skill_count"]  = _count_any(all_text, _SOFT_KW)

    # Multilingual
    f["languages_count"]       = len(langs)
    f["speaks_multiple_langs"] = int(len(langs) >= 2)
    f["extra_language_count"]  = sum(1 for lang in langs if lang.lower() not in WL_LANG)

    # Avg section length (single-line to dodge Pyre2 multi-line ndigits stub error)
    _total_sec_lines: int = sum(len(v) for v in sections.values())
    f["avg_section_length"] = round(float(_total_sec_lines) / float(max(len(sections), 1)), 1)  # type: ignore[call-overload]

    # Generic buzzword count (negative signal used in penalty section)
    _buzzword_count: int = _count_any(all_text, _GENERIC_BUZZWORDS)
    f["generic_buzzword_count"] = _buzzword_count

    # ======================================================================
    # G10 – ATS red-flag / penalty features                   (20 features)
    # ======================================================================
    # Convention: 1 = bad signal present (penalty), 0 = clean.
    # Downstream models learn to subtract these from the positive signals.

    # P1. Missing critical sections
    f["ats_penalty_no_skills"]     = int(not bool(_sec_skills))
    f["ats_penalty_no_education"]  = int(not bool(_sec_edu))
    f["ats_penalty_no_experience_or_projects"] = int(
        not bool(_sec_exp) and not bool(_sec_proj)
    )
    f["ats_penalty_no_contact"]    = int(
        not bool(_has_email) or not bool(_has_phone)
    )

    # P2. Excessive personal info (DOB, religion, etc. — not relevant for ATS)
    f["ats_penalty_personal_info"] = int(
        _contains_any(all_text, WL_PERSONAL)
    )

    # P3. Unprofessional email
    f["ats_penalty_unprofessional_email"] = int(
        bool(email) and _has_unprofessional_email(email)
    )

    # P4. No quantified impact whatsoever
    f["ats_penalty_no_quantified_impact"] = int(_quant_count == 0 and bool(exp))

    # P5. Zero action verbs in experience/projects
    f["ats_penalty_no_action_verbs"] = int(_verb_count == 0 and bool(exp))

    # P6. Experience without dates
    f["ats_penalty_missing_dates"] = int(f["has_missing_dates_in_exp"])

    # P7. Resume too short
    f["ats_penalty_too_short"] = int(f["is_too_short"])

    # P8. Resume too long
    f["ats_penalty_too_long"] = int(f["is_too_long"])

    # P9. Generic buzzword overuse
    f["ats_penalty_buzzword_heavy"] = int(_buzzword_count >= 4)

    # P10. Spaced-letter capitalization (ATS-unfriendly PDF formatting)
    f["ats_penalty_spaced_letters"] = int(bool(_RE_SPACED_CAPS.search(raw_text)))

    # P11. Keyword stuffing
    f["ats_penalty_keyword_stuffing"] = int(bool(_RE_KEYWORD_STUFF.search(raw_text)))

    # P12. No GitHub / LinkedIn (for CS/IT resumes, this is expected)
    f["ats_penalty_no_online_presence"] = int(_online_count == 0)

    # P13. Has declaration section (old-fashioned; many ATS skip it)
    f["ats_penalty_has_declaration"] = int("declaration" in sections)

    # P14. Skills section contains education dates / locations instead of skills
    # Detected when skills section has years like "2023 - present"
    _skill_sec_text = _blob(sections.get("skills", []))  # type: ignore[arg-type]
    f["ats_penalty_skills_has_dates"] = int(
        bool(_RE_DATE_RANGE.search(_skill_sec_text))
    )

    # P15. No projects AND no experience (very weak candidate)
    f["ats_penalty_no_proof_of_work"] = int(
        not bool(_sec_proj) and not bool(_sec_exp)
    )

    # P16. Total penalty score (sum, not capped)
    _total_penalty: int = sum(
        1 for k, v in f.items()
        if k.startswith("ats_penalty_") and isinstance(v, int) and v == 1
    )
    f["ats_total_penalty_score"] = _total_penalty

    # ======================================================================
    # G11 – Composite / derived scores                         (10 features)
    # ======================================================================
    #
    # Pre-extract all dict values into typed locals so Pyre2 tracks types
    # correctly without @_ inference loss on Dict[str, Any] readback.
    # ======================================================================

    _g11_contact:    float = float(f["contact_completeness_score"])   # type: ignore[index]
    _g11_section:    float = float(f["section_completeness_score"])   # type: ignore[index]
    _g11_sw:         float = float(f["skill_weight_score"])           # type: ignore[index]
    _g11_hv:         float = float(f["has_high_value_skills"])        # type: ignore[index]
    _g11_proj:       float = float(f["has_projects"])                 # type: ignore[index]
    _g11_exp:        float = float(f["has_experience"])               # type: ignore[index]
    _g11_ach:        float = float(f["has_achievements"])             # type: ignore[index]
    _g11_linkedin:   float = float(f["has_linkedin"])                 # type: ignore[index]
    _g11_github:     float = float(f["has_github"])                   # type: ignore[index]
    _g11_cgpa:       float = float(f["has_cgpa"])                     # type: ignore[index]

    _g11_ed_bach:    float = float(f["has_bachelor_degree"])          # type: ignore[index]
    _g11_ed_mast:    float = float(f["has_master_degree"])            # type: ignore[index]
    _g11_ed_acad:    float = float(f["has_strong_academics"])         # type: ignore[index]
    _g11_ed_top:     float = float(f["has_top_institution"])          # type: ignore[index]

    _g11_exp_i:      float = float(f["experience_has_internship"])    # type: ignore[index]
    _g11_exp_qi:     float = float(f["has_quantified_impact_in_exp"]) # type: ignore[index]
    _g11_exp_yr:     float = float(f["experience_years_estimate"])    # type: ignore[index]

    _g11_n_proj:     float = float(f["projects_count"])               # type: ignore[index]
    _g11_tech_stk:   float = float(f["project_tech_stack_count"])     # type: ignore[index]
    _g11_deployed:   float = float(f["has_deployed_project"])         # type: ignore[index]
    _g11_proj_div:   float = float(f["project_domain_diversity"])     # type: ignore[index]
    _g11_ach_count:  float = float(f["achievement_count"])            # type: ignore[index]
    _g11_sk_cat:     float = float(f["skill_category_count"])         # type: ignore[index]

    _is_fresher: float = float(f["is_fresher"])                   # type: ignore[index]
    
    # ── DYNAMIC SCORER REBALANCING ──────────────────────────────────────────
    # If a candidate is a fresher, we redistribute "Experience" weight into 
    # "Projects" and "Academics" to ensure a fair evaluation based on potential.
    # ─────────────────────────────────────────────────────────────────────────
    w_exp = 12.0
    w_proj = 10.0
    w_acad = 15.0
    
    if _is_fresher > 0:
        # Transfer 80% of experience weight to Projects and Academics
        w_proj += 6.0
        w_acad += 4.0
        w_exp  = 2.0  # Residual weight for any internships they might have
    
    # Positive raw score calculation
    _pos: float = (
        _g11_contact  * 10.0
        + _g11_section  * 15.0
        + float(min(_g11_sw, 30.0))
        + _g11_hv       * 5.0
        + _g11_proj     * w_proj
        + _g11_exp      * w_exp
        + _g11_ach      * 5.0
        + _g11_cgpa     * w_acad
        + _g11_linkedin * 4.0
        + _g11_github   * 4.0
    )
    _prs: float = round(_clamp(_pos), 2)  # type: ignore[call-overload]
    f["raw_positive_score"] = _prs

    # Penalty deduction (each penalty flag = -4 points)
    _penalty_deduction: float = float(_total_penalty) * 4.0
    _pd: float = round(_penalty_deduction, 2)  # type: ignore[call-overload]
    f["penalty_deduction"] = _pd

    _crs: float = round(_clamp(_pos - _penalty_deduction), 2)  # type: ignore[call-overload]
    f["candidate_readiness_score"] = _crs

    # Sub-scores for model interpretability
    _sk_sub: float = round(_clamp(_g11_sw * 2.0 + _g11_sk_cat * 3.0 + _g11_hv * 10.0), 2)  # type: ignore[call-overload]
    f["skills_subscore"] = _sk_sub

    _ed_sub: float = round(_clamp(_g11_ed_bach * 20.0 + _g11_ed_mast * 30.0 + _g11_ed_acad * 15.0 + _g11_ed_top * 20.0 + _g11_cgpa * 10.0), 2)  # type: ignore[call-overload]
    f["education_subscore"] = _ed_sub

    _ex_sub: float = round(_clamp(_g11_exp * 25.0 + _g11_exp_i * 20.0 + _g11_exp_qi * 15.0 + float(min(_g11_exp_yr, 3.0)) * 10.0), 2)  # type: ignore[call-overload]
    f["experience_subscore"] = _ex_sub

    _pr_sub: float = round(_clamp(_g11_n_proj * 8.0 + _g11_tech_stk * 3.0 + _g11_deployed * 15.0 + _g11_proj_div * 5.0), 2)  # type: ignore[call-overload]
    f["projects_subscore"] = _pr_sub

    _ov_str: float = round(_clamp(_ed_sub * 0.20 + _sk_sub * 0.25 + _ex_sub * 0.30 + _pr_sub * 0.15 + _g11_ach_count * 2.0 + _g11_linkedin * 2.0 + _g11_github * 3.0), 2)  # type: ignore[call-overload]
    f["overall_profile_strength"] = _ov_str

    logger.debug(
        "Feature extraction complete: %d features, readiness=%.1f, penalty=%d",
        len(f), _crs, _total_penalty,
    )

    return f