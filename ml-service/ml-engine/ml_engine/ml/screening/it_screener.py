"""
Deterministic IT resume screener.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


_SOFT_SKILLS = {
    "communication", "teamwork", "leadership", "problem solving", "time management",
    "adaptability", "critical thinking", "team building", "multitasking",
}
_TECH_ALIASES = {
    "js": "javascript",
    "node js": "nodejs",
    "mongo db": "mongodb",
}
_TECH_HINTS = {
    "python", "java", "javascript", "nodejs", "mongodb", "mysql", "sql", "html", "css",
    "react", "angular", "express", "pandas", "matplotlib", "firebase", "php", "deep learning",
    "computer vision", "scnn",
}
_RE_DATE_RANGE = re.compile(
    r"(?:\d{1,2}/)?(?:19|20)\d{2}\s*[-–—]\s*(?:\d{1,2}/)?(?:19|20)\d{2}",
    re.IGNORECASE,
)
_RE_LOCATION_NOISE = re.compile(r"\(|\)|software|technologies|solutions|systems|pvt|ltd|inc", re.IGNORECASE)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _decision(score: float) -> str:
    if score >= 70.0:
        return "SHORTLISTED"
    if score >= 45.0:
        return "MANUAL REVIEW"
    return "REJECTED"


def _subset_features(features: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "skills_count",
        "programming_languages_count",
        "framework_count",
        "database_count",
        "projects_count",
        "has_projects",
        "has_experience",
        "has_internship",
        "degree_type",
        "is_it_candidate",
        "score",
    ]
    return {k: features.get(k, 0) for k in keys}


def _clean_location(location: Any) -> Any:
    loc = str(location or "").strip()
    if not loc:
        return None
    if _RE_LOCATION_NOISE.search(loc):
        return None
    return loc


def _norm_skill(token: str) -> str:
    s = token.strip().lower()
    s = _TECH_ALIASES.get(s, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_refined_skills(normalized: Dict[str, Any]) -> List[str]:
    skills: List[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        n = _norm_skill(s)
        if not n or n in seen:
            return
        if n in _SOFT_SKILLS:
            return
        if len(n.split()) > 3:
            return
        if any(ch in n for ch in ".!?"):
            return
        if n not in _TECH_HINTS:
            return
        seen.add(n)
        skills.append(n)

    for s in normalized.get("skills", []) or []:
        add(str(s))

    for p in normalized.get("projects", []) or []:
        if not isinstance(p, dict):
            continue
        desc = str(p.get("description", "")).lower()
        for t in _TECH_HINTS:
            if re.search(rf"\b{re.escape(t)}\b", desc):
                add(t)

    for e in normalized.get("experience", []) or []:
        line = str(e).lower()
        for t in _TECH_HINTS:
            if re.search(rf"\b{re.escape(t)}\b", line):
                add(t)

    return skills


def _clean_project_name(name: str) -> str:
    n = re.sub(r"(?i)^\s*(project name:|project:|other projects?)\s*", "", name).strip()
    n = re.sub(r"\s{2,}", " ", n).strip(" -:")
    if n.upper() in {"PROJECTS", "OTHER PROJECTS", "PROJECT"}:
        return ""
    return n


def _extract_refined_projects(normalized: Dict[str, Any], refined_skills: List[str]) -> List[Dict[str, Any]]:
    refined: List[Dict[str, Any]] = []
    skill_set = set(refined_skills)

    for p in normalized.get("projects", []) or []:
        if not isinstance(p, dict):
            continue
        name = _clean_project_name(str(p.get("name", "")))
        desc = str(p.get("description", ""))
        desc = re.sub(r"(?i)\bdescription:\s*", "", desc).strip()
        desc = _RE_DATE_RANGE.sub("", desc)
        desc = re.sub(r"\s{2,}", " ", desc).strip(" ,.-")
        desc = re.sub(r"(?:\blike\b\s*)?$", "", desc, flags=re.IGNORECASE).strip()
        if not name or not desc:
            continue

        seen: set[str] = set()
        for t in _TECH_HINTS:
            if re.search(rf"\b{re.escape(t)}\b", desc.lower()) and t not in seen:
                seen.add(t)
                if t not in skill_set:
                    refined_skills.append(t)
                    skill_set.add(t)
        refined.append({
            "name": name,
            "description": desc,
        })
    return refined


def _simplify_experience(normalized: Dict[str, Any]) -> Dict[str, Any]:
    role = ""
    for line in normalized.get("experience", []) or []:
        line_str = str(line).strip()
        low = line_str.lower()
        if any(k in low for k in ("intern", "developer", "engineer", "analyst", "trainee", "frontend")):
            role = "frontend intern" if "frontend" in low and "intern" in low else line_str
            break
    return {
        "role": role,
        "has_experience": 1 if role else 0,
        "has_internship": 1 if role and "intern" in role.lower() else 0,
    }


def _minimal_education(normalized: Dict[str, Any], base_features: Dict[str, Any]) -> Dict[str, Any]:
    education = normalized.get("education", {}) or {}
    if not isinstance(education, dict):
        education = {}
    return {
        "degree_type": str(education.get("degree_type", "")).lower().strip(),
        "is_it_candidate": int(education.get("is_it_candidate", base_features.get("is_it_candidate", 0)) or 0),
        "score": float(education.get("score", base_features.get("score", 0)) or 0),
    }


def _compute_score(f: Dict[str, Any]) -> float:
    # Skills (50)
    skills_component = (
        min(float(f["skills_count"]) / 12.0, 1.0) * 0.4
        + min(float(f["programming_languages_count"]) / 4.0, 1.0) * 0.2
        + min(float(f["framework_count"]) / 3.0, 1.0) * 0.2
        + min(float(f["database_count"]) / 2.0, 1.0) * 0.2
    ) * 50.0

    # Projects (25)
    projects_component = (
        0.5 * float(bool(f["has_projects"]))
        + 0.5 * min(float(f["projects_count"]) / 2.0, 1.0)
    ) * 25.0

    # Experience (15)
    experience_component = (
        0.7 * float(bool(f["has_experience"]))
        + 0.3 * float(bool(f["has_internship"]))
    ) * 15.0

    # Education (10)
    score_norm = 0.0
    score_val = float(f["score"] or 0.0)
    if score_val > 0:
        if score_val <= 10.0:
            score_norm = min(score_val / 10.0, 1.0)
        else:
            score_norm = min(score_val / 100.0, 1.0)
    education_component = (
        0.5 * float(str(f["degree_type"]).lower() == "bachelor")
        + 0.3 * float(bool(f["is_it_candidate"]))
        + 0.2 * score_norm
    ) * 10.0

    return round(_clamp(skills_component + projects_component + experience_component + education_component), 2)


def screen_it_candidate(parsed_resume: Dict[str, Any]) -> Dict[str, Any]:
    identity = parsed_resume.get("identity", {}) or {}
    features = parsed_resume.get("features", {}) or {}
    normalized = parsed_resume.get("normalized_resume", {}) or {}

    refined_skills = _extract_refined_skills(normalized)
    refined_projects = _extract_refined_projects(normalized, refined_skills)
    refined_experience = _simplify_experience(normalized)
    refined_education = _minimal_education(normalized, features)

    identity = {
        "name": identity.get("name"),
        "email": identity.get("email"),
        "phone": identity.get("phone"),
        "location": _clean_location(identity.get("location")),
    }

    selected = _subset_features(features)
    selected["skills_count"] = len(refined_skills)
    selected["programming_languages_count"] = sum(1 for s in refined_skills if s in {"python", "java", "javascript", "php", "c", "c++"})
    selected["framework_count"] = sum(1 for s in refined_skills if s in {"react", "angular", "express", "nodejs", "django", "flask"})
    selected["database_count"] = sum(1 for s in refined_skills if s in {"mysql", "mongodb", "sql", "postgresql", "sqlite", "firebase"})
    selected["projects_count"] = len(refined_projects)
    selected["has_projects"] = 1 if refined_projects else 0
    selected["has_experience"] = refined_experience["has_experience"]
    selected["has_internship"] = refined_experience["has_internship"]
    selected["degree_type"] = refined_education["degree_type"]
    selected["is_it_candidate"] = refined_education["is_it_candidate"]
    selected["score"] = refined_education["score"]

    # Hard rules
    hard_reject_reasons = []
    if not identity.get("email"):
        hard_reject_reasons.append("no_email")
    if not identity.get("phone"):
        hard_reject_reasons.append("no_phone")
    if int(selected.get("skills_count", 0) or 0) == 0:
        hard_reject_reasons.append("no_skills")
    if int(selected.get("projects_count", 0) or 0) == 0 and int(selected.get("has_experience", 0) or 0) == 0:
        hard_reject_reasons.append("no_projects_and_no_experience")

    if hard_reject_reasons:
        return {
            "identity": identity,
            "normalized_data": {
                "skills": refined_skills,
                "projects": refined_projects,
                "experience": refined_experience,
                "education": refined_education,
            },
            "features": {
                "skills_count": selected["skills_count"],
                "programming_languages_count": selected["programming_languages_count"],
                "framework_count": selected["framework_count"],
                "database_count": selected["database_count"],
                "projects_count": selected["projects_count"],
                "has_projects": selected["has_projects"],
                "has_experience": selected["has_experience"],
                "has_internship": selected["has_internship"],
                "degree_type": selected["degree_type"],
                "is_it_candidate": selected["is_it_candidate"],
                "score": selected["score"],
            },
            "score": 0.0,
            "decision": "REJECTED",
            "extracted_features": selected,
            "final_score": 0.0,
            "final_decision": "REJECTED",
            "reject_reasons": hard_reject_reasons,
        }

    score = _compute_score(selected)
    return {
        "identity": identity,
        "normalized_data": {
            "skills": refined_skills,
            "projects": refined_projects,
            "experience": refined_experience,
            "education": refined_education,
        },
        "features": {
            "skills_count": selected["skills_count"],
            "programming_languages_count": selected["programming_languages_count"],
            "framework_count": selected["framework_count"],
            "database_count": selected["database_count"],
            "projects_count": selected["projects_count"],
            "has_projects": selected["has_projects"],
            "has_experience": selected["has_experience"],
            "has_internship": selected["has_internship"],
            "degree_type": selected["degree_type"],
            "is_it_candidate": selected["is_it_candidate"],
            "score": selected["score"],
        },
        "score": score,
        "decision": _decision(score),
        "extracted_features": selected,
        "final_score": score,
        "final_decision": _decision(score),
    }
