"""
Project extraction utilities.

Low-level project extraction from raw text.
"""

import logging
import re
from typing import List, TypedDict

from ml_engine.extraction import load_wordlist

logger = logging.getLogger(__name__)

# Project-related regex patterns
PROJECT_DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,/-]*\d{4}\b"
    r"|\b\d{4}\s*-\s*(?:present|\d{4})\b",
    re.IGNORECASE,
)
_DATE_ONLY_RE = re.compile(
    r"^(?:(?:\d{1,2}/)?(?:19|20)\d{2})\s*[-–—]\s*(?:(?:\d{1,2}/)?(?:19|20)\d{2}|present)$",
    re.IGNORECASE,
)
_PROJECT_NOISE_RE = re.compile(
    r"\b(strengths?|soft skills?|hobbies?|interests?|objective|summary)\b",
    re.IGNORECASE,
)

# Project name cleaning
_PROJECT_GENERIC_NAMES = {
    "project", "projects", "other project", "other projects",
    "project name", "project details",
}

COMMON_HEADERS = set(load_wordlist("common_headers.txt") or [])


class ProjectDetail(TypedDict):
    """Project detail structure."""
    name: str
    duration: str
    description: str


def clean_project_name(name: str) -> str:
    """
    Clean project name by removing prefixes and description text.
    """
    cleaned = str(name or "").strip()
    if not cleaned:
        return ""
    
    # Remove leading dashes/bullets
    cleaned = re.sub(r"^\s*[-•●▪◦►▸■□◆◇*]\s*", "", cleaned)
    
    # Remove common prefixes
    cleaned = re.sub(r"^\s*(project\s*name|name|other\s+projects?|projects?|project|description)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    
    # Remove action verbs from start
    for verb in ["developed", "built", "created", "implemented", "designed"]:
        if cleaned.lower().startswith(verb):
            cleaned = cleaned[len(verb):].strip()
            cleaned = re.sub(r"^\s*(an|a)\s+", "", cleaned, flags=re.IGNORECASE)
            break
    
    # Remove description text after first period/colon
    for delimiter in [".", ":"]:
        if delimiter in cleaned:
            idx = cleaned.index(delimiter)
            after = cleaned[idx+1:].strip()
            if len(after) > 5 and any(v in after.lower() for v in ("developed", "built", "created")):
                cleaned = cleaned[:idx].strip()
                break
    
    # Collapse spaces and trim
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:|")
    
    # Reject generic names
    lowered = cleaned.lower()
    if not cleaned:
        return ""
    if lowered in _PROJECT_GENERIC_NAMES:
        return ""
    if lowered in {"profile summary", "career objective", "objective", "summary"}:
        return ""
    
    return cleaned


def _clean_project_description(description: str) -> str:
    """Clean project description."""
    cleaned = str(description or "").strip()
    cleaned = re.sub(r"^\s*description\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,-")
    cleaned = re.sub(r"(?:\b(?:like|such as|including)\b\s*)$", "", cleaned, flags=re.IGNORECASE).strip(" ,-")
    return cleaned


def _looks_like_project_title(line: str) -> bool:
    """Check if line looks like a project title."""
    clean = line.strip()
    if len(clean) < 4 or len(clean) > 100:
        return False
    if clean.lower() in COMMON_HEADERS:
        return False
    
    # Handle colon-ending titles
    if clean.endswith(":"):
        without_colon = clean.rstrip(":").strip()
        if len(without_colon.split()) <= 4 and len(without_colon) >= 2:
            lower = without_colon.lower()
            if lower not in {"skills", "languages", "tools", "backend", "frontend", "database", "description", "environment"}:
                if lower not in {"description", "environment", "technologies", "tools used"}:
                    return True
        return False
    
    lower = clean.lower()
    if lower.startswith(("languages:", "backend:", "front-end:", "frontend:", "database:",
                         "authentication:", "tools:", "other:", "version control:",
                         "api testing", "api testing & debugging:", "rest api", "php |",
                         "html", "css", "js", "sql", "mongo db", "node js", "angular js")):
        return False
    if lower.startswith(("developed ", "created ", "implemented ", "built ", "designed ")):
        return False
    if re.fullmatch(r"(?:19|20)\d{2}", clean):
        return False
    if "," in clean and len(clean.split()) <= 4:
        return False
    if re.fullmatch(r"(?:\d{1,2}/)?(?:19|20)\d{2}\s*-\s*(?:\d{1,2}/)?(?:19|20)\d{2}", clean):
        return False
    
    if PROJECT_DATE_RE.search(clean):
        return True
    if "(" in clean and ")" in clean and len(clean.split()) <= 10:
        return True
    if len(clean.split()) <= 6 and not clean.endswith("."):
        if re.search(r"\b(?:objective|summary|profile|address|email|phone)\b", lower):
            return False
        if any(c.isupper() for c in clean):
            return True
    return False


def extract_project_details(text: str | List[str]) -> List[ProjectDetail]:
    """
    Extract structured projects from text.
    
    Args:
        text: Raw text or list of lines
        
    Returns:
        List of project dictionaries
    """
    if isinstance(text, list):
        text = "\n".join(text)
    
    lines = _clean_lines(text)
    projects: List[ProjectDetail] = []
    current = None
    pending_duration = ""
    seen_names = set()
    
    def flush_current():
        nonlocal current
        if not current:
            return
        description = " ".join(current["description_parts"]).strip()
        description = PROJECT_DATE_RE.sub(" ", description)
        description = _clean_project_description(description)
        project_name = clean_project_name(current["name"].strip()) or current["name"].strip()
        projects.append({
            "name": project_name,
            "duration": current["duration"].strip(),
            "description": description,
        })
        current = None
    
    for idx, line in enumerate(lines):
        clean = line.strip()
        if len(clean) < 3 or clean.lower() in COMMON_HEADERS:
            continue
        
        # Handle bullet characters
        bullet_match = re.match(r"^\s*[-•●▪◦►▸■□◆◇*]\s*(.+)$", clean)
        if bullet_match:
            bullet_content = bullet_match.group(1).strip()
            if ":" in bullet_content and not bullet_content.endswith(":"):
                parts = bullet_content.split(":", 1)
                prefix = parts[0].strip().lower()
                potential_desc = parts[1].strip() if len(parts) > 1 else ""
                
                if prefix == "project name":
                    actual_name = potential_desc.rstrip(".").strip()
                    if actual_name and len(actual_name) > 2:
                        name_lower = actual_name.lower()
                        if "intern" not in name_lower and "internship" not in name_lower:
                            if actual_name not in seen_names:
                                flush_current()
                                current = {
                                    "name": actual_name,
                                    "duration": "",
                                    "title_raw": bullet_content,
                                    "description_parts": [],
                                }
                                seen_names.add(actual_name)
                                continue
                
                elif prefix in ("environment", "description", "technologies", "tools"):
                    if current is not None:
                        current["description_parts"].append(potential_desc)
                        continue
                
                else:
                    potential_name = parts[0].strip()
                    if len(potential_name) > 3 and len(potential_desc.split()) >= 5:
                        name_lower = potential_name.lower()
                        desc_lower = potential_desc.lower()
                        if "intern" not in name_lower and "internship" not in name_lower:
                            if "intern" not in desc_lower and "internship" not in desc_lower:
                                if potential_name not in seen_names:
                                    flush_current()
                                    current = {
                                        "name": potential_name,
                                        "duration": "",
                                        "title_raw": bullet_content,
                                        "description_parts": [potential_desc],
                                    }
                                    seen_names.add(potential_name)
                                    continue
            
            if current is not None:
                current["description_parts"].append(bullet_content)
                continue
            
            if _looks_like_project_title(bullet_content):
                flush_current()
                duration_match = PROJECT_DATE_RE.search(bullet_content)
                duration = duration_match.group(0) if duration_match else ""
                if _DATE_ONLY_RE.fullmatch(bullet_content.strip()):
                    pending_duration = duration or bullet_content.strip()
                    continue
                name = bullet_content
                if duration:
                    name = name.replace(duration, " ")
                name = re.sub(r"\([^)]*\)", " ", name)
                name = re.sub(r"\s*[|:-]\s*$", "", name)
                name = re.sub(r"\s{2,}", " ", name).strip(" |-:")
                cleaned_name = clean_project_name(name)
                if cleaned_name and cleaned_name not in seen_names:
                    current = {
                        "name": cleaned_name,
                        "duration": duration or pending_duration,
                        "title_raw": bullet_content,
                        "description_parts": [],
                    }
                    seen_names.add(cleaned_name)
                    pending_duration = ""
                continue
        
        if _looks_like_project_title(clean):
            flush_current()
            duration_match = PROJECT_DATE_RE.search(clean)
            duration = duration_match.group(0) if duration_match else ""
            if _DATE_ONLY_RE.fullmatch(clean.strip()):
                pending_duration = duration or clean.strip()
                continue
            name = clean
            if duration:
                name = name.replace(duration, " ")
            name = re.sub(r"\([^)]*\)", " ", name)
            name = re.sub(r"\s*[|:-]\s*$", "", name)
            name = re.sub(r"\s{2,}", " ", name).strip(" |-:")
            cleaned_name = clean_project_name(name)
            if not cleaned_name or cleaned_name in seen_names:
                pending_duration = duration or pending_duration
                continue
            current = {
                "name": cleaned_name,
                "duration": duration or pending_duration,
                "title_raw": clean,
                "description_parts": [],
            }
            seen_names.add(cleaned_name)
            pending_duration = ""
            continue
        
        if current is None and ":" in clean and not clean.endswith(":"):
            parts = clean.split(":", 1)
            potential_name = parts[0].strip()
            potential_desc = parts[1].strip() if len(parts) > 1 else ""
            if len(potential_name) > 3 and len(potential_desc.split()) >= 5:
                name_lower = potential_name.lower()
                desc_lower = potential_desc.lower()
                if "intern" not in name_lower and "internship" not in name_lower:
                    if "intern" not in desc_lower and "internship" not in desc_lower:
                        if potential_name not in seen_names:
                            flush_current()
                            current = {
                                "name": potential_name,
                                "duration": "",
                                "title_raw": clean,
                                "description_parts": [potential_desc],
                            }
                            seen_names.add(potential_name)
                            continue
        
        if current is None:
            continue
        
        low = clean.lower()
        if _PROJECT_NOISE_RE.search(low):
            continue
        if _DATE_ONLY_RE.fullmatch(clean) and not current["duration"]:
            current["duration"] = clean
            continue
        inline_duration = PROJECT_DATE_RE.search(clean)
        if inline_duration and not current["duration"]:
            current["duration"] = inline_duration.group(0)
            clean = clean.replace(inline_duration.group(0), " ").strip(" ,-")
            if not clean:
                continue
        current["description_parts"].append(clean)
    
    flush_current()
    
    # Filter valid projects
    filtered: List[ProjectDetail] = []
    for item in projects:
        name = clean_project_name(item["name"].strip())
        description = item["description"].strip()
        desc_words = len(description.split())
        desc_low = description.lower()
        
        has_action = any(v in desc_low for v in (
            "developed", "created", "implemented", "built", "designed",
            "building", "made", "developing", "using", "used", "develop", "create",
            "engineered", "constructed", "architected", "launched", "delivered"
        ))
        
        tech_terms = ("html", "css", "javascript", "react", "node", "express", "mongo", "sql",
                      "firebase", "git", "python", "java", "api", "rest", "jwt", "auth",
                      "socket", "cors", "mvc", "npm", "postman", "chrome", "dev tools",
                      "fetch", "weather", "temperature", "real-time", "real time")
        tech_count = sum(1 for t in tech_terms if t in desc_low)
        has_tech_stack = tech_count >= 2
        
        if not name or len(name) <= 3:
            continue
        if "intern" in name.lower() or "internship" in name.lower():
            continue
        if "intern" in desc_low or "internship" in desc_low:
            continue
        if name.lower() in {"profile summary", "career objective", "objective", "summary"}:
            continue
        if name.lower() in {"project", "projects", "other project", "other projects", "project name", "project details"}:
            continue
        if "," in name and len(name.split()) <= 4:
            continue
        if any(tok in name.lower() for tok in ("backend", "front-end", "frontend", "version control", "api testing")):
            continue
        if any(tok in name.upper() for tok in ("PROJECTS", "TECHNICAL", "SKILLS")):
            continue
        if name.isupper() and len(name.split()) <= 3:
            continue
        if len(name) > 80:
            continue
        if desc_words < 5:
            continue
        if not has_action and not has_tech_stack:
            continue
        
        item["name"] = name
        item["description"] = description
        filtered.append(item)
    
    return filtered


def _clean_lines(text: str | List[str]) -> List[str]:
    """Clean and normalize text lines."""
    if isinstance(text, list):
        return [line.strip() for line in text if line.strip()]
    
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > 1000:
            continue
        lines.append(line)
    return lines
