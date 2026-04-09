"""
Project normalization builder.

Extracts and normalizes project information from resume sections.
"""

import logging
from typing import Any, Dict, List, TypedDict

from ml_engine.normalization.project_extractor import extract_project_details, _PROJECT_NOISE_RE

logger = logging.getLogger(__name__)


class ProjectDetail(TypedDict):
    """Project detail structure."""
    name: str
    duration: str
    description: str


def build_projects(sections: Dict[str, List[str]]) -> List[ProjectDetail]:
    """
    Build normalized project list from sections.
    
    Args:
        sections: Detected resume sections
        
    Returns:
        List of project dictionaries with name, duration, description
    """
    logger.info("Starting project extraction")
    
    # Extract project lines from all sections
    project_lines = _extract_project_lines_from_sections(sections)
    logger.info(f"Extracted {len(project_lines)} project-related lines")
    
    if not project_lines:
        logger.info("No projects found")
        return []
    
    # Parse into structured format
    projects = extract_project_details("\n".join(project_lines))
    logger.info(f"Built {len(projects)} structured projects")
    
    return projects


def _extract_project_lines_from_sections(sections_map: Dict[str, List[str]]) -> List[str]:
    """Extract project-related lines from sections."""
    if "projects" in sections_map:
        return [
            ln for ln in sections_map["projects"]
            if not _PROJECT_NOISE_RE.search(ln.lower())
        ]
    
    candidates: List[str] = []
    for sec_name, lines in sections_map.items():
        if sec_name in {"education", "languages", "summary", "interests", "skills"}:
            continue
        if sec_name not in {"experience", "achievements"}:
            continue
        for line in lines:
            ll = line.lower()
            if _PROJECT_NOISE_RE.search(ll):
                continue
            if any(k in ll for k in ("project", "system", "application")):
                candidates.append(line)
                continue
            if any(v in ll for v in ("developed", "implemented", "created", "built", "designed")):
                candidates.append(line)
                continue
    
    return candidates


def count_projects(projects: List[ProjectDetail]) -> int:
    """Count valid projects."""
    return len(projects)


def has_internship_in_projects(projects: List[ProjectDetail]) -> bool:
    """Check if any project is actually an internship."""
    for proj in projects:
        name_lower = proj.get("name", "").lower()
        desc_lower = proj.get("description", "").lower()
        
        if "intern" in name_lower or "internship" in name_lower:
            return True
        if "intern" in desc_lower or "internship" in desc_lower:
            return True
    
    return False
