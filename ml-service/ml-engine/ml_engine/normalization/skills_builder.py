"""
Skills normalization builder.

Extracts and normalizes skills from resume sections.
"""

import logging
from typing import Dict, List, Set

from ml_engine.normalization.skill_extractor import extract_skills as extract_skills_from_text
from ml_engine.normalization.skill_extractor import TECH_KEYWORDS

logger = logging.getLogger(__name__)


def build_skills(
    sections: Dict[str, List[str]],
    raw_text: str,
    project_skills: Set[str] = None
) -> List[str]:
    """
    Build normalized skills list.
    
    Args:
        sections: Detected resume sections
        raw_text: Full resume text (for fallback)
        project_skills: Optional skills extracted from project descriptions
        
    Returns:
        List of normalized skills
    """
    logger.info("Starting skills extraction")
    
    skills: List[str] = []
    seen: Set[str] = set()
    
    # Primary: extract from skills section
    if "skills" in sections:
        skills, _ = extract_skills_from_text(sections["skills"])
        seen = {s.lower() for s in skills}
        logger.info(f"Extracted {len(skills)} skills from skills section")
    
    # Fallback 1: try alternate section names
    if not skills:
        skill_section_names = {
            "technical skills", "core competencies", "technical expertise",
            "skills & abilities", "competencies"
        }
        for section_name, content in sections.items():
            if section_name.lower() in skill_section_names:
                skills, _ = extract_skills_from_text(content)
                seen = {s.lower() for s in skills}
                logger.info(f"Extracted {len(skills)} skills from {section_name}")
                break
    
    # Fallback 2: extract from full text
    if not skills:
        skills, _ = extract_skills_from_text(raw_text)
        seen = {s.lower() for s in skills}
        if skills:
            logger.info(f"Extracted {len(skills)} skills from full text (fallback)")
    
    # Fallback 3: direct keyword scan for 2-column resumes
    if not skills:
        skills = _keyword_scan_skills(raw_text)
        seen = {s.lower() for s in skills}
        if skills:
            logger.info(f"Extracted {len(skills)} skills via keyword scan (2-column fallback)")
    
    # Enrich with project skills
    if project_skills:
        for skill in project_skills:
            if skill.lower() not in seen:
                skills.append(skill)
                seen.add(skill.lower())
                logger.debug(f"Added skill from project: {skill}")
    
    logger.info(f"Total skills: {len(skills)}")
    return skills


def _keyword_scan_skills(raw_text: str) -> List[str]:
    """
    Direct keyword scan for skills in raw text.
    
    Used as last resort for 2-column resumes where normal extraction fails.
    """
    raw_lower = raw_text.lower()
    found_skills: List[str] = []
    
    for kw in TECH_KEYWORDS:
        if kw in raw_lower and kw not in [s.lower() for s in found_skills]:
            import re
            if re.search(rf'\b{re.escape(kw)}\b', raw_lower):
                # Capitalize appropriately
                formatted = kw.title() if len(kw) > 3 else kw.upper()
                found_skills.append(formatted)
    
    return found_skills


def extract_project_skills(project_descriptions: List[str]) -> Set[str]:
    """
    Extract skills from project descriptions.
    
    Only extracts exact keyword matches to avoid false positives.
    
    Args:
        project_descriptions: List of project description texts
        
    Returns:
        Set of skill keywords found in projects
    """
    enrichment_keywords = {"deep learning", "computer vision", "machine learning"}
    found: Set[str] = set()
    
    for desc in project_descriptions:
        desc_lower = desc.lower()
        for keyword in enrichment_keywords:
            import re
            if re.search(rf'\b{re.escape(keyword)}\b', desc_lower):
                found.add(keyword)
    
    return found
