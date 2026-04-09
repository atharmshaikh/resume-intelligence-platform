"""
Normalization layer.

Responsible for transforming raw parsed resume sections
into a structured ATS-compatible schema.
"""

from .ats_builder import build_ats_structure, fix_location, recompute_section_flags
from .ats_builder_utils import (
    _clean_lines,
    clean_experience,
    clean_project_name,
)
from .identity_builder import extract_identity
from .skills_builder import build_skills, extract_project_skills
from .skill_extractor import extract_skills
from .education_builder import build_education, get_primary_education
from .experience_builder import build_experience, has_internship
from .project_builder import build_projects, count_projects
from .project_extractor import extract_project_details

__all__ = [
    # Main builder
    "build_ats_structure",
    "fix_location",
    "recompute_section_flags",
    # Utilities
    "_clean_lines",
    "clean_experience",
    "clean_project_name",
    # Individual builders
    "extract_identity",
    "build_skills",
    "extract_project_skills",
    "extract_skills",
    "build_education",
    "get_primary_education",
    "build_experience",
    "has_internship",
    "build_projects",
    "count_projects",
    "extract_project_details",
]
