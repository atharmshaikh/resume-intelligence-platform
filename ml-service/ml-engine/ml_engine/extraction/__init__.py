"""
Extraction Layer - Resume Information Extraction

This module handles extraction of structured information from resume text:
- Entity extraction (name, email, phone, location)
- Section detection and classification
- Keyword loading and management
- Shared extraction utilities

Architecture:
- entity_extractor.py: Identity information extraction
- section_detector.py: Section boundary detection
- section_classifier.py: Section categorization
- keyword_loader.py: Wordlist management
- extraction_utils.py: Shared utilities
"""

from .entity_extractor import (
    extract_entities,
    extract_entities_detailed,
    extract_name,
    extract_email,
    extract_phone,
    extract_location,
)

from .section_detector import (
    detect_sections,
    detect_sections_with_confidence,
)

from .section_classifier import (
    SectionClassifier,
    classify_section,
    classify_sections,
    SectionClassification,
    get_category_confidence,
    is_confident_classification,
    SECTION_SKILLS,
    SECTION_EDUCATION,
    SECTION_EXPERIENCE,
    SECTION_PROJECTS,
    SECTION_ACHIEVEMENTS,
    SECTION_LANGUAGES,
    SECTION_INTERESTS,
    SECTION_SUMMARY,
    SECTION_UNKNOWN,
)

from .keyword_loader import (
    load_wordlist,
    preload_wordlists,
    get_stats,
    clear_cache,
    initialize,
    load_headers,
    load_locations,
    load_section_keywords,
    load_skills,
    load_programming_languages,
    load_frameworks,
    load_databases,
    load_tools,
    load_tech_terms,
    load_common_languages,
    load_resume_terms,
)

from .extraction_utils import (
    normalize_text,
    clean_line,
    normalize_spacing,
    extract_emails,
    extract_phones,
    extract_urls,
    extract_years,
    is_valid_token,
    filter_tokens,
    tokenize,
    is_noise_line,
    is_contact_line,
    is_header_line,
    validate_email,
    validate_phone,
    validate_name,
    validate_location,
    get_context_window,
    find_in_context,
)

__all__ = [
    # Entity extraction
    "extract_entities",
    "extract_entities_detailed",
    "extract_name",
    "extract_email",
    "extract_phone",
    "extract_location",
    # Section detection
    "detect_sections",
    "detect_sections_with_confidence",
    # Section classification
    "SectionClassifier",
    "classify_section",
    "classify_sections",
    "SectionClassification",
    "get_category_confidence",
    "is_confident_classification",
    # Section categories
    "SECTION_SKILLS",
    "SECTION_EDUCATION",
    "SECTION_EXPERIENCE",
    "SECTION_PROJECTS",
    "SECTION_ACHIEVEMENTS",
    "SECTION_LANGUAGES",
    "SECTION_INTERESTS",
    "SECTION_SUMMARY",
    "SECTION_UNKNOWN",
    # Keyword loading
    "load_wordlist",
    "preload_wordlists",
    "get_stats",
    "clear_cache",
    "initialize",
    "load_headers",
    "load_locations",
    "load_section_keywords",
    "load_skills",
    "load_programming_languages",
    "load_frameworks",
    "load_databases",
    "load_tools",
    "load_tech_terms",
    "load_common_languages",
    "load_resume_terms",
    # Utilities
    "normalize_text",
    "clean_line",
    "normalize_spacing",
    "extract_emails",
    "extract_phones",
    "extract_urls",
    "extract_years",
    "is_valid_token",
    "filter_tokens",
    "tokenize",
    "is_noise_line",
    "is_contact_line",
    "is_header_line",
    "validate_email",
    "validate_phone",
    "validate_name",
    "validate_location",
    "get_context_window",
    "find_in_context",
]
