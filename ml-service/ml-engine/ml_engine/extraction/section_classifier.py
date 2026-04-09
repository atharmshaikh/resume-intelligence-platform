"""
Section Classifier - Resume Section Categorization

Classifies detected sections into categories:
- skills
- education  
- experience
- projects
- achievements
- languages
- interests
- summary
- unknown

Uses keyword scoring, semantic hints, and alias mapping.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict

from .keyword_loader import load_wordlist
from .extraction_utils import normalize_text, tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section Categories
# ---------------------------------------------------------------------------

SECTION_SKILLS = "skills"
SECTION_EDUCATION = "education"
SECTION_EXPERIENCE = "experience"
SECTION_PROJECTS = "projects"
SECTION_ACHIEVEMENTS = "achievements"
SECTION_LANGUAGES = "languages"
SECTION_INTERESTS = "interests"
SECTION_SUMMARY = "summary"
SECTION_UNKNOWN = "unknown"

# All valid categories
VALID_CATEGORIES = {
    SECTION_SKILLS, SECTION_EDUCATION, SECTION_EXPERIENCE,
    SECTION_PROJECTS, SECTION_ACHIEVEMENTS, SECTION_LANGUAGES,
    SECTION_INTERESTS, SECTION_SUMMARY, SECTION_UNKNOWN,
}

# ---------------------------------------------------------------------------
# Section Name Aliases
# ---------------------------------------------------------------------------

SECTION_ALIASES: Dict[str, str] = {
    # Skills
    "technical skills": SECTION_SKILLS,
    "technical expertise": SECTION_SKILLS,
    "core competencies": SECTION_SKILLS,
    "skill set": SECTION_SKILLS,
    "skills & abilities": SECTION_SKILLS,
    "competencies": SECTION_SKILLS,
    "tech stack": SECTION_SKILLS,
    "technical proficiency": SECTION_SKILLS,
    
    # Education
    "educational background": SECTION_EDUCATION,
    "academic background": SECTION_EDUCATION,
    "education & training": SECTION_EDUCATION,
    "academic qualifications": SECTION_EDUCATION,
    "qualifications": SECTION_EDUCATION,
    
    # Experience
    "work experience": SECTION_EXPERIENCE,
    "professional experience": SECTION_EXPERIENCE,
    "employment history": SECTION_EXPERIENCE,
    "work history": SECTION_EXPERIENCE,
    "professional background": SECTION_EXPERIENCE,
    "career history": SECTION_EXPERIENCE,
    "my work": SECTION_EXPERIENCE,
    
    # Projects
    "academic projects": SECTION_PROJECTS,
    "personal projects": SECTION_PROJECTS,
    "project work": SECTION_PROJECTS,
    "key projects": SECTION_PROJECTS,
    "major projects": SECTION_PROJECTS,
    "my projects": SECTION_PROJECTS,
    "my work": SECTION_PROJECTS,
    
    # Achievements
    "awards": SECTION_ACHIEVEMENTS,
    "certifications": SECTION_ACHIEVEMENTS,
    "certificates": SECTION_ACHIEVEMENTS,
    "honors": SECTION_ACHIEVEMENTS,
    "recognition": SECTION_ACHIEVEMENTS,
    "accomplishments": SECTION_ACHIEVEMENTS,
    "achievements & awards": SECTION_ACHIEVEMENTS,
    
    # Languages
    "language proficiency": SECTION_LANGUAGES,
    "languages known": SECTION_LANGUAGES,
    "linguistic skills": SECTION_LANGUAGES,
    
    # Interests
    "hobbies": SECTION_INTERESTS,
    "activities": SECTION_INTERESTS,
    "extracurricular": SECTION_INTERESTS,
    "co-curricular": SECTION_INTERESTS,
    "personal interests": SECTION_INTERESTS,
    
    # Summary
    "objective": SECTION_SUMMARY,
    "career objective": SECTION_SUMMARY,
    "profile": SECTION_SUMMARY,
    "professional summary": SECTION_SUMMARY,
    "about me": SECTION_SUMMARY,
    "personal profile": SECTION_SUMMARY,
    "summary": SECTION_SUMMARY,
    "career summary": SECTION_SUMMARY,
}

# ---------------------------------------------------------------------------
# Content-Based Classification Keywords
# ---------------------------------------------------------------------------

CONTENT_KEYWORDS: Dict[str, List[str]] = {
    SECTION_SKILLS: [
        "programming", "coding", "scripting", "framework", "library",
        "database", "tool", "technology", "platform", "language",
        "html", "css", "javascript", "python", "java", "sql",
        "react", "angular", "node", "django", "flask",
        "proficient", "familiar", "experienced", "skilled",
    ],
    SECTION_EDUCATION: [
        "university", "college", "institute", "school", "degree",
        "bachelor", "master", "phd", "diploma", "btech", "mtech",
        "cgpa", "gpa", "percentage", "grade", "major", "minor",
        "graduated", "studied", "pursuing", "enrolled",
    ],
    SECTION_EXPERIENCE: [
        "company", "organization", "firm", "corporation", "startup",
        "role", "position", "responsibility", "managed", "led",
        "developed", "implemented", "designed", "created",
        "intern", "internship", "trainee", "engineer", "developer",
        "worked", "working", "employed", "hired",
    ],
    SECTION_PROJECTS: [
        "project", "system", "application", "platform", "tool",
        "developed", "built", "created", "designed", "implemented",
        "using", "technology", "framework", "database",
        "web", "mobile", "app", "software",
    ],
    SECTION_ACHIEVEMENTS: [
        "award", "honor", "recognition", "certificate", "certification",
        "won", "received", "achieved", "ranked", "selected",
        "first", "second", "third", "winner", "runner",
    ],
    SECTION_LANGUAGES: [
        "english", "hindi", "spanish", "french", "german",
        "language", "fluent", "proficient", "native", "intermediate",
        "speaking", "reading", "writing",
    ],
    SECTION_INTERESTS: [
        "hobby", "interest", "activity", "sport", "music",
        "reading", "travel", "photography", "gaming", "volunteer",
        "playing", "watching", "collecting",
    ],
    SECTION_SUMMARY: [
        "objective", "summary", "profile", "about", "seeking",
        "motivated", "passionate", "dedicated", "experienced",
        "professional", "career", "goal", "aspiration",
    ],
}

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SectionClassification:
    """Result of section classification."""
    section_name: str
    category: str
    confidence: float
    method: str  # 'alias', 'keyword', 'heuristic', 'unknown'
    content_length: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Section Classifier
# ---------------------------------------------------------------------------

class SectionClassifier:
    """
    Classifies resume sections into categories.
    
    Uses multiple strategies:
    1. Exact alias matching (highest confidence)
    2. Keyword scoring (medium confidence)
    3. Heuristic rules (lower confidence)
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize classifier.
        
        Args:
            debug: Enable debug logging for classification decisions
        """
        self.debug = debug
        self._alias_cache: Optional[Dict[str, str]] = None
        self._keyword_cache: Optional[Dict[str, List[str]]] = None
        
        logger.debug(f"SectionClassifier initialized (debug={debug})")
    
    def _get_alias_map(self) -> Dict[str, str]:
        """Get alias mapping (lazy loaded)."""
        if self._alias_cache is None:
            self._alias_cache = {
                k.lower(): v for k, v in SECTION_ALIASES.items()
            }
        return self._alias_cache
    
    def _get_keywords(self) -> Dict[str, List[str]]:
        """Get keyword mapping (lazy loaded)."""
        if self._keyword_cache is None:
            self._keyword_cache = CONTENT_KEYWORDS
        return self._keyword_cache
    
    def classify(self, section_name: str, 
                 content: Optional[List[str]] = None) -> SectionClassification:
        """
        Classify a section.
        
        Args:
            section_name: Name of the section
            content: Optional content lines for keyword analysis
            
        Returns:
            SectionClassification object
        """
        logger.debug(f"Classifying section: '{section_name}'")
        
        # Strategy 1: Check aliases first (highest confidence)
        result = self._check_aliases(section_name)
        if result:
            if self.debug:
                logger.debug(f"  → Classified as '{result.category}' via alias (confidence: {result.confidence:.2f})")
            return result
        
        # Strategy 2: Keyword analysis if content provided
        if content:
            result = self._analyze_keywords(section_name, content)
            if result and result.confidence > 0.7:
                if self.debug:
                    logger.debug(f"  → Classified as '{result.category}' via keywords (confidence: {result.confidence:.2f})")
                return result
        
        # Strategy 3: Heuristics
        result = self._apply_heuristics(section_name, content)
        if result:
            if self.debug:
                logger.debug(f"  → Classified as '{result.category}' via heuristics (confidence: {result.confidence:.2f})")
            return result
        
        # Fallback: unknown
        result = SectionClassification(
            section_name=section_name,
            category=SECTION_UNKNOWN,
            confidence=0.5,
            method="unknown"
        )
        
        if self.debug:
            logger.debug(f"  → Classified as '{SECTION_UNKNOWN}' (fallback)")
        
        return result
    
    def _check_aliases(self, section_name: str) -> Optional[SectionClassification]:
        """Check if section name matches known aliases."""
        normalized = section_name.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = normalized.replace(':', '').strip()
        normalized = normalized.replace('-', ' ').strip()
        
        alias_map = self._get_alias_map()
        
        # Exact match
        if normalized in alias_map:
            return SectionClassification(
                section_name=section_name,
                category=alias_map[normalized],
                confidence=0.95,
                method="alias"
            )
        
        # Partial match (contains alias)
        for alias, category in alias_map.items():
            if alias in normalized or normalized in alias:
                return SectionClassification(
                    section_name=section_name,
                    category=category,
                    confidence=0.8,
                    method="alias_partial"
                )
        
        return None
    
    def _analyze_keywords(self, section_name: str, 
                          content: List[str]) -> Optional[SectionClassification]:
        """Analyze content keywords to determine category."""
        content_text = ' '.join(content).lower()
        keywords = self._get_keywords()
        
        scores: Dict[str, int] = {cat: 0 for cat in keywords.keys()}
        
        # Count keyword matches for each category
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                # Use word boundary for accurate matching
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, content_text):
                    scores[category] += 1

        # Find best match
        if not scores or max(scores.values()) == 0:
            return None

        best_category = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_category]
        total_matches = sum(scores.values())

        # Calculate confidence based on score ratio
        confidence = min(0.9, 0.5 + (best_score / max(total_matches, 1)) * 0.4)
        
        return SectionClassification(
            section_name=section_name,
            category=best_category,
            confidence=confidence,
            method="keyword",
            content_length=len(content_text)
        )
    
    def _apply_heuristics(self, section_name: str, 
                          content: Optional[List[str]] = None) -> Optional[SectionClassification]:
        """Apply heuristic rules for classification."""
        normalized = section_name.lower().strip()
        
        # Heuristic 1: Section name contains category keyword
        for category in CONTENT_KEYWORDS.keys():
            if category in normalized:
                return SectionClassification(
                    section_name=section_name,
                    category=category,
                    confidence=0.75,
                    method="heuristic_name"
                )
        
        # Heuristic 2: Content patterns
        if content:
            content_text = ' '.join(content).lower()
            
            # Education: contains years, degree names
            if any(y in content_text for y in ['btech', 'b.tech', 'bachelor', 'master', 'cgpa']):
                return SectionClassification(
                    section_name=section_name,
                    category=SECTION_EDUCATION,
                    confidence=0.7,
                    method="heuristic_content"
                )
            
            # Skills: mostly single words or short phrases
            if all(len(line.split()) <= 4 for line in content[:5] if line.strip()):
                if any(kw in content_text for kw in ['html', 'css', 'javascript', 'python']):
                    return SectionClassification(
                        section_name=section_name,
                        category=SECTION_SKILLS,
                        confidence=0.7,
                        method="heuristic_content"
                    )
            
            # Projects: contains action verbs + tech terms
            action_verbs = {'developed', 'built', 'created', 'designed', 'implemented'}
            tech_terms = {'html', 'css', 'javascript', 'python', 'java', 'sql'}
            
            has_action = any(v in content_text for v in action_verbs)
            has_tech = any(t in content_text for t in tech_terms)
            
            if has_action and has_tech:
                return SectionClassification(
                    section_name=section_name,
                    category=SECTION_PROJECTS,
                    confidence=0.7,
                    method="heuristic_content"
                )
        
        return None
    
    def classify_all(self, sections: Dict[str, List[str]]) -> Dict[str, SectionClassification]:
        """
        Classify all sections.
        
        Args:
            sections: Dictionary of section_name -> content lines
            
        Returns:
            Dictionary of section_name -> SectionClassification
        """
        logger.info(f"Classifying {len(sections)} sections...")
        
        results = {}
        
        for section_name, content in sections.items():
            classification = self.classify(section_name, content)
            results[section_name] = classification
        
        # Log summary
        category_counts: Dict[str, int] = {}
        for result in results.values():
            cat = result.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info(f"Classification summary: {category_counts}")
        
        return results


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def classify_section(section_name: str, 
                     content: Optional[List[str]] = None,
                     debug: bool = False) -> str:
    """
    Convenience function to classify a section.
    
    Args:
        section_name: Name of the section
        content: Optional content lines
        debug: Enable debug logging
        
    Returns:
        Category string
    """
    classifier = SectionClassifier(debug=debug)
    result = classifier.classify(section_name, content)
    return result.category


def classify_sections(sections: Dict[str, List[str]], 
                      debug: bool = False) -> Dict[str, str]:
    """
    Convenience function to classify all sections.
    
    Args:
        sections: Dictionary of section_name -> content lines
        debug: Enable debug logging
        
    Returns:
        Dictionary of section_name -> category
    """
    classifier = SectionClassifier(debug=debug)
    results = classifier.classify_all(sections)
    return {name: result.category for name, result in results.items()}


def get_category_confidence(classification: SectionClassification) -> float:
    """Get confidence score for a classification."""
    return classification.confidence


def is_confident_classification(classification: SectionClassification, 
                                 threshold: float = 0.7) -> bool:
    """Check if classification meets confidence threshold."""
    return classification.confidence >= threshold
