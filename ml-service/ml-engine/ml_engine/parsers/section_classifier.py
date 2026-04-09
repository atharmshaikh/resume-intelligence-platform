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

Uses keyword matching, semantic hints, and scoring system.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Section categories
SECTION_SKILLS = "skills"
SECTION_EDUCATION = "education"
SECTION_EXPERIENCE = "experience"
SECTION_PROJECTS = "projects"
SECTION_ACHIEVEMENTS = "achievements"
SECTION_LANGUAGES = "languages"
SECTION_INTERESTS = "interests"
SECTION_SUMMARY = "summary"
SECTION_UNKNOWN = "unknown"

# Section name aliases
SECTION_ALIASES: Dict[str, str] = {
    # Skills
    "technical skills": SECTION_SKILLS,
    "technical expertise": SECTION_SKILLS,
    "core competencies": SECTION_SKILLS,
    "skill set": SECTION_SKILLS,
    "skills & abilities": SECTION_SKILLS,
    "competencies": SECTION_SKILLS,
    
    # Education
    "educational background": SECTION_EDUCATION,
    "academic background": SECTION_EDUCATION,
    "education & training": SECTION_EDUCATION,
    "academic qualifications": SECTION_EDUCATION,
    
    # Experience
    "work experience": SECTION_EXPERIENCE,
    "professional experience": SECTION_EXPERIENCE,
    "employment history": SECTION_EXPERIENCE,
    "work history": SECTION_EXPERIENCE,
    "professional background": SECTION_EXPERIENCE,
    "career history": SECTION_EXPERIENCE,
    
    # Projects
    "academic projects": SECTION_PROJECTS,
    "personal projects": SECTION_PROJECTS,
    "project work": SECTION_PROJECTS,
    "key projects": SECTION_PROJECTS,
    "major projects": SECTION_PROJECTS,
    
    # Achievements
    "awards": SECTION_ACHIEVEMENTS,
    "certifications": SECTION_ACHIEVEMENTS,
    "certificates": SECTION_ACHIEVEMENTS,
    "honors": SECTION_ACHIEVEMENTS,
    "recognition": SECTION_ACHIEVEMENTS,
    "accomplishments": SECTION_ACHIEVEMENTS,
    
    # Languages
    "language proficiency": SECTION_LANGUAGES,
    "languages known": SECTION_LANGUAGES,
    
    # Interests
    "hobbies": SECTION_INTERESTS,
    "activities": SECTION_INTERESTS,
    "extracurricular": SECTION_INTERESTS,
    "co-curricular": SECTION_INTERESTS,
    
    # Summary
    "objective": SECTION_SUMMARY,
    "career objective": SECTION_SUMMARY,
    "profile": SECTION_SUMMARY,
    "professional summary": SECTION_SUMMARY,
    "about me": SECTION_SUMMARY,
    "personal profile": SECTION_SUMMARY,
}

# Content-based classification keywords
CONTENT_KEYWORDS: Dict[str, List[str]] = {
    SECTION_SKILLS: [
        "programming", "coding", "scripting", "framework", "library",
        "database", "tool", "technology", "platform", "language",
        "html", "css", "javascript", "python", "java", "sql",
        "react", "angular", "node", "django", "flask",
    ],
    SECTION_EDUCATION: [
        "university", "college", "institute", "school", "degree",
        "bachelor", "master", "phd", "diploma", "btech", "mtech",
        "cgpa", "gpa", "percentage", "grade", "major", "minor",
    ],
    SECTION_EXPERIENCE: [
        "company", "organization", "firm", "corporation", "startup",
        "role", "position", "responsibility", "managed", "led",
        "developed", "implemented", "designed", "created",
        "intern", "internship", "trainee", "engineer", "developer",
    ],
    SECTION_PROJECTS: [
        "project", "system", "application", "platform", "tool",
        "developed", "built", "created", "designed", "implemented",
        "using", "technology", "framework", "database",
    ],
    SECTION_ACHIEVEMENTS: [
        "award", "honor", "recognition", "certificate", "certification",
        "won", "received", "achieved", "ranked", "selected",
    ],
    SECTION_LANGUAGES: [
        "english", "hindi", "spanish", "french", "german",
        "language", "fluent", "proficient", "native", "intermediate",
    ],
    SECTION_INTERESTS: [
        "hobby", "interest", "activity", "sport", "music",
        "reading", "travel", "photography", "gaming", "volunteer",
    ],
    SECTION_SUMMARY: [
        "objective", "summary", "profile", "about", "seeking",
        "motivated", "passionate", "dedicated", "experienced",
    ],
}


@dataclass
class SectionClassification:
    """Result of section classification."""
    section_name: str
    category: str
    confidence: float
    method: str  # 'alias', 'keyword', 'heuristic', 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_name": self.section_name,
            "category": self.category,
            "confidence": self.confidence,
            "method": self.method,
        }


class SectionClassifier:
    """
    Classifies resume sections into categories.
    
    Uses multiple strategies:
    1. Exact alias matching
    2. Keyword scoring
    3. Heuristic rules
    """
    
    def __init__(self):
        self.classifications: List[SectionClassification] = []
    
    def classify(self, section_name: str, content: List[str] | None = None) -> SectionClassification:
        """
        Classify a section.
        
        Args:
            section_name: Name of the section
            content: Optional content lines for keyword analysis
            
        Returns:
            SectionClassification object
        """
        # Strategy 1: Check aliases first (highest confidence)
        result = self._check_aliases(section_name)
        if result:
            return result
        
        # Strategy 2: Keyword analysis if content provided
        if content:
            result = self._analyze_keywords(section_name, content)
            if result and result.confidence > 0.7:
                return result
        
        # Strategy 3: Heuristics
        result = self._apply_heuristics(section_name, content)
        if result:
            return result
        
        # Fallback: unknown
        return SectionClassification(
            section_name=section_name,
            category=SECTION_UNKNOWN,
            confidence=0.5,
            method="unknown"
        )
    
    def _check_aliases(self, section_name: str) -> Optional[SectionClassification]:
        """Check if section name matches known aliases."""
        normalized = section_name.lower().strip()
        
        # Remove common prefixes/suffixes
        normalized = normalized.replace(':', '').strip()
        normalized = normalized.replace('-', ' ').strip()
        
        if normalized in SECTION_ALIASES:
            return SectionClassification(
                section_name=section_name,
                category=SECTION_ALIASES[normalized],
                confidence=0.95,
                method="alias"
            )
        
        # Partial match
        for alias, category in SECTION_ALIASES.items():
            if alias in normalized or normalized in alias:
                return SectionClassification(
                    section_name=section_name,
                    category=category,
                    confidence=0.8,
                    method="alias_partial"
                )
        
        return None
    
    def _analyze_keywords(self, section_name: str, content: List[str]) -> Optional[SectionClassification]:
        """Analyze content keywords to determine category."""
        content_text = ' '.join(content or []).lower()
        
        scores: Dict[str, int] = {cat: 0 for cat in CONTENT_KEYWORDS.keys()}
        
        # Count keyword matches for each category
        for category, keywords in CONTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_text:
                    scores[category] += 1
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            return None
        
        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]
        total_matches = sum(scores.values())
        
        # Calculate confidence based on score ratio
        confidence = min(0.9, 0.5 + (best_score / max(total_matches, 1)) * 0.4)
        
        return SectionClassification(
            section_name=section_name,
            category=best_category,
            confidence=confidence,
            method="keyword"
        )
    
    def _apply_heuristics(self, section_name: str, content: List[str] | None = None) -> Optional[SectionClassification]:
        """Apply heuristic rules for classification."""
        normalized = section_name.lower().strip()
        
        # Heuristic: Section name contains category keyword
        for category in CONTENT_KEYWORDS.keys():
            if category in normalized:
                return SectionClassification(
                    section_name=section_name,
                    category=category,
                    confidence=0.75,
                    method="heuristic_name"
                )
        
        # Heuristic: Content patterns
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
        
        return None
    
    def classify_all(self, sections: Dict[str, List[str]]) -> Dict[str, SectionClassification]:
        """
        Classify all sections.
        
        Args:
            sections: Dictionary of section_name -> content lines
            
        Returns:
            Dictionary of section_name -> SectionClassification
        """
        results = {}
        
        for section_name, content in sections.items():
            classification = self.classify(section_name, content)
            results[section_name] = classification
            self.classifications.append(classification)
            
            logger.debug(f"Classified '{section_name}' → {classification.category} ({classification.method}, {classification.confidence:.2f})")
        
        return results


def classify_section(section_name: str, content: List[str] | None = None) -> str:
    """
    Convenience function to classify a section.
    
    Args:
        section_name: Name of the section
        content: Optional content lines
        
    Returns:
        Category string
    """
    classifier = SectionClassifier()
    result = classifier.classify(section_name, content)
    return result.category


def classify_sections(sections: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Convenience function to classify all sections.
    
    Args:
        sections: Dictionary of section_name -> content lines
        
    Returns:
        Dictionary of section_name -> category
    """
    classifier = SectionClassifier()
    results = classifier.classify_all(sections)
    return {name: result.category for name, result in results.items()}
