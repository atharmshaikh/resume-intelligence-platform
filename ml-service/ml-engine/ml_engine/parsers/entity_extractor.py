"""
Entity Extractor - Resume Information Extraction

Extracts structured entities from resume text:
- Name
- Email
- Phone
- Location
- Links (LinkedIn, GitHub, Portfolio)

Designed for robustness with Indian resumes.
"""

import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Regex patterns
EMAIL_PATTERN = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
PHONE_PATTERN = re.compile(r'(?:\+?\d[\d\s\-]{8,13}\d)')
URL_PATTERN = re.compile(r'(?:https?://|www\.)\S+')
LINKEDIN_PATTERN = re.compile(r'linkedin\.com/in/[A-Za-z0-9\-_]+', re.IGNORECASE)
GITHUB_PATTERN = re.compile(r'github\.com/[A-Za-z0-9\-_]+', re.IGNORECASE)

# Location patterns (Indian context)
INDIAN_CITIES = {
    'mumbai', 'delhi', 'bangalore', 'bengaluru', 'chennai', 'kolkata',
    'hyderabad', 'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow',
    'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
    'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik',
    'faridabad', 'meerut', 'rajkot', 'varanasi', 'srinagar', 'aurangabad',
    'dhanbad', 'amritsar', 'navi mumbai', 'allahabad', 'ranchi', 'howrah',
    'coimbatore', 'vijayawada', 'jodhpur', 'madurai', 'raipur', 'kota',
    'guwahati', 'chandigarh', 'thiruvananthapuram', 'solapur', 'hubli',
    'mysore', 'tiruchirappalli', 'bareilly', 'aligarh', 'tiruppur',
    'gurgaon', 'moradabad', 'jalandhar', 'bhubaneswar', 'salem',
    'warangal', 'guntur', 'bhiwandi', 'saharanpur', 'gorakhpur', 'bikaner',
    'noida', 'bhilai', 'jamshedpur', 'cuttack', 'kochi', 'dehradun',
}

# Name extraction patterns
NAME_PATTERN = re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')
NAME_AFTER_LABEL = re.compile(r'(?:name|candidate|applicant)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE)


@dataclass
class ExtractedEntities:
    """Container for extracted entities."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def is_complete(self) -> bool:
        """Check if essential entities are present."""
        return bool(self.email or self.phone)


class EntityExtractor:
    """
    Extracts entities from resume text.
    
    Optimized for Indian resumes with robust fallback logic.
    """
    
    def __init__(self):
        self.entities = ExtractedEntities()
    
    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from text.
        
        Args:
            text: Resume text
            
        Returns:
            ExtractedEntities object
        """
        logger.info("Extracting entities from resume...")
        
        lines = text.splitlines()
        
        # Extract from full text
        self.entities.email = self._extract_email(text)
        self.entities.phone = self._extract_phone(text)
        self.entities.linkedin = self._extract_linkedin(text)
        self.entities.github = self._extract_github(text)
        self.entities.portfolio = self._extract_portfolio(text)
        
        # Extract location (try header lines first)
        self.entities.location = self._extract_location(lines[:10])
        if not self.entities.location:
            self.entities.location = self._extract_location(text)
        
        # Extract name (try multiple strategies)
        self.entities.name = self._extract_name(lines[:10])
        if not self.entities.name:
            self.entities.name = self._extract_name_from_content(text)
        
        logger.info(f"Extracted: name={self.entities.name}, email={self.entities.email}, phone={self.entities.phone}")
        
        return self.entities
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address."""
        match = EMAIL_PATTERN.search(text)
        if match:
            email = match.group()
            # Validate common providers
            if any(provider in email.lower() for provider in ['gmail', 'yahoo', 'hotmail', 'outlook', 'rediffmail']):
                return email
            return email
        return None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        matches = PHONE_PATTERN.findall(text)
        
        if matches:
            # Prefer Indian format (+91 or 10 digit)
            for match in matches:
                cleaned = match.replace(' ', '').replace('-', '')
                if cleaned.startswith('+91') and len(cleaned) == 13:
                    return match
                if len(cleaned) == 10:
                    return match
            
            # Return first valid match
            return matches[0]
        
        return None
    
    def _extract_location(self, text: str | List[str]) -> Optional[str]:
        """Extract location (city)."""
        if isinstance(text, list):
            text = ' '.join(text)
        
        text_lower = text.lower()
        
        # Look for Indian cities
        for city in INDIAN_CITIES:
            # Use word boundary to avoid false matches
            pattern = r'\b' + re.escape(city) + r'\b'
            if re.search(pattern, text_lower):
                # Return properly capitalized city name
                return city.title()
        
        return None
    
    def _extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL."""
        match = LINKEDIN_PATTERN.search(text)
        if match:
            return 'https://' + match.group()
        return None
    
    def _extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL."""
        match = GITHUB_PATTERN.search(text)
        if match:
            return 'https://' + match.group()
        return None
    
    def _extract_portfolio(self, text: str) -> Optional[str]:
        """Extract portfolio/personal website URL."""
        matches = URL_PATTERN.findall(text)
        
        for url in matches:
            # Exclude LinkedIn and GitHub
            if 'linkedin' in url.lower() or 'github' in url.lower():
                continue
            
            # Look for portfolio indicators
            if any(kw in url.lower() for kw in ['portfolio', 'personal', 'website', 'blog']):
                if not url.startswith('http'):
                    return 'https://' + url
                return url
        
        return None
    
    def _extract_name(self, lines: List[str]) -> Optional[str]:
        """Extract name from header lines."""
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Skip lines with contact info
            if EMAIL_PATTERN.search(stripped) or PHONE_PATTERN.search(stripped):
                continue
            
            # Try name pattern (First Last format)
            match = NAME_PATTERN.match(stripped)
            if match:
                name = match.group(1)
                # Validate: 2-4 words, no numbers
                words = name.split()
                if 2 <= len(words) <= 4 and not any(w.isdigit() for w in words):
                    return name
            
            # Try labeled name
            match = NAME_AFTER_LABEL.search(stripped)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_name_from_content(self, text: str) -> Optional[str]:
        """Extract name from content using heuristics."""
        # Look for "I am", "I'm" patterns
        patterns = [
            r"(?:I am|I'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"(?:My name is|This is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"(?:candidate|applicant)\s*(?:name)?\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted entities to dictionary."""
        return self.entities.to_dict()


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Convenience function to extract entities.
    
    Args:
        text: Resume text
        
    Returns:
        Dictionary of entities
    """
    extractor = EntityExtractor()
    entities = extractor.extract(text)
    return entities.to_dict()
