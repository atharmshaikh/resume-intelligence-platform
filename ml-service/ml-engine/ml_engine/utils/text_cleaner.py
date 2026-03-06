"""
Text cleaning utilities.
Keeps text normalization consistent across the engine.
"""

import re

# Normalize weird unicode bullets
BULLET_PATTERN = re.compile(r"[•●▪◦►▸■□◆◇]")

# Remove PDF CID artifacts like (cid:123)
CID_PATTERN = re.compile(r"\(cid:\d+\)")

# Remove icon unicode range (FontAwesome etc)
ICON_PATTERN = re.compile(r"[\uf000-\uf8ff]")

# Collapse multi spaces
MULTISPACE_PATTERN = re.compile(r"[ \t]+")

# Collapse multi blank lines
MULTILINE_PATTERN = re.compile(r"\n\s*\n")

def normalize_spaced_headers(text: str) -> str:
    """
    Convert headers like 'T E C H S K I L L S'
    into 'TECHSKILLS'.
    """

    lines = text.split("\n")
    normalized = []

    for line in lines:

        clean = line.strip()

        if re.match(r"^([A-Z]\s){3,}[A-Z]$", clean):
            clean = clean.replace(" ", "")

        normalized.append(clean)

    return "\n".join(normalized)

def normalize_bullets(text: str) -> str:
    """
    Replace different bullet symbols with '-'
    """

    return BULLET_PATTERN.sub("-", text)

def remove_pdf_artifacts(text: str) -> str:
    """
    Remove PDF parsing artifacts like (cid:123)
    and icon unicode ranges.
    """

    text = CID_PATTERN.sub("", text)
    text = ICON_PATTERN.sub("", text)

    return text

def merge_broken_words(text: str) -> str:
    """
    Merge words split by line breaks.
    Example:
    Java
    Script -> JavaScript
    """

    return re.sub(r"([a-z])\n([a-z])", r"\1\2", text)

def normalize_ocr_spacing(text: str) -> str:
    """
    Fix OCR-like spacing such as:
    'H TM L' -> 'HTML'
    'S QL' -> 'SQL'
    'P RO JE CT S' -> 'PROJECTS'
    """

    rtext = re.sub(r"\b(?:[A-Z]\s){2,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), text)

    # Normalize common skill spacing
    text = re.sub(r"\bMy\s+SQL\b", "MySQL", text, flags=re.I)
    text = re.sub(r"\bNode\s+JS\b", "NodeJS", text, flags=re.I)
    text = re.sub(r"\bPower\s+BI\b", "PowerBI", text, flags=re.I)

    return text

def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove redundant characters.
    """
    
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r", "\n")

    # Remove PDF artifacts
    text = remove_pdf_artifacts(text)

    # Normalize bullet symbols
    text = normalize_bullets(text)

    # Ensure emails and URLs are separated from surrounding text
    text = re.sub(r"\s*@\s*", "@", text)
    text = re.sub(r"(\.(?:com|in|org|net))([A-Za-z])", r"\1 \2", text)

    # Normalize spaced headers
    text = normalize_spaced_headers(text)
    text = normalize_ocr_spacing(text)

    # Separate punctuation from words to prevent merging
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  

    # Ensure section headers start on new lines
    text = re.sub(
        r"(EDUCATION|PROJECTS|SKILLS|CERTIFICATIONS|EXPERIENCE|ACTIVITIES|PROFILE|SUMMARY|LANGUAGES|INTERESTS|OBJECTIVE)",
        r"\n\1",
        text,
        flags=re.IGNORECASE
    )

    # Split merged lowercase words (common OCR issue)
    text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)

    # Normalize whitespace
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = MULTILINE_PATTERN.sub("\n", text) 

    # Separate common web tokens
    text = re.sub(r"(github\.com)", r" \1 ", text)
    text = re.sub(r"(linkedin\.com)", r" \1 ", text)   

    # Remove resume builder watermarks
    text = re.sub(r"powered by .*", "", text, flags=re.I)

    text = merge_broken_words(text)

    return text.strip()