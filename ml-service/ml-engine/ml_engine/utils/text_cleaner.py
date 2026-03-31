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

ENHANCV_PATTERN = re.compile(
    r"(powered by|enhancv\.com)",
    re.IGNORECASE
)


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
    Merge words split by line breaks ONLY if they are hyphenated.
    Example:
    Inter-
    active -> Interactive
    """
    # merge letter-\nletter pairs
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)

    return text

def normalize_ocr_spacing(text: str) -> str:
    """
    Fix OCR-like spacing such as:
    'H TM L' -> 'HTML'
    'S QL' -> 'SQL'
    'P RO JE CT S' -> 'PROJECTS'
    """

    text = re.sub(
        r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", 
        lambda m: m.group(0).replace(" ", ""), 
        text
    )

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

    # Merge broken words early before other transformations
    text = merge_broken_words(text)

    # Remove PDF artifacts
    text = remove_pdf_artifacts(text)

    # Normalize bullet symbols
    text = normalize_bullets(text)

    # Normalize spaced headers like "S K I L L S"
    text = normalize_spaced_headers(text)

    # Ensure emails and URLs are separated from surrounding text
    text = re.sub(r"\s*@\s*", "@", text)
    text = re.sub(r"(\.(?:com|in|org|net))([A-Za-z])", r"\1 \2", text)

    # Normalize spaced headers
    # Fix split section headers like "PROFESSIONAL\nEXPERIENCE"
    text = re.sub(
        r"\b(professional|work)\s*\n\s*(experience)\b",
        r"\1 \2",
        text,
        flags=re.IGNORECASE
    )

    # Apply OCR fixes only when spacing artifacts exist
    if re.search(r"\b[A-Za-z]\s[A-Za-z]\s[A-Za-z]", text):
        text = normalize_ocr_spacing(text)

    # split merged capital tokens like CPPJavaPython
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)

    # Ensure common resume headers that appear alone on a line
    # get proper newline separation. Only matches when the header
    # is the ENTIRE content of a line (no surrounding text).
    # This does NOT split mid-sentence words like "my skills".
    _HEADER_ISOLATION = re.compile(
        r"(?m)^[ \t]*(EDUCATION|PROJECTS|SKILLS|EXPERIENCE|PROFILE|SUMMARY"
        r"|LANGUAGES|ACHIEVEMENTS|OBJECTIVE|CERTIFICATIONS|DECLARATION)[ \t]*$",
        re.IGNORECASE
    )
    text = _HEADER_ISOLATION.sub(lambda m: "\n" + m.group(0).strip() + "\n", text)


    # Split merged lowercase words (common OCR issue)
    text = re.sub(r"\b([a-z]{3,})([A-Z][a-z]+)\b", r"\1 \2", text)

    # Normalize whitespace
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = MULTILINE_PATTERN.sub("\n", text) 

    # Remove Enhancv artifacts
    text = ENHANCV_PATTERN.sub("", text)

    # Separate common web tokens
    text = re.sub(r"(github\.com)", r" \1 ", text)
    text = re.sub(r"(linkedin\.com)", r" \1 ", text)   

    # Remove resume builder watermarks
    text = re.sub(r"powered by .*", "", text, flags=re.I)

    return text.strip()