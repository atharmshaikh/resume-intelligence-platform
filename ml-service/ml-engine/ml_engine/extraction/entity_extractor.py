import re
from .keyword_loader import load_wordlist
COMMON_HEADERS = load_wordlist("common_headers.txt")
LOCATION_KEYWORDS = load_wordlist("locations.txt")

EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3,4}\)?[\s\-]?)?\d{6,10}"
)


def extract_email(text: str):
    """Extract email address from resume text."""

    match = EMAIL_PATTERN.search(text)

    if match:

        email = match.group().lower()
        
        email = email.strip(".,;:")
        # trim accidental trailing characters
        email = re.sub(r"(com|in|org)[a-z]+$", r"\1", email)

        return email

    return None


def extract_phone(text: str):
    """
    Extract first valid phone number.
    """

    for match in PHONE_PATTERN.finditer(text):

        phone = match.group()

        phone = re.sub(r"[^\d+]", "", phone)

        digits = phone.replace("+", "")

        # valid international range
        if 10 <= len(digits) <= 13:
            return phone

    return None


def extract_name(text: str):

    lines = text.splitlines()

    for line in lines[:6]:

        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower in COMMON_HEADERS:
            continue

        if (
            2 <= len(line.split()) <= 4 
            and len(line) < 40
            and not any(char.isdigit() for char in line)
            and re.fullmatch(r"[A-Za-z .'-]+", line)
        ):
            return line.title()

        # reject common role phrases
        ROLE_WORDS = {"developer", "engineer", "designer", "development", "profile"}

        if any(word in lower for word in ROLE_WORDS):
            continue

        return line.title()
    
    return None


def extract_location(text: str):

    for line in text.split("\n")[:20]:

        line_lower = line.lower()

        for keyword in LOCATION_KEYWORDS:

            if re.search(rf"\b{re.escape(keyword)}\b", line_lower):
                
                clean = line

                # remove email fragments
                clean = re.sub(r"\S+@\S+", "", clean)

                # remove phone numbers
                clean = re.sub(r"\d{8,}", "", clean)

                clean = clean.strip()

                if len(clean) > 60:
                    return None

                return clean    
    return None


def extract_entities(text: str):

    entities: dict[str, str | None] = {
    "name": None,
    "email": None,
    "phone": None,
    "location": None,
    }

    entities["name"] = extract_name(text)
    entities["email"] = extract_email(text)
    entities["phone"] = extract_phone(text)
    entities["location"] = extract_location(text)

    return entities