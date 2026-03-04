import re


EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+"
)

PHONE_PATTERN = re.compile(
    r"\+?\d[\d\s\-]{8,15}\d"
)


def extract_email(text: str):

    match = EMAIL_PATTERN.search(text)

    if match:
        return match.group()

    return None


def extract_phone(text: str):

    match = PHONE_PATTERN.search(text)

    if match:
        phone = match.group()

        phone = phone.replace(" ", "")
        phone = phone.replace("-", "")

        return phone

    return None


def extract_name(text: str):

    lines = text.split("\n")

    for line in lines[:5]:

        line = line.strip()

        if len(line.split()) >= 2 and len(line) < 40:
            return line

    return None


def extract_location(text: str):

    location_keywords = [
        "india",
        "gujarat",
        "delhi",
        "mumbai",
        "bangalore",
    ]

    for line in text.split("\n"):

        line_lower = line.lower()

        for keyword in location_keywords:

            if keyword in line_lower:
                return line.strip()

    return None


def extract_entities(text: str):

    entities = {}

    entities["name"] = extract_name(text)
    entities["email"] = extract_email(text)
    entities["phone"] = extract_phone(text)
    entities["location"] = extract_location(text)

    return entities