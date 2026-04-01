from ml_engine.utils.text_cleaner import clean_text
from ml_engine.extraction.section_detector import detect_sections
from ml_engine.extraction.entity_extractor import extract_entities
from ml_engine.features.feature_extractor import extract_features
# Create a dummy schema to avoid full import issues if needed, but let's try importing properly
from ml_engine.schemas.resume_schema import ResumeSchema
import json

raw_extracted_text = """AYAN RANA
ayanmrana@gmail.com
+91 7016641560
Anand, Gujarat, India
 linkedin.com /in/ayan-rana
PROFILE
Enthusiastic engineering professional with a strong interest in app development.
EDUCATION
Bachelor of Technology (B.Tech) in Information Technology Madhuben & Bhanubhai Patel Institute of Technology CGPA: 8.94
Diploma in Computer Engineering Shri K.J. Polytechnic CGPA: 9.44
SKILLS
C
CPP
Java
Python
My SQL
PHP
HTML
CSS
PROFESSIONAL EXPERIENCE
Cyber Security Intern Ultron Technologies
PROJECTS
Number Plate Detection System
Vehicle Rental System
ACHIEVEMENTS
Code Unnati Innovation Marathon 2025
LANGUAGES
English
Gujarati
Hindi"""

cleaned = clean_text(raw_extracted_text)
print("--- CLEANED TEXT ---")
print(cleaned)
print("--------------------")

sections = detect_sections(cleaned)
print("--- DETECTED SECTIONS ---")
print(json.dumps(sections, indent=2))
print("-------------------------")

entities = extract_entities(cleaned)
print("--- ENTITIES ---")
print(json.dumps(entities, indent=2))

resume = ResumeSchema(
    name=entities.get("name"),
    email=entities.get("email"),
    phone=entities.get("phone"),
    location=entities.get("location"),
    raw_text=cleaned,
    sections=sections,
    skills=sections.get("skills", []),
    education=sections.get("education", []),
    experience=sections.get("experience", []),
    projects=sections.get("projects", []),
)

features = extract_features(resume)
print("--- FEATURES ---")
# limit output
print(json.dumps({k: v for k, v in features.items() if v != 0}, indent=2))
