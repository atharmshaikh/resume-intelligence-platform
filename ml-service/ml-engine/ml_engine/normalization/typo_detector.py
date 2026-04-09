"""
typo_detector.py
================
Detects professional typos in resume contact information and skill sets.
"""
import re
from typing import List

# Common email domain typos
EMAIL_TYPOS = {
    "gamil.com": "gmail.com",
    "gmial.com": "gmail.com",
    "gmal.com": "gmail.com",
    "gmai.com": "gmail.com",
    "hotmal.com": "hotmail.com",
    "yaho.com": "yahoo.com",
    "outlok.com": "outlook.com",
    "protonmal.com": "protonmail.com",
    "gamil.co.in": "gmail.co.in"
}

def check_contact_typos(email: str, phone: str) -> List[str]:
    """
    Returns a list of identified typos in contact data.
    """
    typos = []
    
    # Check Email
    if email:
        domain = email.split("@")[-1].lower() if "@" in email else ""
        if domain in EMAIL_TYPOS:
            typos.append(f"Email domain typo: '{domain}' instead of '{EMAIL_TYPOS[domain]}'")
            
    # Check Phone
    if phone:
        clean_phone = re.sub(r"\D", "", phone)
        # Standard Indian phone is 10 digits
        if len(clean_phone) > 0 and len(clean_phone) < 10:
             typos.append(f"Incomplete phone number: {phone}")
        elif len(clean_phone) > 13:
             typos.append(f"Excessive digits in phone: {phone}")
             
    return typos
