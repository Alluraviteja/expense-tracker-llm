import spacy
import re
from datetime import datetime

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Simple keyword mapping for categories
CATEGORY_MAP = {
    "groceries": "food",
    "lunch": "food",
    "dinner": "food",
    "restaurant": "food",
    "rent": "living",
    "utilities": "living",
    "electricity": "living",
    "uber": "transport",
    "gas": "transport",
    "train": "transport",
}

# Keywords to detect recurrence
RECURRENCE_KEYWORDS = {
    "daily": "DAILY",
    "every day": "DAILY",
    "weekly": "WEEKLY",
    "every week": "WEEKLY",
    "monthly": "MONTHLY",
    "this month": "MONTHLY",
    "yearly": "YEARLY",
    "annually": "YEARLY",
}

# Extract amount from string using regex
def extract_amount(text):
    match = re.search(r'\$?(\d+(\.\d{1,2})?)', text)
    return float(match.group(1)) if match else 0.0

# Detect recurrence from known keywords
def detect_recurrence(text):
    for keyword, freq in RECURRENCE_KEYWORDS.items():
        if keyword in text.lower():
            return True, freq
    return False, "NONE"

# Map known keywords to categories
def extract_category_and_description(text):
    for keyword, category in CATEGORY_MAP.items():
        if keyword in text.lower():
            return category, keyword
    return "other", "expense"

# Main parser function
def parse_expense(text):
    doc = nlp(text)

    amount = extract_amount(text)
    is_recurring, recurrence_type = detect_recurrence(text)
    category, description = extract_category_and_description(text)
    date = datetime.utcnow().isoformat() + "Z"  # ISO format date

    return {
        "description": description,
        "amount": amount,
        "category": category,
        "date": date,
        "isRecurring": is_recurring,
        "recurrenceType": recurrence_type
    }
