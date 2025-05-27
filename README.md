# AI Expense Tracker

An intelligent expense tracking system that uses Natural Language Processing (NLP) to parse and categorize expenses from text input.

## Features

- Parse expense descriptions using SpaCy NLP
- Extract amounts using regex
- Categorize expenses automatically
- Detect recurring expenses
- ISO format date tracking

## Setup

1. Install the required dependencies:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

2. The main functionality is in `app/parser.py` which provides expense parsing capabilities.

## Usage

The parser can process natural language expense descriptions and extract structured data including:
- Amount
- Category
- Description
- Date
- Recurrence information

Example:
```python
from app.parser import parse_expense

result = parse_expense("Spent $50 on groceries")
print(result)
```

## Categories

Currently supported expense categories:
- Food (groceries, restaurants, etc.)
- Living (rent, utilities)
- Transport (uber, gas, train)
- Other (default category)
