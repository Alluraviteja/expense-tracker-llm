"""Test script for expense parser."""

from app.parser import parse_expense
from pprint import pprint


def test_expense_parsing():
    """Test expense parsing with various examples."""

    test_cases = [
        # Basic expenses with amounts
        "Spent $25.50 on lunch at cafeteria",
        "Paid $150 for monthly train pass",
        "Coffee and snacks $8.75",

        # Expenses with dates
        "Bought groceries for $85.75 on March 15, 2024",
        "Netflix subscription $15.99 charged on 2024-03-20",
        "Paid rent $1200 for April 2024",
        "Yesterday's taxi ride $35",

        # Recurring expenses
        "Monthly gym membership $49.99",
        "Paid electric bill $120 for this month",
        "Weekly meal prep service $89.99",
        "Quarterly insurance payment $450",

        # Health related
        "Doctor's appointment copay $30",
        "Dental cleaning $150 at smile dental",
        "Monthly prescription refill $45.99 at CVS",
        "Physical therapy session $75",

        # Education expenses
        "Bought new textbooks for $200",
        "Online course subscription $29.99 monthly",
        "School supplies from Staples $65.50",
        "Coding bootcamp payment $1500",

        # Entertainment
        "Movie tickets for Avengers $15 per person",
        "Concert tickets $85 for next week",
        "Disney+ annual subscription $79.99",
        "Board game from Amazon $29.99",

        # Shopping with multiple items
        "Weekly grocery shopping at Walmart: fruits $20, vegetables $15, meat $25",
        "Target run: household items $45.99, clothes $89.99",
        "Amazon order: books $35, electronics $129.99",

        # Transportation
        "Uber ride from airport to home $45.50 yesterday",
        "Monthly parking pass $175",
        "Car maintenance and oil change $89.99",
        "Bus fare $2.75",

        # Utilities and bills
        "Internet bill $79.99 for June",
        "Water and sewage $65.50 quarterly",
        "Electricity bill $110.25 due next week",
        "Gas bill $45.75",

        # Subscriptions
        "Annual Amazon Prime subscription renewed for $139",
        "Spotify family plan $14.99 monthly",
        "iCloud storage upgrade $2.99",
        "Gym membership renewal $50",

        # Edge cases
        "Emergency car repair 499.99 dollars",
        "Paid in cash twenty five dollars for lunch",
        "Transfer to savings: 1000",
        "Monthly allowance 300.00",

        # Complex descriptions
        "Split dinner bill with friends at Italian restaurant ($35 my share)",
        "Home office setup: desk $299, chair $199, monitor $249",
        "Birthday party expenses: cake $45, decorations $30, food $150",
        "Weekend trip: hotel $200, food $120, activities $180"
    ]

    print("Testing expense parsing with different examples:\n")

    for text in test_cases:
        print(f"\nInput text: {text}")
        print("-" * 50)
        result = parse_expense(text)
        pprint(result)
        print("=" * 80)


if __name__ == "__main__":
    test_expense_parsing()
