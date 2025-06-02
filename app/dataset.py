"""Dataset handling for expense tracking."""

from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from dateutil import parser


class ExpenseDataset:
    """Dataset class for expense tracking."""

    CATEGORIES = [
        "food", "transport", "shopping", "entertainment",
        "utilities", "health", "education", "other"
    ]

    def __init__(self):
        self.expenses: List[Dict[str, Any]] = []
        self.data_file = Path("data/expenses.json")
        self._load_data()

    def _load_data(self):
        """Load existing data if available."""
        if self.data_file.exists():
            with open(self.data_file, "r") as f:
                self.expenses = json.load(f)

    def save_data(self):
        """Save data to file."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, "w") as f:
            json.dump(self.expenses, f, indent=2)

    def add_expense(self, text: str, amount: float, category: str,
                    date: str = None, is_recurring: bool = False) -> Dict[str, Any]:
        """Add a new expense entry."""
        if category not in self.CATEGORIES:
            category = "other"

        if date is None:
            date = datetime.now().isoformat()
        else:
            try:
                date = parser.parse(date).isoformat()
            except:
                date = datetime.now().isoformat()

        expense = {
            "text": text,
            "amount": amount,
            "category": category,
            "date": date,
            "is_recurring": is_recurring,
            "created_at": datetime.now().isoformat()
        }

        self.expenses.append(expense)
        self.save_data()
        return expense

    def get_training_data(self) -> Tuple[List[str], List[Dict[str, float]], List[Dict[str, Any]]]:
        """Get data in format suitable for training."""
        texts = []
        category_labels = []
        metadata = []

        for expense in self.expenses:
            texts.append(expense["text"])

            # One-hot encode categories
            cats = {cat: 1.0 if cat == expense["category"] else 0.0
                    for cat in self.CATEGORIES}
            category_labels.append(cats)

            meta = {
                "amount": expense["amount"],
                "date": expense["date"],
                "is_recurring": expense["is_recurring"]
            }
            metadata.append(meta)

        return texts, category_labels, metadata

    def get_expense_stats(self) -> Dict[str, Any]:
        """Get statistical information about expenses."""
        if not self.expenses:
            return {"total": 0, "by_category": {}, "recurring_total": 0}

        df = pd.DataFrame(self.expenses)
        stats = {
            "total": df["amount"].sum(),
            "by_category": df.groupby("category")["amount"].sum().to_dict(),
            "recurring_total": df[df["is_recurring"]]["amount"].sum(),
            "average": df["amount"].mean(),
            "max": df["amount"].max(),
            "min": df["amount"].min()
        }
        return stats


# Example training data
SAMPLE_EXPENSES = [
    # Food expenses
    {
        "text": "Lunch at cafeteria",
        "amount": 12.50,
        "category": "food",
        "is_recurring": False
    },
    {
        "text": "Grocery shopping at Walmart",
        "amount": 85.75,
        "category": "food",
        "is_recurring": False
    },
    {
        "text": "Ordered pizza for dinner",
        "amount": 25.99,
        "category": "food",
        "is_recurring": False
    },
    {
        "text": "Weekly meal prep ingredients",
        "amount": 120.00,
        "category": "food",
        "is_recurring": True
    },

    # Transport expenses
    {
        "text": "Monthly train pass",
        "amount": 150.00,
        "category": "transport",
        "is_recurring": True
    },
    {
        "text": "Uber ride from airport",
        "amount": 45.50,
        "category": "transport",
        "is_recurring": False
    },
    {
        "text": "Bus ticket to downtown",
        "amount": 2.75,
        "category": "transport",
        "is_recurring": False
    },
    {
        "text": "Car maintenance and oil change",
        "amount": 89.99,
        "category": "transport",
        "is_recurring": False
    },

    # Entertainment expenses
    {
        "text": "Netflix subscription",
        "amount": 15.99,
        "category": "entertainment",
        "is_recurring": True
    },
    {
        "text": "Movie tickets for Avengers",
        "amount": 30.00,
        "category": "entertainment",
        "is_recurring": False
    },
    {
        "text": "Spotify Premium monthly",
        "amount": 9.99,
        "category": "entertainment",
        "is_recurring": True
    },
    {
        "text": "Concert tickets for next month",
        "amount": 75.00,
        "category": "entertainment",
        "is_recurring": False
    },

    # Utilities expenses
    {
        "text": "Electric bill payment",
        "amount": 120.00,
        "category": "utilities",
        "is_recurring": True
    },
    {
        "text": "Water and sewage monthly bill",
        "amount": 45.00,
        "category": "utilities",
        "is_recurring": True
    },
    {
        "text": "Internet service provider monthly",
        "amount": 79.99,
        "category": "utilities",
        "is_recurring": True
    },
    {
        "text": "Gas bill for heating",
        "amount": 65.00,
        "category": "utilities",
        "is_recurring": True
    },

    # Health expenses
    {
        "text": "Doctor's appointment copay",
        "amount": 50.00,
        "category": "health",
        "is_recurring": False
    },
    {
        "text": "Monthly prescription refill",
        "amount": 25.00,
        "category": "health",
        "is_recurring": True
    },
    {
        "text": "Dental cleaning and checkup",
        "amount": 150.00,
        "category": "health",
        "is_recurring": False
    },
    {
        "text": "Gym membership monthly fee",
        "amount": 49.99,
        "category": "health",
        "is_recurring": True
    },

    # Education expenses
    {
        "text": "Textbooks for new semester",
        "amount": 200.00,
        "category": "education",
        "is_recurring": False
    },
    {
        "text": "Online course subscription",
        "amount": 29.99,
        "category": "education",
        "is_recurring": True
    },
    {
        "text": "School supplies and stationery",
        "amount": 45.00,
        "category": "education",
        "is_recurring": False
    },
    {
        "text": "Language learning app premium",
        "amount": 19.99,
        "category": "education",
        "is_recurring": True
    },

    # Shopping expenses
    {
        "text": "New winter jacket",
        "amount": 89.99,
        "category": "shopping",
        "is_recurring": False
    },
    {
        "text": "Amazon Prime yearly subscription",
        "amount": 139.00,
        "category": "shopping",
        "is_recurring": True
    },
    {
        "text": "Home decor items",
        "amount": 65.00,
        "category": "shopping",
        "is_recurring": False
    },
    {
        "text": "Monthly beauty box subscription",
        "amount": 25.00,
        "category": "shopping",
        "is_recurring": True
    },

    # Other expenses
    {
        "text": "Birthday gift for friend",
        "amount": 50.00,
        "category": "other",
        "is_recurring": False
    },
    {
        "text": "Charitable donation monthly",
        "amount": 20.00,
        "category": "other",
        "is_recurring": True
    },
    {
        "text": "Pet supplies and food",
        "amount": 55.00,
        "category": "other",
        "is_recurring": False
    },
    {
        "text": "Monthly cloud storage subscription",
        "amount": 9.99,
        "category": "other",
        "is_recurring": True
    }
]


def initialize_sample_data():
    """Initialize dataset with sample expenses."""
    dataset = ExpenseDataset()
    if not dataset.expenses:
        for expense in SAMPLE_EXPENSES:
            dataset.add_expense(
                text=expense["text"],
                amount=expense["amount"],
                category=expense["category"],
                is_recurring=expense["is_recurring"]
            )
    return dataset
