"""Expense parser using custom transformer model."""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
from app.model import ExpenseModel
from app.dataset import ExpenseDataset


class ExpenseParser:
    """Parser class for expense tracking."""

    def __init__(self, model_dir: str = "models"):
        self.dataset = ExpenseDataset()

        # Try to load trained model, or create new one
        model_path = Path(model_dir)
        if model_path.exists() and (model_path / "model.pt").exists():
            # Load the saved state
            checkpoint = torch.load(model_path / "model.pt")

            # Create model instance
            self.model = ExpenseModel(
                model_name=checkpoint['config']['model_name'],
                num_categories=checkpoint['config']['num_categories']
            )

            # Load state dict and tokenizer
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.tokenizer = checkpoint['tokenizer']
        else:
            print("Warning: No trained model found. Using untrained model.")
            self.model = ExpenseModel(
                num_categories=len(self.dataset.CATEGORIES))

    def parse_expense(self, text: str) -> Dict[str, Any]:
        """Parse expense text and return structured data."""
        # Get predictions from model
        prediction = self.model.predict(text, self.dataset.CATEGORIES)

        # Add expense to dataset
        self.dataset.add_expense(
            text=text,
            amount=prediction["amount"],
            category=prediction["category"]["label"],
            date=prediction["date"],
            is_recurring=prediction["is_recurring"]
        )

        return prediction

    def get_stats(self) -> Dict[str, Any]:
        """Get expense statistics."""
        return self.dataset.get_expense_stats()


def parse_expense(text: str) -> Dict[str, Any]:
    """Convenience function to parse a single expense."""
    parser = ExpenseParser()
    return parser.parse_expense(text)
