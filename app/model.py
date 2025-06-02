import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
from dateutil import parser
import json
from pathlib import Path
from word2number import w2n


class ExpenseModel(nn.Module):
    """Custom model for expense tracking using transformers."""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_categories: int = 8, max_amount: float = 1000.0):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_amount = max_amount
        hidden_size = self.transformer.config.hidden_size

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Attention for category classification
        self.category_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Category head
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_categories)
        )

        # Amount head with proper scaling
        self.amount_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

        # Recurring flag head
        self.recurring_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply layer normalization
        sequence_output = self.layer_norm(sequence_output)

        # Attention pooling
        attention_weights = self.category_attention(
            sequence_output).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_output = torch.bmm(
            attention_weights.unsqueeze(1), sequence_output).squeeze(1)

        # Heads
        # Will use CrossEntropyLoss (no activation needed)
        category_logits = self.category_classifier(attended_output)
        # Already has sigmoid for [0,1] output
        amount_pred = self.amount_extractor(attended_output)
        recurring_pred = self.recurring_classifier(
            attended_output)  # Already has sigmoid

        return category_logits, amount_pred, recurring_pred

    def normalize_amount(self, amount: float) -> float:
        """Normalize amount to [0,1] range."""
        return min(amount / self.max_amount, 1.0)

    def denormalize_amount(self, value: float) -> float:
        """Convert normalized value back to actual amount."""
        return max(value * self.max_amount, 0.0)

    def extract_amount(self, text: str) -> float:
        """Extract amount from text via regex or fallback to model."""
        patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*dollars?',
            r'(\d+(?:\.\d{2})?)'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return float(matches[0])

        # Try word2number
        words = text.lower().replace('-', ' ').split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 6, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                try:
                    amount = w2n.word_to_num(phrase)
                    if amount > 0:
                        return float(amount)
                except ValueError:
                    continue

        # Fallback to model prediction
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            _, amount_pred, _ = self(
                inputs["input_ids"], inputs["attention_mask"])
            return self.denormalize_amount(abs(amount_pred.item()))

    def extract_date(self, text: str) -> Optional[str]:
        try:
            patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}-\d{2}-\d{4}',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return parser.parse(match.group()).isoformat()

            return datetime.now().isoformat()
        except:
            return datetime.now().isoformat()

    def predict(self, text: str, categories: List[str]) -> Dict[str, any]:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)

        with torch.no_grad():
            category_logits, amount_pred, recurring_pred = self(
                inputs["input_ids"], inputs["attention_mask"]
            )

            probs = torch.softmax(category_logits, dim=-1).squeeze().tolist()
            if not isinstance(probs, list):
                probs = [probs]

            category_scores = {cat: float(score)
                               for cat, score in zip(categories, probs)}
            predicted_category = max(
                category_scores.items(), key=lambda x: x[1])[0]
            confidence = max(probs)

            amount = self.extract_amount(text)
            date = self.extract_date(text)
            is_recurring = bool(recurring_pred.squeeze().item() > 0.5)

            return {
                "text": text,
                "amount": round(amount, 2),
                "category": {
                    "label": predicted_category,
                    "scores": category_scores
                },
                "date": date,
                "is_recurring": is_recurring,
                "confidence": float(confidence)
            }


def load_model(model_dir: str = "models") -> ExpenseModel:
    model_path = Path(model_dir)
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)

    model = ExpenseModel(
        model_name=str(model_path / "transformer"),
        num_categories=config["num_categories"],
        max_amount=config.get("max_amount", 1000.0)
    )
    model.load_state_dict(torch.load(
        model_path / "model.pt", map_location=torch.device('cpu')))
    return model
