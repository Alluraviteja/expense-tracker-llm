"""Training script for the expense tracking model."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import os
from pathlib import Path
from tqdm import tqdm
import json
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from app.dataset import ExpenseDataset, initialize_sample_data
from app.model import ExpenseModel


class ExpenseTrainingDataset(Dataset):
    """Dataset class for training the expense model."""

    def __init__(self, texts: List[str], categories: List[Dict[str, float]],
                 metadata: List[Dict[str, Any]], tokenizer, max_amount: float = 1000.0):
        self.texts = texts
        self.categories = categories
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.max_amount = max_amount

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        category = self.categories[idx]
        meta = self.metadata[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        # Convert category dict to tensor
        category_tensor = torch.tensor(
            list(category.values()), dtype=torch.float)

        # Normalize amount between 0 and 1
        normalized_amount = min(meta["amount"] / self.max_amount, 1.0)
        amount = torch.tensor([normalized_amount], dtype=torch.float)

        is_recurring = torch.tensor(
            [int(meta["is_recurring"])], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "categories": category_tensor,
            "amount": amount,
            "is_recurring": is_recurring
        }


def plot_confusion_matrix(true_categories, pred_categories, categories, save_dir: str = "models"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(true_categories, pred_categories)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories,
                yticklabels=categories)
    plt.title('Category Prediction Confusion Matrix')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    save_path = Path(save_dir)
    plt.savefig(save_path / "confusion_matrix.png")
    plt.close()


def evaluate_model(model: ExpenseModel, dataset: Dataset, categories: List[str]):
    """Evaluate model performance and generate confusion matrix."""
    model.eval()
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_true_categories = []
    all_pred_categories = []
    all_true_amounts = []
    all_pred_amounts = []
    all_true_recurring = []
    all_pred_recurring = []

    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            category_logits, amount_pred, recurring_pred = model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            # Get predictions
            pred_categories = torch.argmax(category_logits, dim=1)
            true_categories = torch.argmax(batch["categories"], dim=1)

            # Store predictions and true values
            all_true_categories.extend(true_categories.cpu().numpy())
            all_pred_categories.extend(pred_categories.cpu().numpy())
            all_true_amounts.extend(batch["amount"].cpu().numpy())
            all_pred_amounts.extend(amount_pred.cpu().numpy())
            all_true_recurring.extend(batch["is_recurring"].cpu().numpy())
            all_pred_recurring.extend(
                (recurring_pred > 0.5).float().cpu().numpy())

    # Generate confusion matrix
    plot_confusion_matrix(all_true_categories, all_pred_categories, categories)

    # Print classification report
    print("\nCategory Classification Report:")
    print(classification_report(all_true_categories, all_pred_categories,
                                target_names=categories))

    # Calculate amount prediction metrics
    amount_mse = np.mean((np.array(all_true_amounts) -
                         np.array(all_pred_amounts)) ** 2)
    amount_mae = np.mean(
        np.abs(np.array(all_true_amounts) - np.array(all_pred_amounts)))

    print("\nAmount Prediction Metrics:")
    print(f"MSE: {amount_mse:.4f}")
    print(f"MAE: {amount_mae:.4f}")

    # Calculate recurring prediction accuracy
    recurring_accuracy = np.mean(
        np.array(all_true_recurring) == np.array(all_pred_recurring))
    print("\nRecurring Prediction Accuracy:", f"{recurring_accuracy:.4f}")


def train_model(model: ExpenseModel, train_dataset: Dataset,
                num_epochs: int = 10, batch_size: int = 8,
                learning_rate: float = 1e-5) -> ExpenseModel:
    """Train the expense model."""

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Loss functions
    category_criterion = torch.nn.CrossEntropyLoss()
    amount_criterion = torch.nn.MSELoss()
    recurring_criterion = torch.nn.BCELoss()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        category_losses = 0
        amount_losses = 0
        recurring_losses = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            category_logits, amount_pred, recurring_pred = model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            # Get category target indices
            category_targets = torch.argmax(batch["categories"], dim=1)

            # Calculate individual losses
            cat_loss = category_criterion(category_logits, category_targets)
            amt_loss = amount_criterion(amount_pred, batch["amount"])
            rec_loss = recurring_criterion(
                recurring_pred, batch["is_recurring"])

            # Scale losses to balance them
            category_loss = cat_loss * 1.0  # Base weight for category loss
            amount_loss = amt_loss * 0.5    # Reduced weight for amount loss
            recurring_loss = rec_loss * 0.3  # Reduced weight for recurring loss

            # Combined loss
            loss = category_loss + amount_loss + recurring_loss

            # Track individual losses
            category_losses += cat_loss.item()
            amount_losses += amt_loss.item()
            recurring_losses += rec_loss.item()
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update progress bar with detailed losses
            progress_bar.set_postfix({
                "total_loss": f"{loss.item():.4f}",
                "cat_loss": f"{cat_loss.item():.4f}",
                "amt_loss": f"{amt_loss.item():.4f}",
                "rec_loss": f"{rec_loss.item():.4f}"
            })

        # Calculate average losses
        num_batches = len(train_loader)
        avg_total_loss = total_loss / num_batches
        avg_category_loss = category_losses / num_batches
        avg_amount_loss = amount_losses / num_batches
        avg_recurring_loss = recurring_losses / num_batches

        print(f"Epoch {epoch + 1} - "
              f"Total Loss: {avg_total_loss:.4f}, "
              f"Category Loss: {avg_category_loss:.4f}, "
              f"Amount Loss: {avg_amount_loss:.4f}, "
              f"Recurring Loss: {avg_recurring_loss:.4f}")

    return model


def save_model(model: ExpenseModel, save_dir: str = "models"):
    """Save the trained model."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save the full model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': model.tokenizer,
        'config': {
            'model_name': model.transformer.config._name_or_path,
            'num_categories': model.category_classifier[-1].out_features
        }
    }, save_path / "model.pt")

    # Save configuration separately for easier loading
    with open(save_path / "config.json", "w") as f:
        json.dump({
            'model_name': model.transformer.config._name_or_path,
            'num_categories': model.category_classifier[-1].out_features
        }, f)


def load_model(model_dir: str = "models") -> ExpenseModel:
    """Load a trained model."""
    model_path = Path(model_dir)

    # Load the saved state
    checkpoint = torch.load(model_path / "model.pt")

    # Create model instance
    model = ExpenseModel(
        model_name=checkpoint['config']['model_name'],
        num_categories=checkpoint['config']['num_categories']
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.tokenizer = checkpoint['tokenizer']

    return model


def main():
    """Main training function."""
    # Initialize dataset with sample data
    dataset = initialize_sample_data()

    # Get training data
    texts, categories, metadata = dataset.get_training_data()

    # Create model
    model = ExpenseModel(num_categories=len(dataset.CATEGORIES))

    # Create training dataset
    train_dataset = ExpenseTrainingDataset(
        texts=texts,
        categories=categories,
        metadata=metadata,
        tokenizer=model.tokenizer
    )

    # Train model
    print("Training model...")
    model = train_model(model, train_dataset)

    # Evaluate model and generate confusion matrix
    print("\nEvaluating model performance...")
    evaluate_model(model, train_dataset, dataset.CATEGORIES)

    # Save model
    print("\nSaving model...")
    save_model(model)
    print("Training complete!")


if __name__ == "__main__":
    main()
