import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class PotholeDetectionTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None

    def setup_training(self, learning_rate=0.001):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=50):
        if self.optimizer is None:
            self.setup_training()

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Load best model
        self.model.load_state_dict(torch.load('models/best_model.pth'))
        return self.model

    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'sequence_length': self.model.sequence_length,
                'n_features': self.model.n_features,
                'n_classes': self.model.n_classes
            }
        }, filepath)
