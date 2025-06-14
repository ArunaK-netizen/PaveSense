import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
from models.model_utils import calculate_metrics, EarlyStopping


class PotholeDetectionTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def setup_training(self, learning_rate=0.001, weight_decay=1e-5):
        """Setup optimizer, loss function, and scheduler"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Binary cross-entropy for binary classification
        self.criterion = nn.BCELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions / total_samples:.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Store predictions and targets for metrics
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
                targets_np = targets.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets_np.flatten())
                all_probabilities.extend(probabilities.flatten())

        avg_loss = total_loss / len(val_loader)

        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )

        return avg_loss, metrics

    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=15):
        """Complete training loop"""
        if self.optimizer is None:
            self.setup_training()

        # Early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)

            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])

            # Print epoch results
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            if 'auc' in val_metrics:
                print(f"Val AUC: {val_metrics['auc']:.4f}")
            print("-" * 50)

            # Early stopping check
            if early_stopping(val_loss, self.model):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def save_model(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_config': {
                'sequence_length': self.model.sequence_length,
                'n_features': self.model.n_features,
                'n_classes': self.model.n_classes,
                'lstm_hidden': self.model.lstm_hidden,
                'lstm_layers': self.model.lstm_layers
            }
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from {filepath}")
        return checkpoint.get('model_config', {})
