import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# Import your CNN-LSTM model
from models.cnn_lstm_model import CNNLSTMPotholeDetector


class RealDataTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None

    def load_processed_data(self):
        """Load processed dataset"""
        print("📂 Loading processed dataset...")

        try:
            X_train = np.load("datasets/processed/X_train.npy")
            X_val = np.load("datasets/processed/X_val.npy")
            X_test = np.load("datasets/processed/X_test.npy")
            y_train = np.load("datasets/processed/y_train.npy")
            y_val = np.load("datasets/processed/y_val.npy")
            y_test = np.load("datasets/processed/y_test.npy")

            with open("datasets/processed/metadata.json", "r") as f:
                metadata = json.load(f)

            print(f"✅ Dataset loaded successfully:")
            print(f"   Training: {len(X_train)} sequences")
            print(f"   Validation: {len(X_val)} sequences")
            print(f"   Test: {len(X_test)} sequences")
            print(f"   Sequence length: {metadata['sequence_length']}")
            print(f"   Features: {metadata['n_features']}")

            return X_train, X_val, X_test, y_train, y_val, y_test, metadata

        except FileNotFoundError:
            print("❌ Processed dataset not found!")
            print("💡 Run 'process_dataset.py' first to process your collected data")
            exit(1)

    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """Create PyTorch data loaders"""
        print(f"🔄 Creating data loaders (batch_size={batch_size})...")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def create_model(self, metadata):
        """Create CNN-LSTM model"""
        print("🤖 Creating CNN-LSTM model...")

        self.model = CNNLSTMPotholeDetector(
            sequence_length=metadata['sequence_length'],
            n_features=metadata['n_features'],
            n_classes=2
        ).to(self.device)

        # Setup training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.BCELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✅ Model created:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        return self.model

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                predictions = (outputs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))

        return avg_loss, accuracy, all_predictions, all_targets

    def train_model(self, train_loader, val_loader, epochs=100):
        """Complete training loop"""
        print(f"🚀 Starting training for {epochs} epochs...")

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc, val_predictions, val_targets = self.validate_epoch(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch + 1:3d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_config': {
                        'sequence_length': self.model.sequence_length,
                        'n_features': self.model.n_features,
                        'n_classes': self.model.n_classes
                    },
                    'training_history': history,
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': val_acc
                }, 'models/trained_pothole_model.pth')

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def evaluate_model(self, test_loader):
        """Evaluate model on test set"""
        print("\n📊 Evaluating model on test set...")

        # Load best model
        checkpoint = torch.load('models/trained_pothole_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                probabilities = outputs.cpu().numpy()
                predictions = (outputs > 0.5).float().cpu().numpy()
                targets_np = targets.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets_np.flatten())
                all_probabilities.extend(probabilities.flatten())

        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))

        print(f"✅ Test Results:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(all_targets, all_predictions,
                                    target_names=['Normal Road', 'Pothole']))

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        print(f"\n🔢 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"              Normal  Pothole")
        print(f"Actual Normal   {cm[0, 0]:4d}     {cm[0, 1]:4d}")
        print(f"      Pothole   {cm[1, 0]:4d}     {cm[1, 1]:4d}")

        return accuracy, all_predictions, all_targets, all_probabilities

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(history['train_accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Training history plot saved to 'models/training_history.png'")


def main():
    print("🚗 Training CNN-LSTM Model on Real Pothole Data")
    print("=" * 60)

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")

    trainer = RealDataTrainer(device)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, metadata = trainer.load_processed_data()

    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )

    # Create model
    model = trainer.create_model(metadata)

    # Train model
    history = trainer.train_model(train_loader, val_loader, epochs=100)

    # Plot training history
    trainer.plot_training_history(history)

    # Evaluate model
    accuracy, predictions, targets, probabilities = trainer.evaluate_model(test_loader)

    print(f"\n🎉 Training completed!")
    print(f"✅ Best model saved to 'models/trained_pothole_model.pth'")
    print(f"📊 Final test accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"\n🚀 Ready to use with your main detection system!")


if __name__ == "__main__":
    main()
