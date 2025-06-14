import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from models.cnn_lstm_model import CNNLSTMPotholeDetector
from training.trainer import PotholeDetectionTrainer
from data.dataset import create_data_loaders
from data.data_preprocessor import SensorDataPreprocessor
from models.model_utils import plot_training_history, plot_confusion_matrix


def load_training_data():
    """Load and prepare training data"""
    # This is where you'd load your collected sensor data
    # For now, we'll create synthetic data as example

    # Load your actual data here
    # data = pd.read_csv('sensor_data.csv')
    # Replace with your actual data loading

    # Synthetic data generation (replace with real data)
    np.random.seed(42)
    n_samples = 10000
    sequence_length = 50

    # Generate synthetic sensor data
    data = np.random.randn(n_samples, 6)  # 6 features (3 accel + 3 gyro)

    # Create synthetic labels (1 for pothole, 0 for normal)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    return data, labels


def main():
    # Load and preprocess data
    print("Loading training data...")
    data, labels = load_training_data()

    # Preprocess data
    preprocessor = SensorDataPreprocessor()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        sequence_length=50, batch_size=32
    )

    # Create model
    model = CNNLSTMPotholeDetector(
        sequence_length=50,
        n_features=6,
        n_classes=2,
        cnn_filters=[64, 128, 64],
        lstm_hidden=100,
        lstm_layers=2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = PotholeDetectionTrainer(model)
    trainer.setup_training(learning_rate=0.001)

    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader, val_loader,
        epochs=100, early_stopping_patience=15
    )

    # Save model
    trainer.save_model('models/trained_pothole_model.pth')

    # Plot training history
    plot_training_history(
        history['train_loss'], history['val_loss'],
        history['train_accuracy'], history['val_accuracy']
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
