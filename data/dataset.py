import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


class SensorDataset(Dataset):
    def __init__(self, data, labels, sequence_length=50, transform=None):
        """
        Dataset for sensor data

        Args:
            data: Raw sensor data (accelerometer + gyroscope)
            labels: Corresponding labels (0 for no pothole, 1 for pothole)
            sequence_length: Length of each sequence
            transform: Optional data transforms
        """
        self.sequence_length = sequence_length
        self.transform = transform

        # Create sequences
        self.sequences, self.labels = self._create_sequences(data, labels)

    def _create_sequences(self, data, labels):
        """Create overlapping sequences from continuous data"""
        sequences = []
        sequence_labels = []

        # Convert to numpy if pandas DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(labels, pd.Series):
            labels = labels.values

        # Create sliding window sequences
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]  # Use label of last timestep

            sequences.append(seq)
            sequence_labels.append(label)

        return np.array(sequences), np.array(sequence_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        if self.transform:
            sequence = self.transform(sequence)

        return torch.FloatTensor(sequence), torch.FloatTensor([label])


class RealTimeBuffer:
    """Buffer for real-time sensor data processing"""

    def __init__(self, sequence_length=50, n_features=6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.buffer = np.zeros((sequence_length, n_features))
        self.scaler = StandardScaler()
        self.is_fitted = False

    def add_sample(self, accelerometer_data, gyroscope_data):
        """Add new sensor sample to buffer"""
        # Combine accelerometer and gyroscope data
        sample = np.concatenate([accelerometer_data, gyroscope_data])

        # Shift buffer and add new sample
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = sample

    def get_sequence(self, normalize=True):
        """Get current sequence for prediction"""
        sequence = self.buffer.copy()

        if normalize and self.is_fitted:
            sequence = self.scaler.transform(sequence)

        return torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension

    def fit_scaler(self, data):
        """Fit scaler on training data"""
        self.scaler.fit(data)
        self.is_fitted = True

    def is_ready(self):
        """Check if buffer has enough data for prediction"""
        return not np.allclose(self.buffer[0], 0)  # Check if first sample is not zero


def create_data_loaders(train_data, train_labels, val_data, val_labels,
                        sequence_length=50, batch_size=32, num_workers=4):
    """Create training and validation data loaders"""

    train_dataset = SensorDataset(train_data, train_labels, sequence_length)
    val_dataset = SensorDataset(val_data, val_labels, sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
