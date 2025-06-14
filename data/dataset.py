import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque


class SensorDataset(Dataset):
    def __init__(self, data, labels, sequence_length=50):
        self.sequence_length = sequence_length
        self.sequences, self.labels = self._create_sequences(data, labels)

    def _create_sequences(self, data, labels):
        sequences = []
        sequence_labels = []

        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]

            sequences.append(seq)
            sequence_labels.append(label)

        return np.array(sequences), np.array(sequence_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor([label])


class RealTimeBuffer:
    def __init__(self, sequence_length=50, n_features=6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.buffer = np.zeros((sequence_length, n_features))

    def add_sample(self, accelerometer_data, gyroscope_data):
        sample = np.concatenate([accelerometer_data, gyroscope_data])
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = sample

    def get_sequence(self):
        return torch.FloatTensor(self.buffer).unsqueeze(0)  # Add batch dimension

    def is_ready(self):
        return not np.allclose(self.buffer[0], 0)
