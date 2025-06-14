import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMPotholeDetector(nn.Module):
    def __init__(self, sequence_length=50, n_features=6, n_classes=2):
        super(CNNLSTMPotholeDetector, self).__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes

        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        # LSTM layers for temporal dependencies
        cnn_output_length = sequence_length // 4
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=100,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        # Dense layers for classification
        self.fc1 = nn.Linear(100, 64)
        self.dropout4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout5 = nn.Dropout(0.3)

        # Output layer
        if n_classes == 2:
            self.output = nn.Linear(32, 1)  # Binary classification
        else:
            self.output = nn.Linear(32, n_classes)

    def forward(self, x):
        """Forward pass - input shape: (batch_size, sequence_length, n_features)"""
        # Reshape for CNN: (batch_size, n_features, sequence_length)
        x = x.transpose(1, 2)

        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        # Reshape for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)

        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last timestep

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)

        # Output layer
        x = self.output(x)

        if self.n_classes == 2:
            x = torch.sigmoid(x)  # Binary classification
        else:
            x = F.softmax(x, dim=1)  # Multi-class

        return x
