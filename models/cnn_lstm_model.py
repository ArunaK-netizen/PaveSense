import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, LSTM, Linear, Dropout, BatchNorm1d


class CNNLSTMPotholeDetector(nn.Module):
    def __init__(self, sequence_length=50, n_features=6, n_classes=2,
                 cnn_filters=[64, 128, 64], lstm_hidden=100, lstm_layers=2):
        """
        CNN-LSTM model for pothole detection using PyTorch

        Args:
            sequence_length: Length of input sequences (time steps)
            n_features: Number of features (3 accel + 3 gyro = 6)
            n_classes: Number of classes (pothole=1, no_pothole=0)
            cnn_filters: List of CNN filter sizes
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
        """
        super(CNNLSTMPotholeDetector, self).__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # CNN layers for feature extraction
        self.conv1 = Conv1d(n_features, cnn_filters[0], kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(cnn_filters[0])
        self.pool1 = MaxPool1d(kernel_size=2)
        self.dropout1 = Dropout(0.2)

        self.conv2 = Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(cnn_filters[1])
        self.pool2 = MaxPool1d(kernel_size=2)
        self.dropout2 = Dropout(0.2)

        self.conv3 = Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=3, padding=1)
        self.bn3 = BatchNorm1d(cnn_filters[2])
        self.dropout3 = Dropout(0.2)

        # Calculate sequence length after CNN layers
        cnn_output_length = sequence_length // 4  # Two MaxPool1d with kernel_size=2

        # LSTM layers for temporal dependencies
        self.lstm = LSTM(
            input_size=cnn_filters[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Dense layers for classification
        self.fc1 = Linear(lstm_hidden, 64)
        self.dropout4 = Dropout(0.4)
        self.fc2 = Linear(64, 32)
        self.dropout5 = Dropout(0.3)

        # Output layer
        if n_classes == 2:
            self.output = Linear(32, 1)  # Binary classification
        else:
            self.output = Linear(32, n_classes)  # Multi-class

    def forward(self, x):
        """
        Forward pass
        Input shape: (batch_size, sequence_length, n_features)
        """
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

        # Use the last output for classification
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

    def predict_proba(self, x):
        """Predict probabilities"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict(self, x, threshold=0.5):
        """Make binary predictions"""
        probs = self.predict_proba(x)
        if self.n_classes == 2:
            return (probs > threshold).int()
        else:
            return torch.argmax(probs, dim=1)
