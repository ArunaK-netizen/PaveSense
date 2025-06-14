import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMPotholeDetector(nn.Module):
    def __init__(self, sequence_length=50, n_features=6, n_classes=2):
        super(CNNLSTMPotholeDetector, self).__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes

        # CNN layers
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        # LSTM layers
        cnn_output_length = sequence_length // 4
        self.lstm = nn.LSTM(64, 100, num_layers=2, batch_first=True, dropout=0.3)

        # Dense layers
        self.fc1 = nn.Linear(100, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, n_features)
        x = x.transpose(1, 2)  # (batch_size, n_features, sequence_length)

        # CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))

        # Back to LSTM format
        x = x.transpose(1, 2)  # (batch_size, sequence_length, features)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last timestep

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.output(x)

        return torch.sigmoid(x)
