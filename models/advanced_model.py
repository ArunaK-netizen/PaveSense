"""
PaveSense Advanced Pothole Detection Model
==========================================
Multi-Scale CNN + Self-Attention + BiLSTM

5-class event classifier:
  0: normal       - smooth road / stationary
  1: pothole      - sharp Z-drop → rebound, asymmetric impulse
  2: speed_bump   - gradual Z-rise, symmetric profile
  3: phone_drop   - freefall (mag→0) then impact spike
  4: disturbance  - braking, turning, rough road, door slam, etc.

Input: (batch, sequence_length, 13)  — 13 engineered features
Output: (batch, 5) — class probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleCNN(nn.Module):
    """Extract features at multiple temporal scales using parallel Conv1D banks."""

    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        # Small kernel: local spike patterns (pothole impact)
        self.conv_small = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # Medium kernel: event-level patterns (speed bump shape)
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        # Large kernel: full event envelope (braking, turning)
        self.conv_large = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)

        # Projection to merge multi-scale features
        self.projection = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels * 2, kernel_size=1),
            nn.BatchNorm1d(out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        """x: (batch, channels, seq_len)"""
        s = self.conv_small(x)
        m = self.conv_medium(x)
        l = self.conv_large(x)

        # Concatenate multi-scale features
        combined = torch.cat([s, m, l], dim=1)  # (batch, out*3, seq_len)
        combined = self.projection(combined)     # (batch, out*2, seq_len)
        combined = self.pool(combined)           # (batch, out*2, seq_len//2)
        combined = self.dropout(combined)

        return combined


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention to focus on impact moments."""

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """x: (batch, seq_len, hidden_size)"""
        batch_size, seq_len, _ = x.shape
        residual = x

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.output_proj(attn_output)

        # Residual connection + layer norm
        output = self.layer_norm(residual + attn_output)
        return output


class PaveSenseModel(nn.Module):
    """
    Advanced pothole detection model.

    Architecture:
        MultiScale CNN → BiLSTM → Self-Attention → Classification Head

    Designed to discriminate potholes from speed bumps, phone drops,
    braking, and other false-positive sources.
    """

    EVENT_CLASSES = ['normal', 'pothole', 'speed_bump', 'phone_drop', 'disturbance']
    N_CLASSES = 5
    N_ENGINEERED_FEATURES = 13

    def __init__(self, sequence_length=100, n_features=13, n_classes=5,
                 cnn_channels=64, lstm_hidden=128, lstm_layers=2,
                 num_attention_heads=4, dropout=0.3):
        super().__init__()

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes

        # Multi-scale CNN feature extractor
        self.cnn = MultiScaleCNN(n_features, out_channels=cnn_channels)
        cnn_out_features = cnn_channels * 2  # from projection

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out_features = lstm_hidden * 2  # bidirectional

        # Self-attention layer
        self.attention = SelfAttention(lstm_out_features, num_heads=num_attention_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, sequence_length, n_features) — engineered features

        Returns:
            (batch, n_classes) — class logits (apply softmax outside for probs)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale CNN feature extraction
        x = self.cnn(x)  # (batch, cnn_out, seq_len//2)

        # Transpose back for LSTM: (batch, seq_len//2, cnn_out)
        x = x.transpose(1, 2)

        # BiLSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # (batch, seq_len//2, lstm_hidden*2)

        # Self-attention
        attn_out = self.attention(lstm_out)  # (batch, seq_len//2, lstm_hidden*2)

        # Global average pooling over time dimension
        x = attn_out.mean(dim=1)  # (batch, lstm_hidden*2)

        # Classification
        logits = self.classifier(x)  # (batch, n_classes)

        return logits

    def predict_proba(self, x):
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x, threshold=0.5):
        """
        Get prediction with pothole-specific threshold.

        Returns:
            dict with 'class', 'confidence', 'is_pothole', 'probabilities'
        """
        probs = self.predict_proba(x)
        pred_class = torch.argmax(probs, dim=-1)
        pred_confidence = probs.max(dim=-1).values

        # For pothole class specifically, allow custom threshold
        pothole_prob = probs[:, 1]  # index 1 = pothole
        is_pothole = pothole_prob > threshold

        return {
            'class_idx': pred_class,
            'class_name': [self.EVENT_CLASSES[i] for i in pred_class.cpu().tolist()],
            'confidence': pred_confidence,
            'is_pothole': is_pothole,
            'pothole_confidence': pothole_prob,
            'probabilities': probs,
        }

    @staticmethod
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
