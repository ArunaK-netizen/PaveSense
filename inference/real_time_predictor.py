import torch
import numpy as np
import threading
import time
from collections import deque
from data.dataset import RealTimeBuffer
from data.data_preprocessor import SensorDataPreprocessor


class RealTimePotholePredictor:
    def __init__(self, model, sequence_length=50, confidence_threshold=0.7,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Real-time pothole detection predictor

        Args:
            model: Trained PyTorch model
            sequence_length: Length of input sequences
            confidence_threshold: Minimum confidence for pothole detection
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold

        # Data buffer and preprocessor
        self.buffer = RealTimeBuffer(sequence_length, n_features=6)
        self.preprocessor = SensorDataPreprocessor()

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)  # Last 5 predictions
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # 2 seconds cooldown between detections

        # Statistics
        self.total_predictions = 0
        self.pothole_detections = 0

    def setup_with_training_data(self, training_data):
        """Setup scaler with training data statistics"""
        self.buffer.fit_scaler(training_data)

    def add_sensor_data(self, accelerometer_values, gyroscope_values):
        """
        Add new sensor data point

        Args:
            accelerometer_values: [x, y, z] accelerometer readings
            gyroscope_values: [x, y, z] gyroscope readings
        """
        self.buffer.add_sample(accelerometer_values, gyroscope_values)

    def predict(self):
        """
        Make pothole prediction based on current buffer

        Returns:
            dict: Prediction results with confidence and decision
        """
        if not self.buffer.is_ready():
            return {
                'pothole_detected': False,
                'confidence': 0.0,
                'message': 'Buffer not ready'
            }

        try:
            # Get sequence from buffer
            sequence = self.buffer.get_sequence(normalize=True)
            sequence = sequence.to(self.device)

            # Make prediction
            with torch.no_grad():
                output = self.model(sequence)
                confidence = output.item()

            # Store prediction for smoothing
            self.prediction_history.append(confidence)

            # Smooth predictions using moving average
            smoothed_confidence = np.mean(list(self.prediction_history))

            # Determine if pothole is detected
            current_time = time.time()
            pothole_detected = (
                    smoothed_confidence > self.confidence_threshold and
                    (current_time - self.last_detection_time) > self.detection_cooldown
            )

            if pothole_detected:
                self.last_detection_time = current_time
                self.pothole_detections += 1

            self.total_predictions += 1

            return {
                'pothole_detected': pothole_detected,
                'confidence': smoothed_confidence,
                'raw_confidence': confidence,
                'message': 'Pothole detected!' if pothole_detected else 'Road OK'
            }

        except Exception as e:
            return {
                'pothole_detected': False,
                'confidence': 0.0,
                'message': f'Prediction error: {str(e)}'
            }

    def get_statistics(self):
        """Get prediction statistics"""
        detection_rate = (self.pothole_detections / max(self.total_predictions, 1)) * 100
        return {
            'total_predictions': self.total_predictions,
            'pothole_detections': self.pothole_detections,
            'detection_rate': detection_rate
        }

    def reset_statistics(self):
        """Reset prediction statistics"""
        self.total_predictions = 0
        self.pothole_detections = 0
        self.last_detection_time = 0
        self.prediction_history.clear()


class IntegratedPotholeDetector:
    """Integration with existing sensor data collection system"""

    def __init__(self, model_path, sequence_length=50):
        # Load model
        self.model = self._load_model(model_path)
        self.predictor = RealTimePotholePredictor(self.model, sequence_length)

        # Callback for detections
        self.detection_callback = None

    def _load_model(self, model_path):
        """Load trained model"""
        from models.cnn_lstm_model import CNNLSTMPotholeDetector

        checkpoint = torch.load(model_path)
        config = checkpoint['model_config']

        model = CNNLSTMPotholeDetector(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            n_classes=config['n_classes'],
            lstm_hidden=config['lstm_hidden'],
            lstm_layers=config['lstm_layers']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def set_detection_callback(self, callback):
        """Set callback function for pothole detections"""
        self.detection_callback = callback

    def process_sensor_data(self, accelerometer_values, gyroscope_values):
        """
        Process sensor data and make prediction
        This replaces the rule-based process_sensor_data function
        """
        # Add data to predictor
        self.predictor.add_sensor_data(accelerometer_values, gyroscope_values)

        # Make prediction
        result = self.predictor.predict()

        # Call detection callback if pothole detected
        if result['pothole_detected'] and self.detection_callback:
            self.detection_callback(result)

        return result
