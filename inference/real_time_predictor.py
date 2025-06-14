import torch
import numpy as np
import time
from collections import deque
from data.dataset import RealTimeBuffer
import os


class IntegratedPotholeDetector:
    def __init__(self, model_path, sequence_length=50, confidence_threshold=0.7,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Real-time pothole detection predictor"""
        self.model_path = model_path
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold

        # Data buffer and preprocessor
        self.buffer = RealTimeBuffer(sequence_length, n_features=6)

        # Load model
        self.model = self._load_model()

        # Detection callback and smoothing
        self.detection_callback = None
        self.prediction_history = deque(maxlen=5)
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds

        # Statistics
        self.total_predictions = 0
        self.pothole_detections = 0

    def _load_model(self):
        """Load the trained ML model"""
        try:
            if os.path.exists(self.model_path):
                from models.cnn_lstm_model import CNNLSTMPotholeDetector

                checkpoint = torch.load(self.model_path, map_location=self.device)

                if 'model_config' in checkpoint:
                    config = checkpoint['model_config']
                    model = CNNLSTMPotholeDetector(
                        sequence_length=config['sequence_length'],
                        n_features=config['n_features'],
                        n_classes=config['n_classes']
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Direct model load
                    model = checkpoint

                model = model.to(self.device)
                model.eval()
                print("✅ ML model loaded successfully")
                return model
            else:
                print("❌ Model file not found")
                return None
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return None

    def set_detection_callback(self, callback):
        """Set callback function for pothole detections"""
        self.detection_callback = callback

    def process_sensor_data(self, accelerometer_values, gyroscope_values):
        """Process sensor data and make prediction"""
        # Add data to buffer
        self.buffer.add_sample(accelerometer_values, gyroscope_values)

        # Calculate magnitudes for fallback detection
        accel_magnitude = np.sqrt(np.sum(np.array(accelerometer_values) ** 2))
        gyro_magnitude = np.sqrt(np.sum(np.array(gyroscope_values) ** 2))

        # Use ML model if available and buffer is ready
        if self.model is not None and self.buffer.is_ready():
            try:
                sequence = self.buffer.get_sequence(normalize=self.buffer.is_fitted)
                sequence = sequence.to(self.device)

                with torch.no_grad():
                    output = self.model(sequence)
                    confidence = output.item()

                # Store prediction for smoothing
                self.prediction_history.append(confidence)
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

                    result = {
                        'pothole_detected': True,
                        'confidence': smoothed_confidence,
                        'raw_confidence': confidence,
                        'message': f'ML: Pothole detected! (conf: {smoothed_confidence:.3f})'
                    }

                    if self.detection_callback:
                        self.detection_callback(result)

                    return result

                self.total_predictions += 1
                return {
                    'pothole_detected': False,
                    'confidence': smoothed_confidence,
                    'raw_confidence': confidence,
                    'message': f'ML: Road OK (conf: {smoothed_confidence:.3f})'
                }

            except Exception as e:
                print(f"ML prediction error: {e}")

        # Fallback to rule-based detection
        if accel_magnitude > 15.0 or gyro_magnitude > 2.5:
            confidence = min(0.9, (accel_magnitude + gyro_magnitude) / 20.0)
            current_time = time.time()

            if (current_time - self.last_detection_time) > self.detection_cooldown:
                self.last_detection_time = current_time
                self.pothole_detections += 1

                result = {
                    'pothole_detected': True,
                    'confidence': confidence,
                    'message': f'Rule: Pothole detected! (accel: {accel_magnitude:.2f})'
                }

                if self.detection_callback:
                    self.detection_callback(result)

                return result

        self.total_predictions += 1
        return {
            'pothole_detected': False,
            'confidence': accel_magnitude / 20.0,
            'message': f'Rule: Road OK (accel: {accel_magnitude:.2f})'
        }

    def get_statistics(self):
        """Get prediction statistics"""
        detection_rate = (self.pothole_detections / max(self.total_predictions, 1)) * 100
        return {
            'total_predictions': self.total_predictions,
            'pothole_detections': self.pothole_detections,
            'detection_rate': detection_rate,
            'model_loaded': self.model is not None
        }
