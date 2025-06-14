import torch
import numpy as np
import time
from collections import deque
from data.dataset import RealTimeBuffer
import os


class IntegratedPotholeDetector:
    def __init__(self, model_path, sequence_length=50, confidence_threshold=0.7):
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.detection_callback = None

        # Initialize buffer
        self.buffer = RealTimeBuffer(sequence_length, n_features=6)

        # Load model
        self.model = self._load_model(model_path)

        # Detection cooldown
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # 2 seconds

    def _load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                from models.cnn_lstm_model import CNNLSTMPotholeDetector

                checkpoint = torch.load(model_path, map_location='cpu')

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

                model.eval()
                print("✅ ML model loaded successfully")
                return model
            else:
                print("⚠️ Model file not found")
                return None
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return None

    def set_detection_callback(self, callback):
        self.detection_callback = callback

    def process_sensor_data(self, accelerometer_values, gyroscope_values):
        # Add to buffer
        self.buffer.add_sample(accelerometer_values, gyroscope_values)

        # Calculate basic metrics for fallback
        accel_magnitude = np.sqrt(np.sum(np.array(accelerometer_values) ** 2))
        gyro_magnitude = np.sqrt(np.sum(np.array(gyroscope_values) ** 2))

        # Use ML model if available
        if self.model is not None and self.buffer.is_ready():
            try:
                sequence = self.buffer.get_sequence()
                with torch.no_grad():
                    output = self.model(sequence)
                    confidence = output.item()

                current_time = time.time()
                pothole_detected = (
                        confidence > self.confidence_threshold and
                        (current_time - self.last_detection_time) > self.detection_cooldown
                )

                if pothole_detected:
                    self.last_detection_time = current_time
                    result = {
                        'pothole_detected': True,
                        'confidence': confidence,
                        'message': f'ML: Pothole detected! (conf: {confidence:.3f})'
                    }

                    if self.detection_callback:
                        self.detection_callback(result)

                    return result

                return {
                    'pothole_detected': False,
                    'confidence': confidence,
                    'message': f'ML: Road OK (conf: {confidence:.3f})'
                }

            except Exception as e:
                print(f"ML prediction error: {e}")

        # Fallback to rule-based detection
        if accel_magnitude > 15.0 or gyro_magnitude > 2.5:
            confidence = min(0.9, (accel_magnitude + gyro_magnitude) / 20.0)
            current_time = time.time()

            if (current_time - self.last_detection_time) > self.detection_cooldown:
                self.last_detection_time = current_time
                result = {
                    'pothole_detected': True,
                    'confidence': confidence,
                    'message': f'Rule: Pothole detected! (accel: {accel_magnitude:.2f})'
                }

                if self.detection_callback:
                    self.detection_callback(result)

                return result

        return {
            'pothole_detected': False,
            'confidence': accel_magnitude / 20.0,
            'message': f'Rule: Road OK (accel: {accel_magnitude:.2f})'
        }
