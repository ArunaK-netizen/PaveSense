import os
import sys
import time
import numpy as np
from collections import deque
import threading
import logging

# ML Model imports
import torch
from inference.real_time_predictor import IntegratedPotholeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplePotholeDetector:
    def __init__(self, model_path='models/trained_pothole_model.pth', sequence_length=50):
        """
        Simple pothole detector that just classifies sensor data
        """
        self.sequence_length = sequence_length
        self.sensor_buffer = {
            "accelerometer": deque(maxlen=sequence_length),
            "gyroscope": deque(maxlen=sequence_length)
        }

        # Load ML model
        self.ml_detector = self._load_model(model_path)

        # Statistics
        self.total_readings = 0
        self.pothole_detections = 0

        logger.info("Simple Pothole Detector initialized")

    def _load_model(self, model_path):
        """Load the trained ML model"""
        try:
            if os.path.exists(model_path):
                ml_detector = IntegratedPotholeDetector(model_path, self.sequence_length)
                logger.info("✅ ML model loaded successfully")
                return ml_detector
            else:
                logger.warning("❌ Model file not found. Using dummy classifier.")
                return self._create_dummy_classifier()
        except Exception as e:
            logger.error(f"❌ Model loading error: {e}")
            return self._create_dummy_classifier()

    def _create_dummy_classifier(self):
        """Create a simple rule-based classifier as fallback"""

        class DummyClassifier:
            def process_sensor_data(self, accel_data, gyro_data):
                # Simple threshold-based detection
                accel_magnitude = np.sqrt(np.sum(np.array(accel_data) ** 2))
                gyro_magnitude = np.sqrt(np.sum(np.array(gyro_data) ** 2))

                # Basic thresholds (adjust based on your data)
                if accel_magnitude > 15.0 or gyro_magnitude > 2.0:
                    confidence = min(0.9, (accel_magnitude + gyro_magnitude) / 20.0)
                    return {
                        'pothole_detected': True,
                        'confidence': confidence,
                        'message': 'Pothole detected (rule-based)'
                    }

                return {
                    'pothole_detected': False,
                    'confidence': 0.1,
                    'message': 'Road OK'
                }

        return DummyClassifier()

    def add_sensor_data(self, accelerometer_data, gyroscope_data):
        """
        Add new sensor data point

        Args:
            accelerometer_data: [x, y, z] accelerometer values
            gyroscope_data: [x, y, z] gyroscope values
        """
        self.sensor_buffer["accelerometer"].append(accelerometer_data)
        self.sensor_buffer["gyroscope"].append(gyroscope_data)

        # Classify if we have enough data
        if len(self.sensor_buffer["accelerometer"]) >= self.sequence_length:
            self._classify_current_data()

    def _classify_current_data(self):
        """Classify current sensor data"""
        try:
            # Get latest sensor readings
            accel_current = np.array(self.sensor_buffer["accelerometer"][-1])
            gyro_current = np.array(self.sensor_buffer["gyroscope"][-1])

            # Classify using ML model
            result = self.ml_detector.process_sensor_data(accel_current, gyro_current)

            # Update statistics
            self.total_readings += 1
            if result['pothole_detected']:
                self.pothole_detections += 1

            # Print results
            self._print_results(accel_current, gyro_current, result)

        except Exception as e:
            logger.error(f"Classification error: {e}")

    def _print_results(self, accel_data, gyro_data, result):
        """Print classification results"""
        # Clear previous line and print new data
        print(f"\r{' ' * 100}", end='\r')  # Clear line

        accel_mag = np.sqrt(np.sum(accel_data ** 2))
        gyro_mag = np.sqrt(np.sum(gyro_data ** 2))

        status_icon = "🚨" if result['pothole_detected'] else "✅"
        confidence_bar = "█" * int(result['confidence'] * 20)

        print(f"{status_icon} Accel: [{accel_data[0]:6.2f}, {accel_data[1]:6.2f}, {accel_data[2]:6.2f}] "
              f"({accel_mag:6.2f}) | Gyro: [{gyro_data[0]:6.2f}, {gyro_data[1]:6.2f}, {gyro_data[2]:6.2f}] "
              f"({gyro_mag:6.2f}) | Confidence: {result['confidence']:.3f} {confidence_bar:<20} | "
              f"Detections: {self.pothole_detections}/{self.total_readings}", end='')

        # Print alert on new line for potholes
        if result['pothole_detected']:
            print(f"\n🚨 POTHOLE DETECTED! Confidence: {result['confidence']:.3f}")

    def get_statistics(self):
        """Get detection statistics"""
        detection_rate = (self.pothole_detections / max(self.total_readings, 1)) * 100
        return {
            'total_readings': self.total_readings,
            'pothole_detections': self.pothole_detections,
            'detection_rate': detection_rate
        }


def simulate_sensor_data():
    """
    Simulate sensor data for testing
    In real implementation, replace this with actual sensor data collection
    """
    while True:
        # Generate random sensor data (replace with real sensor readings)
        accel_data = [
            np.random.normal(0, 2),  # x
            np.random.normal(0, 2),  # y
            np.random.normal(9.8, 1)  # z (gravity)
        ]

        gyro_data = [
            np.random.normal(0, 0.5),  # x
            np.random.normal(0, 0.5),  # y
            np.random.normal(0, 0.5)  # z
        ]

        # Occasionally add "pothole" data (high acceleration/gyroscope values)
        if np.random.random() < 0.05:  # 5% chance of pothole
            accel_data[0] += np.random.uniform(10, 20)  # Sharp acceleration
            accel_data[1] += np.random.uniform(5, 15)
            gyro_data[0] += np.random.uniform(2, 5)  # Sharp rotation

        yield accel_data, gyro_data


def main():
    """Main function to run the pothole detector"""
    print("🚗 Simple Pothole Detection System")
    print("=" * 60)

    # Initialize detector
    detector = SimplePotholeDetector()

    print("📊 Starting sensor data classification...")
    print("📱 In real implementation, connect your mobile sensor data here")
    print("🔄 Press Ctrl+C to stop\n")

    try:
        # Simulate continuous sensor data (replace with real sensor input)
        sensor_generator = simulate_sensor_data()

        for accel_data, gyro_data in sensor_generator:
            # Add sensor data and classify
            detector.add_sensor_data(accel_data, gyro_data)

            # Small delay to simulate real sensor sampling rate
            time.sleep(0.1)  # 10 Hz sampling rate

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping detection system...")

        # Print final statistics
        stats = detector.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total readings: {stats['total_readings']}")
        print(f"   Pothole detections: {stats['pothole_detections']}")
        print(f"   Detection rate: {stats['detection_rate']:.2f}%")
        print("\n✅ System stopped.")


if __name__ == "__main__":
    main()
