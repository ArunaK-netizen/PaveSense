import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.stats import zscore


class SensorDataPreprocessor:
    def __init__(self, sampling_rate=50):  # 50 Hz typical for mobile sensors
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()

    def remove_outliers(self, data, threshold=3):
        """Remove outliers using Z-score"""
        z_scores = np.abs(zscore(data, axis=0))
        return data[~(z_scores > threshold).any(axis=1)]

    def apply_filters(self, data, low_cutoff=0.5, high_cutoff=20):
        """Apply bandpass filter to remove noise"""
        nyquist = self.sampling_rate / 2
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist

        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')

        # Apply filter to each column
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])

        return filtered_data

    def calculate_magnitude(self, accel_data, gyro_data):
        """Calculate magnitude of accelerometer and gyroscope"""
        accel_magnitude = np.sqrt(np.sum(accel_data ** 2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(gyro_data ** 2, axis=1))

        return accel_magnitude, gyro_magnitude

    def extract_features(self, data, window_size=10):
        """Extract statistical features from sensor data"""
        features = []

        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]

            # Statistical features
            mean_vals = np.mean(window, axis=0)
            std_vals = np.std(window, axis=0)
            max_vals = np.max(window, axis=0)
            min_vals = np.min(window, axis=0)
            rms_vals = np.sqrt(np.mean(window ** 2, axis=0))

            # Combine all features
            window_features = np.concatenate([
                mean_vals, std_vals, max_vals, min_vals, rms_vals
            ])

            features.append(window_features)

        return np.array(features)

    def normalize_data(self, data, fit_scaler=True):
        """Normalize data using StandardScaler"""
        if fit_scaler:
            return self.scaler.fit_transform(data)
        else:
            return self.scaler.transform(data)

    def preprocess_sensor_data(self, accelerometer_data, gyroscope_data,
                               apply_filtering=True, normalize=True):
        """Complete preprocessing pipeline"""
        # Combine accelerometer and gyroscope data
        combined_data = np.hstack([accelerometer_data, gyroscope_data])

        # Remove outliers
        combined_data = self.remove_outliers(combined_data)

        # Apply filtering
        if apply_filtering:
            combined_data = self.apply_filters(combined_data)

        # Normalize data
        if normalize:
            combined_data = self.normalize_data(combined_data)

        return combined_data

    def create_labels_from_events(self, data_length, pothole_events,
                                  event_window=25):  # 0.5 seconds at 50Hz
        """Create labels from pothole events"""
        labels = np.zeros(data_length)

        for event_idx in pothole_events:
            start_idx = max(0, event_idx - event_window // 2)
            end_idx = min(data_length, event_idx + event_window // 2)
            labels[start_idx:end_idx] = 1

        return labels
