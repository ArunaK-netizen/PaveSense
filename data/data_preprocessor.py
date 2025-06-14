import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.stats import zscore


class SensorDataPreprocessor:
    def __init__(self, sampling_rate=50):
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
