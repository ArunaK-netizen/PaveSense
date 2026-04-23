"""
PaveSense Feature Engineering Pipeline
=======================================
Transforms 6 raw sensor values into 13 physics-aware features
that help distinguish potholes from speed bumps, phone drops, etc.

Raw inputs (6):
    accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

Engineered outputs (13):
    0-2:  accel_x, accel_y, accel_z          (raw accelerometer)
    3-5:  gyro_x, gyro_y, gyro_z             (raw gyroscope)
    6:    accel_magnitude                     (total acceleration)
    7:    gyro_magnitude                      (total angular velocity)
    8:    jerk_z                              (d(accel_z)/dt — impact sharpness)
    9:    accel_z_detrended                   (gravity-removed vertical accel)
    10:   vertical_asymmetry                  (drop vs rise ratio — pothole signature)
    11:   spectral_energy_band                (energy in 5-25Hz pothole band)
    12:   freefall_score                      (how close to freefall — phone drop detector)
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


# === Constants ===
GRAVITY = 9.81
SAMPLING_RATE = 50  # Hz
FREEFALL_THRESHOLD = 2.0  # m/s² — below this = likely freefall
POTHOLE_FREQ_LOW = 5.0    # Hz — lower bound of pothole frequency signature
POTHOLE_FREQ_HIGH = 25.0  # Hz — upper bound


def compute_features_batch(raw_sequences, sampling_rate=SAMPLING_RATE):
    """
    Compute engineered features for a batch of sequences.

    Args:
        raw_sequences: np.ndarray of shape (N, seq_len, 6)
            Columns: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        sampling_rate: sensor sampling rate in Hz

    Returns:
        np.ndarray of shape (N, seq_len, 13) — engineered features
    """
    N, seq_len, _ = raw_sequences.shape
    features = np.zeros((N, seq_len, 13), dtype=np.float32)

    for i in range(N):
        features[i] = compute_features_single(raw_sequences[i], sampling_rate)

    return features


def compute_features_single(raw_sequence, sampling_rate=SAMPLING_RATE):
    """
    Compute engineered features for a single sequence.

    Args:
        raw_sequence: np.ndarray of shape (seq_len, 6)

    Returns:
        np.ndarray of shape (seq_len, 13)
    """
    seq_len = raw_sequence.shape[0]
    features = np.zeros((seq_len, 13), dtype=np.float32)

    accel_x = raw_sequence[:, 0]
    accel_y = raw_sequence[:, 1]
    accel_z = raw_sequence[:, 2]
    gyro_x = raw_sequence[:, 3]
    gyro_y = raw_sequence[:, 4]
    gyro_z = raw_sequence[:, 5]

    # --- Feature 0-5: Raw sensor values ---
    features[:, 0] = accel_x
    features[:, 1] = accel_y
    features[:, 2] = accel_z
    features[:, 3] = gyro_x
    features[:, 4] = gyro_y
    features[:, 5] = gyro_z

    # --- Feature 6: Accelerometer magnitude ---
    accel_mag = np.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)
    features[:, 6] = accel_mag

    # --- Feature 7: Gyroscope magnitude ---
    gyro_mag = np.sqrt(gyro_x ** 2 + gyro_y ** 2 + gyro_z ** 2)
    features[:, 7] = gyro_mag

    # --- Feature 8: Jerk Z (d(accel_z)/dt) ---
    # Jerk = rate of change of acceleration — potholes produce sharp jerks
    jerk_z = np.zeros(seq_len, dtype=np.float32)
    jerk_z[1:] = np.diff(accel_z) * sampling_rate  # multiply by sample rate for m/s³
    jerk_z[0] = jerk_z[1]  # pad first value
    features[:, 8] = jerk_z

    # --- Feature 9: Detrended accel_z (gravity removed) ---
    # Simple gravity removal: subtract median (assumes mostly upright)
    gravity_estimate = np.median(accel_z)
    features[:, 9] = accel_z - gravity_estimate

    # --- Feature 10: Vertical asymmetry ---
    # Potholes: sharp drop (negative) followed by gradual recovery (positive)
    # Speed bumps: more symmetric (both positive humps)
    # Compute rolling asymmetry using a short window
    asymmetry = _compute_vertical_asymmetry(accel_z - gravity_estimate, window=10)
    features[:, 10] = asymmetry

    # --- Feature 11: Spectral energy in pothole frequency band ---
    spectral_energy = _compute_spectral_energy_rolling(
        accel_z, sampling_rate, POTHOLE_FREQ_LOW, POTHOLE_FREQ_HIGH, window=20
    )
    features[:, 11] = spectral_energy

    # --- Feature 12: Freefall score ---
    # Closer to 0 = more likely freefall (phone drop)
    # Normalized: 1.0 = definite freefall, 0.0 = normal
    freefall_score = np.clip(1.0 - (accel_mag / GRAVITY), 0.0, 1.0)
    features[:, 12] = freefall_score

    return features


def _compute_vertical_asymmetry(detrended_z, window=10):
    """
    Compute asymmetry ratio: how much the negative peak exceeds the positive peak.

    Potholes produce asymmetric patterns (sharp dip).
    Speed bumps produce symmetric patterns (gradual rise/fall).
    
    Returns values in [-1, 1]:
        Negative → symmetric or no event
        Positive → asymmetric drop (pothole-like)
    """
    seq_len = len(detrended_z)
    asymmetry = np.zeros(seq_len, dtype=np.float32)

    half_window = window // 2
    for i in range(half_window, seq_len - half_window):
        window_data = detrended_z[i - half_window:i + half_window]
        pos_peak = np.max(window_data)
        neg_peak = np.abs(np.min(window_data))

        denom = pos_peak + neg_peak
        if denom > 0.5:  # Only compute if there's meaningful signal
            # Positive asymmetry = negative peak dominates (drop = pothole)
            asymmetry[i] = (neg_peak - pos_peak) / denom
        else:
            asymmetry[i] = 0.0

    return asymmetry


def _compute_spectral_energy_rolling(accel_z, sampling_rate, freq_low, freq_high, window=20):
    """
    Compute rolling spectral energy in the pothole frequency band.

    Potholes create characteristic energy bursts in 5-25Hz range.
    Normal driving has energy mainly below 5Hz.
    """
    seq_len = len(accel_z)
    energy = np.zeros(seq_len, dtype=np.float32)

    half_window = window // 2

    for i in range(half_window, seq_len - half_window):
        segment = accel_z[i - half_window:i + half_window]

        # Apply Hann window to reduce spectral leakage
        windowed = segment * np.hanning(len(segment))

        # FFT
        spectrum = np.abs(rfft(windowed))
        freqs = rfftfreq(len(segment), d=1.0 / sampling_rate)

        # Energy in pothole band
        band_mask = (freqs >= freq_low) & (freqs <= freq_high)
        band_energy = np.sum(spectrum[band_mask] ** 2)
        total_energy = np.sum(spectrum ** 2) + 1e-8

        energy[i] = band_energy / total_energy  # Normalized ratio

    return energy


# === Pre-filters (applied before ML inference) ===

def detect_freefall(accel_magnitude_sequence, threshold=FREEFALL_THRESHOLD, min_duration_ms=50,
                    sampling_rate=SAMPLING_RATE):
    """
    Detect if a freefall event occurred in the sequence.
    Phone drops show near-zero acceleration before impact.

    Args:
        accel_magnitude_sequence: 1D array of acceleration magnitudes
        threshold: magnitude below which is considered freefall
        min_duration_ms: minimum freefall duration in milliseconds

    Returns:
        bool: True if freefall detected (likely phone drop, NOT a pothole)
    """
    min_samples = int(min_duration_ms * sampling_rate / 1000)
    below_threshold = accel_magnitude_sequence < threshold

    # Find consecutive runs below threshold
    run_length = 0
    for val in below_threshold:
        if val:
            run_length += 1
            if run_length >= min_samples:
                return True
        else:
            run_length = 0

    return False


def should_suppress_detection(speed_kmh, min_speed=5.0):
    """
    Suppress detections at very low speeds.
    At walking speeds or when stopped, sensor spikes are noise.
    """
    return speed_kmh < min_speed


# === Normalization ===

class FeatureNormalizer:
    """Online-updatable feature normalizer using running statistics."""

    def __init__(self, n_features=13):
        self.n_features = n_features
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.var = np.ones(n_features, dtype=np.float32)
        self.count = 0
        self.fitted = False

    def fit(self, data):
        """
        Fit normalizer on training data.

        Args:
            data: np.ndarray of shape (N, seq_len, n_features) or (N*seq_len, n_features)
        """
        if data.ndim == 3:
            data = data.reshape(-1, self.n_features)
        self.mean = np.mean(data, axis=0).astype(np.float32)
        self.var = np.var(data, axis=0).astype(np.float32) + 1e-8
        self.count = data.shape[0]
        self.fitted = True

    def transform(self, data):
        """Normalize data using fitted statistics."""
        if not self.fitted:
            return data
        return (data - self.mean) / np.sqrt(self.var)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = int(data['count'])
        self.fitted = True
