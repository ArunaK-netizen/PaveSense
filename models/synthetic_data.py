"""
PaveSense Synthetic Data Generator
====================================
Generates realistic synthetic sensor data for 5 event types,
based on physics models of how each event affects accelerometer/gyroscope.

Classes:
    0: normal       — smooth driving, stationary, gentle road vibration
    1: pothole      — sharp asymmetric Z-drop then rebound
    2: speed_bump   — symmetric gradual Z-rise then return
    3: phone_drop   — freefall (acc_mag → 0) then chaotic impact
    4: disturbance  — braking, turning, road joints, rough surface
"""

import numpy as np
from typing import Tuple, List


# === Sensor noise model ===
ACCEL_NOISE_STD = 0.15   # m/s² — typical smartphone accelerometer noise
GYRO_NOISE_STD = 0.02    # rad/s — typical smartphone gyroscope noise
GRAVITY = 9.81
SAMPLING_RATE = 50       # Hz
SEQUENCE_LENGTH = 100    # 2 seconds at 50Hz


def generate_dataset(
    n_samples_per_class: int = 2000,
    sequence_length: int = SEQUENCE_LENGTH,
    sampling_rate: int = SAMPLING_RATE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced synthetic dataset.

    Returns:
        data: (N*5, seq_len, 6) — raw sensor readings
        labels: (N*5,) — class labels 0-4
    """
    rng = np.random.RandomState(seed)

    all_data = []
    all_labels = []

    generators = [
        (0, generate_normal),
        (1, generate_pothole),
        (2, generate_speed_bump),
        (3, generate_phone_drop),
        (4, generate_disturbance),
    ]

    for label, gen_fn in generators:
        for i in range(n_samples_per_class):
            sample = gen_fn(sequence_length, sampling_rate, rng)
            all_data.append(sample)
            all_labels.append(label)

    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(len(labels))
    return data[idx], labels[idx]


def _add_baseline_noise(sequence, rng):
    """Add realistic sensor noise to a sequence."""
    seq_len = sequence.shape[0]
    sequence[:, :3] += rng.normal(0, ACCEL_NOISE_STD, (seq_len, 3))
    sequence[:, 3:] += rng.normal(0, GYRO_NOISE_STD, (seq_len, 3))
    return sequence


def _add_driving_vibration(sequence, rng, intensity=1.0):
    """Add low-frequency road vibration (1-5Hz)."""
    seq_len = sequence.shape[0]
    t = np.arange(seq_len) / SAMPLING_RATE

    for axis in range(3):
        freq = rng.uniform(1.0, 5.0)
        amplitude = rng.uniform(0.05, 0.3) * intensity
        phase = rng.uniform(0, 2 * np.pi)
        sequence[:, axis] += amplitude * np.sin(2 * np.pi * freq * t + phase)

    return sequence


# ==========================================
# Event Generators
# ==========================================

def generate_normal(seq_len, sampling_rate, rng):
    """Normal driving: gravity on Z + road vibration + noise."""
    sequence = np.zeros((seq_len, 6), dtype=np.float32)

    # Gravity on Z-axis (phone roughly upright in mount)
    # Add slight tilt variation
    tilt_angle = rng.uniform(-0.15, 0.15)  # radians
    sequence[:, 2] = GRAVITY * np.cos(tilt_angle)
    sequence[:, 1] = GRAVITY * np.sin(tilt_angle) * rng.choice([-1, 1])

    sequence = _add_driving_vibration(sequence, rng, intensity=rng.uniform(0.3, 1.0))
    sequence = _add_baseline_noise(sequence, rng)

    return sequence


def generate_pothole(seq_len, sampling_rate, rng):
    """
    Pothole: Sharp asymmetric Z-axis event.

    Physics:
    1. Wheel enters hole → vehicle drops → Z-accel decreases sharply
    2. Wheel hits bottom/exit → vehicle rebounds → Z-accel spikes up
    3. The drop is sharper than the recovery (asymmetric)
    4. Brief pitch change in gyroscope
    5. Total duration: 100-300ms (5-15 samples at 50Hz)
    """
    sequence = np.zeros((seq_len, 6), dtype=np.float32)

    # Baseline: gravity + driving
    tilt = rng.uniform(-0.1, 0.1)
    sequence[:, 2] = GRAVITY * np.cos(tilt)
    sequence[:, 1] = GRAVITY * np.sin(tilt)

    # Pothole event parameters
    event_center = rng.randint(seq_len // 4, 3 * seq_len // 4)
    severity = rng.uniform(0.5, 1.0)  # 0.5=mild, 1.0=severe

    # Drop magnitude (3-12 m/s² below gravity)
    drop_magnitude = rng.uniform(3.0, 12.0) * severity
    # Rise magnitude (50-80% of drop — asymmetric!)
    rise_magnitude = drop_magnitude * rng.uniform(0.5, 0.8)

    # Duration in samples
    drop_duration = rng.randint(3, 8)  # Sharp drop: 60-160ms
    rise_duration = rng.randint(5, 12)  # Slower recovery: 100-240ms

    # Generate asymmetric impulse
    t_drop = np.linspace(0, np.pi, drop_duration)
    drop_profile = -drop_magnitude * np.sin(t_drop)

    t_rise = np.linspace(0, np.pi, rise_duration)
    rise_profile = rise_magnitude * np.sin(t_rise)

    # Apply to Z-axis
    drop_start = max(0, event_center - drop_duration)
    drop_end = min(seq_len, drop_start + drop_duration)
    actual_drop = drop_end - drop_start
    sequence[drop_start:drop_end, 2] += drop_profile[:actual_drop]

    rise_start = drop_end
    rise_end = min(seq_len, rise_start + rise_duration)
    actual_rise = rise_end - rise_start
    sequence[rise_start:rise_end, 2] += rise_profile[:actual_rise]

    # Slight X/Y disturbance (lateral roll from pothole)
    x_disturbance = rng.uniform(0.5, 2.0) * severity
    sequence[drop_start:rise_end, 0] += x_disturbance * rng.choice([-1, 1]) * \
        np.sin(np.linspace(0, np.pi, rise_end - drop_start))

    # Gyroscope: brief pitch spike (X-axis rotation)
    gyro_magnitude = rng.uniform(0.3, 2.0) * severity
    gyro_duration = drop_duration + rise_duration
    gyro_start = max(0, event_center - 2)
    gyro_end = min(seq_len, gyro_start + gyro_duration)
    actual_gyro = gyro_end - gyro_start
    t_gyro = np.linspace(0, np.pi, actual_gyro)
    sequence[gyro_start:gyro_end, 3] += gyro_magnitude * np.sin(t_gyro)  # Pitch (X-axis rotation)
    # Small yaw component
    sequence[gyro_start:gyro_end, 5] += rng.uniform(-0.3, 0.3) * severity * np.sin(t_gyro)

    sequence = _add_driving_vibration(sequence, rng, intensity=rng.uniform(0.5, 1.0))
    sequence = _add_baseline_noise(sequence, rng)

    return sequence


def generate_speed_bump(seq_len, sampling_rate, rng):
    """
    Speed bump: Gradual, symmetric Z-axis rise and fall.

    Key differences from pothole:
    1. Z-accel goes UP first (vehicle goes over bump), not DOWN
    2. Very symmetric profile (rise ≈ fall)
    3. Longer duration: 300-800ms (15-40 samples)
    4. Preceded by deceleration (driver slows down)
    5. Sustained gyro pitch change (not brief)
    """
    sequence = np.zeros((seq_len, 6), dtype=np.float32)

    # Baseline
    tilt = rng.uniform(-0.1, 0.1)
    sequence[:, 2] = GRAVITY * np.cos(tilt)
    sequence[:, 1] = GRAVITY * np.sin(tilt)

    event_center = rng.randint(seq_len // 3, 2 * seq_len // 3)
    severity = rng.uniform(0.4, 1.0)

    # Bump magnitude (2-6 m/s² — less extreme than pothole)
    bump_magnitude = rng.uniform(2.0, 6.0) * severity

    # Symmetric duration: both sides roughly equal
    half_duration = rng.randint(8, 20)  # 160-400ms per side

    # Symmetric Gaussian-like profile (smooth hump)
    total_duration = half_duration * 2
    t = np.linspace(-np.pi, np.pi, total_duration)
    # Positive hump (goes UP) - key difference from pothole
    bump_profile = bump_magnitude * np.cos(t) * 0.5 + bump_magnitude * 0.5

    bump_start = max(0, event_center - half_duration)
    bump_end = min(seq_len, bump_start + total_duration)
    actual_len = bump_end - bump_start
    sequence[bump_start:bump_end, 2] += bump_profile[:actual_len]

    # Deceleration before bump (driver slows down) — Y-axis
    decel_start = max(0, bump_start - rng.randint(10, 20))
    decel_magnitude = rng.uniform(1.0, 3.0) * severity
    decel_len = bump_start - decel_start
    if decel_len > 0:
        sequence[decel_start:bump_start, 1] -= decel_magnitude * \
            np.linspace(0, 1, decel_len)

    # Sustained gyro pitch (longer than pothole)
    gyro_magnitude = rng.uniform(0.5, 1.5) * severity
    gyro_t = np.linspace(0, 2 * np.pi, actual_len)
    sequence[bump_start:bump_end, 3] += gyro_magnitude * np.sin(gyro_t)

    sequence = _add_driving_vibration(sequence, rng, intensity=rng.uniform(0.3, 0.8))
    sequence = _add_baseline_noise(sequence, rng)

    return sequence


def generate_phone_drop(seq_len, sampling_rate, rng):
    """
    Phone drop: Freefall → chaotic impact → settling.

    Key signature:
    1. Acceleration magnitude drops to near ZERO (freefall = no gravity felt)
    2. Followed by large chaotic spike on ALL axes (impact)
    3. Gyroscope goes wild in ALL axes (tumbling)
    4. Very different from pothole (which only affects Z primarily)
    """
    sequence = np.zeros((seq_len, 6), dtype=np.float32)

    # Baseline: phone in hand / on dash
    sequence[:, 2] = GRAVITY
    sequence = _add_baseline_noise(sequence, rng)

    # Drop event
    drop_start = rng.randint(seq_len // 4, seq_len // 2)

    # Phase 1: Freefall (50-200ms → 3-10 samples)
    freefall_duration = rng.randint(3, 10)
    freefall_end = min(seq_len, drop_start + freefall_duration)

    # During freefall: ALL accelerometer axes → ~0 (key signature!)
    for axis in range(3):
        sequence[drop_start:freefall_end, axis] = rng.normal(0, 0.3, freefall_end - drop_start)

    # Phase 2: Impact (1-3 samples → very brief)
    impact_start = freefall_end
    impact_duration = rng.randint(1, 4)
    impact_end = min(seq_len, impact_start + impact_duration)

    # Impact: LARGE spike on ALL axes (chaotic, not directional)
    for axis in range(3):
        impact_magnitude = rng.uniform(15.0, 40.0)
        sequence[impact_start:impact_end, axis] = rng.normal(0, impact_magnitude,
                                                              impact_end - impact_start)

    # Phase 3: Bouncing/settling (5-15 samples)
    settle_start = impact_end
    settle_duration = rng.randint(5, 15)
    settle_end = min(seq_len, settle_start + settle_duration)

    for axis in range(3):
        t = np.arange(settle_end - settle_start)
        damping = np.exp(-t * 0.3)
        sequence[settle_start:settle_end, axis] = (
            rng.uniform(3.0, 8.0) * damping * np.sin(rng.uniform(5, 15) * t * 2 * np.pi / sampling_rate)
        )

    # After settling: phone on new surface (different gravity projection)
    new_tilt = rng.uniform(-1.0, 1.0)  # Could be on its side
    if settle_end < seq_len:
        sequence[settle_end:, 0] = GRAVITY * np.sin(new_tilt) * rng.uniform(-1, 1)
        sequence[settle_end:, 2] = GRAVITY * np.cos(new_tilt)

    # Gyroscope: wild rotation during freefall + impact (ALL axes)
    for axis in range(3, 6):
        # Freefall: moderate tumbling
        sequence[drop_start:freefall_end, axis] = rng.uniform(-5.0, 5.0)
        # Impact: extreme rotation
        if impact_end > impact_start:
            sequence[impact_start:impact_end, axis] = rng.normal(0, 8.0,
                                                                  impact_end - impact_start)
        # Settling
        if settle_end > settle_start:
            t = np.arange(settle_end - settle_start)
            damping = np.exp(-t * 0.4)
            sequence[settle_start:settle_end, axis] = (
                rng.uniform(2.0, 5.0) * damping * rng.choice([-1, 1])
            )

    return sequence


def generate_disturbance(seq_len, sampling_rate, rng):
    """
    General disturbance: braking, turning, road joints, rough road, door slam.
    
    Key characteristics:
    1. Braking: sustained Y-axis shift (NOT impulsive)
    2. Turning: sustained X-axis shift + Z-gyro (yaw)
    3. Road joints: small periodic Z bumps (regular pattern)
    4. Rough road: high-frequency Z noise (sustained, not impulsive)
    """
    sequence = np.zeros((seq_len, 6), dtype=np.float32)

    # Baseline
    tilt = rng.uniform(-0.1, 0.1)
    sequence[:, 2] = GRAVITY * np.cos(tilt)
    sequence[:, 1] = GRAVITY * np.sin(tilt)

    disturbance_type = rng.choice(['braking', 'turning', 'road_joint', 'rough_road'])

    if disturbance_type == 'braking':
        # Sustained forward deceleration
        brake_start = rng.randint(seq_len // 4, seq_len // 2)
        brake_duration = rng.randint(20, 50)  # 400ms-1s
        brake_end = min(seq_len, brake_start + brake_duration)
        brake_magnitude = rng.uniform(2.0, 6.0)

        t = np.linspace(0, np.pi, brake_end - brake_start)
        sequence[brake_start:brake_end, 1] -= brake_magnitude * np.sin(t)
        # Slight pitch from braking
        sequence[brake_start:brake_end, 3] += rng.uniform(0.1, 0.5) * np.sin(t)

    elif disturbance_type == 'turning':
        # Sustained lateral acceleration + yaw
        turn_start = rng.randint(seq_len // 4, seq_len // 2)
        turn_duration = rng.randint(25, 60)
        turn_end = min(seq_len, turn_start + turn_duration)
        turn_magnitude = rng.uniform(2.0, 5.0)
        direction = rng.choice([-1, 1])

        t = np.linspace(0, np.pi, turn_end - turn_start)
        sequence[turn_start:turn_end, 0] += direction * turn_magnitude * np.sin(t)
        # Yaw (Z-axis gyro)
        sequence[turn_start:turn_end, 5] += direction * rng.uniform(0.5, 2.0) * np.sin(t)

    elif disturbance_type == 'road_joint':
        # Small periodic Z bumps (regular spacing like expansion joints)
        n_joints = rng.randint(2, 5)
        joint_spacing = seq_len // (n_joints + 1)
        bump_magnitude = rng.uniform(1.0, 3.0)

        for j in range(n_joints):
            pos = (j + 1) * joint_spacing
            if pos + 3 < seq_len:
                t = np.linspace(0, np.pi, 3)
                sequence[pos:pos + 3, 2] += bump_magnitude * np.sin(t)

    elif disturbance_type == 'rough_road':
        # Sustained high-frequency Z noise
        rough_start = rng.randint(0, seq_len // 3)
        rough_duration = rng.randint(30, seq_len - rough_start)
        rough_end = min(seq_len, rough_start + rough_duration)

        t = np.arange(rough_end - rough_start) / sampling_rate
        for freq in rng.uniform(10, 30, size=3):
            amp = rng.uniform(0.5, 2.0)
            sequence[rough_start:rough_end, 2] += amp * np.sin(2 * np.pi * freq * t)

    sequence = _add_driving_vibration(sequence, rng, intensity=rng.uniform(0.3, 1.0))
    sequence = _add_baseline_noise(sequence, rng)

    return sequence


# ==========================================
# Data Augmentation
# ==========================================

def augment_sample(sample, rng):
    """Apply random augmentation to a single sample."""
    augmented = sample.copy()

    # Random noise injection (50% chance)
    if rng.random() < 0.5:
        noise_scale = rng.uniform(0.05, 0.2)
        augmented[:, :3] += rng.normal(0, noise_scale * ACCEL_NOISE_STD, augmented[:, :3].shape)
        augmented[:, 3:] += rng.normal(0, noise_scale * GYRO_NOISE_STD, augmented[:, 3:].shape)

    # Time warping (30% chance) — stretch/compress by small factor
    if rng.random() < 0.3:
        warp_factor = rng.uniform(0.9, 1.1)
        seq_len = augmented.shape[0]
        new_len = int(seq_len * warp_factor)
        old_indices = np.linspace(0, seq_len - 1, new_len)
        new_indices = np.linspace(0, seq_len - 1, seq_len)
        for col in range(augmented.shape[1]):
            augmented[:, col] = np.interp(new_indices, old_indices,
                                          np.interp(old_indices, np.arange(seq_len), augmented[:, col]))

    # Amplitude scaling (40% chance) — simulate different vehicle suspensions
    if rng.random() < 0.4:
        scale = rng.uniform(0.8, 1.3)
        gravity_removed = augmented[:, 2] - GRAVITY
        augmented[:, 2] = gravity_removed * scale + GRAVITY
        augmented[:, 0:2] *= scale

    # Gravity rotation (20% chance) — simulate different phone orientations
    if rng.random() < 0.2:
        angle = rng.uniform(-0.3, 0.3)  # Small tilt change
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        new_y = augmented[:, 1] * cos_a - augmented[:, 2] * sin_a
        new_z = augmented[:, 1] * sin_a + augmented[:, 2] * cos_a
        augmented[:, 1] = new_y
        augmented[:, 2] = new_z

    return augmented


def augment_dataset(data, labels, augmentation_factor=2, seed=42):
    """
    Augment the entire dataset.

    Args:
        data: (N, seq_len, 6)
        labels: (N,)
        augmentation_factor: how many augmented copies per sample

    Returns:
        augmented_data, augmented_labels (includes originals)
    """
    rng = np.random.RandomState(seed)
    all_data = [data]
    all_labels = [labels]

    for _ in range(augmentation_factor):
        aug_data = np.array([augment_sample(sample, rng) for sample in data])
        all_data.append(aug_data)
        all_labels.append(labels.copy())

    return np.concatenate(all_data), np.concatenate(all_labels)


if __name__ == '__main__':
    print("🔄 Generating synthetic dataset...")
    data, labels = generate_dataset(n_samples_per_class=2000, seed=42)
    print(f"✅ Generated {len(labels)} samples")
    print(f"   Shape: {data.shape}")
    print(f"   Classes: {np.bincount(labels)}")
    print(f"   Labels: normal={np.sum(labels==0)}, pothole={np.sum(labels==1)}, "
          f"speed_bump={np.sum(labels==2)}, phone_drop={np.sum(labels==3)}, "
          f"disturbance={np.sum(labels==4)}")
