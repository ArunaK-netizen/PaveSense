import pandas as pd
import numpy as np

# Define the number of samples for each dataset
NUM_SAMPLES = 1000
TIME_WINDOW = 10  # Number of sensor readings per window

# Function to generate normal road sensor data
def generate_normal_road_data(num_samples, time_window):
    data = []

    for _ in range(num_samples):
        accel_x = np.random.normal(0, 0.2, time_window)  # Small variations
        accel_y = np.random.normal(0, 0.2, time_window)
        accel_z = np.random.normal(9.8, 0.3, time_window)  # Stable gravity component
        gyro_x = np.random.normal(0, 0.05, time_window)
        gyro_y = np.random.normal(0, 0.05, time_window)
        gyro_z = np.random.normal(0, 0.05, time_window)
        speed = np.random.uniform(30, 60, time_window)  # Speed in km/h

        features = [
            np.mean(accel_x), np.mean(accel_y), np.mean(accel_z),
            np.mean(gyro_x), np.mean(gyro_y), np.mean(gyro_z),
            np.std(accel_x), np.std(accel_y), np.std(accel_z),
            np.std(gyro_x), np.std(gyro_y), np.std(gyro_z),
            np.max(accel_x), np.max(accel_y), np.max(accel_z),
            np.max(gyro_x), np.max(gyro_y), np.max(gyro_z),
            np.min(accel_x), np.min(accel_y), np.min(accel_z),
            np.min(gyro_x), np.min(gyro_y), np.min(gyro_z),
            np.mean(speed),  # Average speed
            0  # Label: Normal road
        ]
        data.append(features)

    columns = [
        "mean_accel_x", "mean_accel_y", "mean_accel_z",
        "mean_gyro_x", "mean_gyro_y", "mean_gyro_z",
        "std_accel_x", "std_accel_y", "std_accel_z",
        "std_gyro_x", "std_gyro_y", "std_gyro_z",
        "max_accel_x", "max_accel_y", "max_accel_z",
        "max_gyro_x", "max_gyro_y", "max_gyro_z",
        "min_accel_x", "min_accel_y", "min_accel_z",
        "min_gyro_x", "min_gyro_y", "min_gyro_z",
        "mean_speed", "pothole"
    ]

    return pd.DataFrame(data, columns=columns)

# Function to generate pothole sensor data
def generate_pothole_data(num_samples, time_window):
    data = []

    for _ in range(num_samples):
        # Simulate a sudden spike/drop in sensor readings due to pothole impact
        accel_x = np.concatenate((np.random.normal(0, 0.2, time_window // 2), np.random.normal(-5, 3, time_window // 2)))
        accel_y = np.concatenate((np.random.normal(0, 0.2, time_window // 2), np.random.normal(3, 2, time_window // 2)))
        accel_z = np.concatenate((np.random.normal(9.8, 0.3, time_window // 2), np.random.normal(5, 3, time_window // 2)))
        gyro_x = np.concatenate((np.random.normal(0, 0.05, time_window // 2), np.random.normal(1, 0.5, time_window // 2)))
        gyro_y = np.concatenate((np.random.normal(0, 0.05, time_window // 2), np.random.normal(-1, 0.5, time_window // 2)))
        gyro_z = np.concatenate((np.random.normal(0, 0.05, time_window // 2), np.random.normal(1, 0.5, time_window // 2)))
        speed = np.random.uniform(20, 50, time_window)  # Slightly lower speeds due to potholes

        features = [
            np.mean(accel_x), np.mean(accel_y), np.mean(accel_z),
            np.mean(gyro_x), np.mean(gyro_y), np.mean(gyro_z),
            np.std(accel_x), np.std(accel_y), np.std(accel_z),
            np.std(gyro_x), np.std(gyro_y), np.std(gyro_z),
            np.max(accel_x), np.max(accel_y), np.max(accel_z),
            np.max(gyro_x), np.max(gyro_y), np.max(gyro_z),
            np.min(accel_x), np.min(accel_y), np.min(accel_z),
            np.min(gyro_x), np.min(gyro_y), np.min(gyro_z),
            np.mean(speed),  # Average speed
            1  # Label: Pothole
        ]
        data.append(features)

    columns = [
        "mean_accel_x", "mean_accel_y", "mean_accel_z",
        "mean_gyro_x", "mean_gyro_y", "mean_gyro_z",
        "std_accel_x", "std_accel_y", "std_accel_z",
        "std_gyro_x", "std_gyro_y", "std_gyro_z",
        "max_accel_x", "max_accel_y", "max_accel_z",
        "max_gyro_x", "max_gyro_y", "max_gyro_z",
        "min_accel_x", "min_accel_y", "min_accel_z",
        "min_gyro_x", "min_gyro_y", "min_gyro_z",
        "mean_speed", "pothole"
    ]

    return pd.DataFrame(data, columns=columns)

# Generate the datasets
print("🔄 Generating Normal Road Data...")
normal_road_df = generate_normal_road_data(NUM_SAMPLES, TIME_WINDOW)
print("✅ Normal Road Data Generated.")

print("🔄 Generating Pothole Data...")
pothole_df = generate_pothole_data(NUM_SAMPLES, TIME_WINDOW)
print("✅ Pothole Data Generated.")

# Save datasets
normal_road_df.to_csv("normal_road_data.csv", index=False)
pothole_df.to_csv("pothole_data.csv", index=False)

print("📂 Datasets saved: normal_road_data.csv & pothole_data.csv")
