import websocket
import json
import threading
import time
import matplotlib.pyplot as plt
import numpy as np

# Data storage for accelerometer values and timestamps
x_values = []
y_values = []
z_values = []
timestamps = []

# Pothole detection parameters
POTHOLE_THRESHOLD = 1.5  # Adjust based on your field tests
detection_points = []

# Start time for relative timestamps
start_time = time.time()


def on_message(ws, message):
    global x_values, y_values, z_values, timestamps
    values = json.loads(message)['values']
    x = values[0]
    y = values[1]
    z = values[2]

    current_time = time.time() - start_time

    x_values.append(x)
    y_values.append(y)
    z_values.append(z)
    timestamps.append(current_time)

    # Simple pothole detection logic
    if len(z_values) > 5:
        # Look for sudden drop followed by rise in z-axis
        if (z_values[-2] - z_values[-5] < -POTHOLE_THRESHOLD and
                z_values[-1] - z_values[-2] > POTHOLE_THRESHOLD):
            detection_points.append((current_time, z))
            print(f"Pothole detected at {current_time:.2f}s!")

    print(f"x = {x}, y = {y}, z = {z} at time = {current_time:.2f}s")


def on_error(ws, error):
    print("error occurred", error)


def on_close(ws, close_code, reason):
    print("connection closed:", reason)


def on_open(ws):
    print("connected")


def connect(url):
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    # Run websocket in a thread to allow stopping
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

    # Run for 30 seconds then stop (adjust based on your test duration)
    time.sleep(30)
    ws.close()

    # Plot the results
    plot_results()


def plot_results():
    plt.figure(figsize=(12, 6))

    # Plot Z-axis acceleration
    plt.plot(timestamps, z_values, 'b-', label='Z-axis Acceleration')

    # Plot detection threshold lines
    if len(timestamps) > 0:
        plt.axhline(y=9.8 - POTHOLE_THRESHOLD, color='r', linestyle='--',
                    label=f'Lower Threshold (-{POTHOLE_THRESHOLD}g)')
        plt.axhline(y=9.8 + POTHOLE_THRESHOLD, color='g', linestyle='--',
                    label=f'Upper Threshold (+{POTHOLE_THRESHOLD}g)')

    # Mark detected potholes
    if detection_points:
        times, values = zip(*detection_points)
        plt.scatter(times, values, color='red', s=100, marker='o',
                    label='Pothole Detected')

    plt.title('Accelerometer Z-axis Readings During Pothole Encounters')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.grid(True)

    # Add annotation explaining the pattern
    plt.annotate('Characteristic pothole pattern:\nSudden drop followed by rise',
                 xy=(0.05, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

    plt.tight_layout()
    plt.savefig('pothole_detection.png')
    plt.show()


# Connect to the websocket and collect data
connect("ws://192.168.1.33:8080/sensor/connect?type=android.sensor.accelerometer")
