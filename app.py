import websocket
import json
import pickle
import numpy as np
import pandas as pd
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from geopy.distance import geodesic

app = Flask(__name__)
socketio = SocketIO(app)

# Load trained model
print("🔄 Loading trained model...")
with open('model.p', 'rb') as file:
    pothole_data = pickle.load(file)

classifier = pothole_data['classifier']
feature_names = pothole_data.get("feature_names", None)

threshold = 0  # Adjust threshold for pothole detection
pothole_locations = []
sensor_data_buffer = {"accelerometer": [], "gyroscope": [], "speed": []}
lat, long = "13.0827", "80.2707"  # Default to Chennai

print("✅ Model loaded successfully!")

# WebSocket Sensor Event Handlers
def on_accelerometer_event(values, timestamp):
    print(f"📡 Accelerometer Data: {values} | Timestamp: {timestamp}")
    sensor_data_buffer["accelerometer"].append(values)
    process_sensor_data()

def on_gyroscope_event(values, timestamp):
    print(f"📡 Gyroscope Data: {values} | Timestamp: {timestamp}")
    sensor_data_buffer["gyroscope"].append(values)
    process_sensor_data()

def on_speed_event(values, timestamp):
    print(f"📡 Speed Data: {values} | Timestamp: {timestamp}")
    sensor_data_buffer["speed"].append(values)
    process_sensor_data()

# Sensor WebSocket Handler
class Sensor:
    def __init__(self, address, sensor_type, on_sensor_event):
        self.address = address
        self.sensor_type = sensor_type
        self.on_sensor_event = on_sensor_event

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            values = data["values"]
            timestamp = data["timestamp"]
            self.on_sensor_event(values=values, timestamp=timestamp)
        except Exception as e:
            print(f"❌ Error processing {self.sensor_type} message:", e)

    def on_error(self, ws, error):
        print(f"❌ Error in {self.sensor_type} WebSocket:", error)

    def on_close(self, ws, close_code, reason):
        print(f"🔌 Connection closed for {self.sensor_type}: {reason}")

    def on_open(self, ws):
        print(f"✅ Connected to {self.sensor_type} WebSocket!")

    def make_websocket_connection(self):
        ws_url = f"ws://{self.address}/sensor/connect?type={self.sensor_type}"
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        ws.run_forever()

    def connect(self):
        thread = threading.Thread(target=self.make_websocket_connection)
        thread.start()

# Process Incoming Sensor Data
def process_sensor_data():
    """ Processes sensor data and checks for potholes. """
    global lat, long

    if len(sensor_data_buffer["accelerometer"]) == 0 or len(sensor_data_buffer["gyroscope"]) == 0:
        return  # Wait for enough sensor data before processing

    # Convert sensor data into features for the model
    input_features = preprocess_sensor_data(sensor_data_buffer)

    # Get model decision score
    decision_scores = classifier.decision_function(input_features)
    pothole_detected = decision_scores[0] > threshold

    print(f"🧠 Model Decision Score: {decision_scores[0]} | 🚧 Pothole Detected: {pothole_detected}")

    if pothole_detected:
        print("🚧 POTHOLE DETECTED! Using Chennai GPS Data.")
        pothole_locations.append((lat, long))
        socketio.emit("update_map", {"latitude": lat, "longitude": long})
    else:
        print("✅ No pothole detected.")

# Convert Raw Sensor Data to Features
def preprocess_sensor_data(buffer):
    """
    Convert raw sensor data into statistical features.
    """
    print("🛠️ Preprocessing sensor data...")

    accel_data = np.array(buffer["accelerometer"])
    gyro_data = np.array(buffer["gyroscope"])
    speed_data = np.array(buffer["speed"]) if buffer["speed"] else np.array([0])

    if accel_data.size == 0 or gyro_data.size == 0:
        return np.zeros((1, 24))  # Ensure proper feature format

    # Compute statistical features
    features = [
        np.max(accel_data[:, 0]), np.max(accel_data[:, 1]), np.max(accel_data[:, 2]),  # Max Accel
        np.max(gyro_data[:, 0]), np.max(gyro_data[:, 1]), np.max(gyro_data[:, 2]),  # Max Gyro
        np.min(accel_data[:, 0]), np.min(accel_data[:, 1]), np.min(accel_data[:, 2]),  # Min Accel
        np.min(gyro_data[:, 0]), np.min(gyro_data[:, 1]), np.min(gyro_data[:, 2]),  # Min Gyro
        np.mean(accel_data[:, 0]), np.mean(accel_data[:, 1]), np.mean(accel_data[:, 2]),  # Mean Accel
        np.mean(gyro_data[:, 0]), np.mean(gyro_data[:, 1]), np.mean(gyro_data[:, 2]),  # Mean Gyro
        np.std(accel_data[:, 0]), np.std(accel_data[:, 1]), np.std(accel_data[:, 2]),  # Std Accel
        np.std(gyro_data[:, 0]), np.std(gyro_data[:, 1]), np.std(gyro_data[:, 2]),  # Std Gyro
    ]

    features = np.array(features).reshape(1, -1)

    if feature_names:
        features_df = pd.DataFrame(features, columns=feature_names)
    else:
        features_df = pd.DataFrame(features)

    print("📊 Preprocessed features:", features_df.values.tolist())
    return features_df

# Flask Routes
@app.route("/")
def index():
    print("🖥️ Rendering index.html")
    return render_template("index.html")

# Start WebSocket Connections for Sensors
if __name__ == "__main__":
    sensor_address = "192.168.1.36:8080"

    print("🛠️ Starting WebSocket connections for sensors...")
    Sensor(sensor_address, "android.sensor.accelerometer", on_accelerometer_event).connect()
    Sensor(sensor_address, "android.sensor.gyroscope", on_gyroscope_event).connect()
    Sensor(sensor_address, "android.sensor.speed", on_speed_event).connect()

    print("🚀 Starting Flask server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
