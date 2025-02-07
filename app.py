import websocket
import json
import pickle
import numpy as np
import pandas as pd
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from geopy.distance import geodesic
import requests

app = Flask(__name__)
socketio = SocketIO(app)

# Load trained model
print("🔄 Loading trained model...")
with open('model.p', 'rb') as file:
    pothole_data = pickle.load(file)

classifier = pothole_data['classifier']
threshold = 0  # Adjust threshold for pothole detection
feature_names = pothole_data.get("feature_names", None)

pothole_locations = []
lat, long = "13.0827", "80.2707"  # Default to Chennai

print("✅ Model loaded successfully!")


# Sensor Event Handlers
def on_accelerometer_event(values, timestamp):
    print(f"📡 Accelerometer Data: {values} | Timestamp: {timestamp}")
    process_sensor_data("accelerometer", values)


def on_gyroscope_event(values, timestamp):
    print(f"📡 Gyroscope Data: {values} | Timestamp: {timestamp}")
    process_sensor_data("gyroscope", values)


def on_magnetic_field_event(values, timestamp):
    print(f"📡 Magnetic Field Data: {values} | Timestamp: {timestamp}")
    process_sensor_data("magnetic_field", values)


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
def process_sensor_data(sensor_type, values):
    """ Processes sensor data and checks for potholes. """
    global lat, long

    print(f"⚙️ Processing {sensor_type} data: {values}")

    # Ensure all sensor data is received before making predictions
    if sensor_type not in ["accelerometer", "gyroscope"]:
        print("⚠️ Ignoring unsupported sensor type:", sensor_type)
        return

    # Convert sensor data into features for the model
    input_features = preprocess_sensor_data(values)

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
def preprocess_sensor_data(raw_values):
    """
    Convert raw sensor data into statistical features (mean, std, max, min).
    Ensures the model receives exactly 4 features.
    """
    print("🛠️ Preprocessing sensor data...")
    features = []

    if len(raw_values) > 0:
        features.append(np.mean(raw_values))
        features.append(np.std(raw_values))
        features.append(np.max(raw_values))
        features.append(np.min(raw_values))
    else:
        features.extend([0, 0, 0, 0])

    features = np.array(features).reshape(1, -1)

    if feature_names:
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df = features_df[feature_names]  # Ensure correct order
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

    print("🚀 Starting Flask server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
