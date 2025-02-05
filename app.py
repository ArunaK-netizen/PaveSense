import websocket
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from geopy.distance import geodesic


app = Flask(__name__)
socketio = SocketIO(app)

# Load trained model
with open('model.p', 'rb') as file:
    pothole_data = pickle.load(file)

classifier = pothole_data['classifier']
threshold = 2
feature_names = pothole_data.get("feature_names", None)

pothole_locations = []


def on_message(ws, message):
    try:
        data = json.loads(message)
        sensor_data = data['values']

        raw_sensor_data = {
            "gyro_x": sensor_data[0:4],
            "gyro_y": sensor_data[4:8],
            "gyro_z": sensor_data[8:12],
            "acc_x": sensor_data[12:16],
            "acc_y": sensor_data[16:20],
            "acc_z": sensor_data[20:24],
        }

        input_features = preprocess_sensor_data(raw_sensor_data)
        decision_scores = classifier.decision_function(input_features)
        is_pothole = decision_scores[0] > threshold

        if is_pothole:
            latitude, longitude = retrieve_gps_location()
            pothole_locations.append((latitude, longitude))

            # Broadcast pothole location to clients
            socketio.emit("update_map", {"latitude": latitude, "longitude": longitude})
    except Exception as e:
        print("Error processing message:", e)

def retrieve_gps_location():
    """
    Retrieve GPS location and store detected potholes.
    """
    print("Location: Anupuram")
    return 0, 0


def compute_speed(new_location, new_time):
    """
    Compute speed based on GPS displacement over time.
    """
    global last_gps_location, last_gps_time

    if last_gps_location is None or last_gps_time is None:
        last_gps_location, last_gps_time = new_location, new_time
        return 0.0  # First reading, no speed available

    distance = geodesic(last_gps_location, new_location).meters  # Meters
    time_diff = new_time - last_gps_time  # Seconds
    speed = (distance / time_diff) if time_diff > 0 else 0.0  # Speed in m/s

    last_gps_location, last_gps_time = new_location, new_time
    return speed

import requests

def send_pothole_location(lat, lon):
    url = "http://127.0.0.1:5000/"  # Flask Server URL
    data = {"latitude": lat, "longitude": lon}
    try:
        requests.post(url, json=data)
    except Exception as e:
        print("Error sending pothole data:", e)


def preprocess_sensor_data(raw_sensor_data):
    """
    Calculate statistical features (mean, std, max, min) for each sensor axis.
    Ensure the model receives exactly 24 features as expected.
    """
    features = []

    for axis, values in raw_sensor_data.items():
        if len(values) > 0:
            features.append(np.mean(values))
            features.append(np.std(values))
            features.append(np.max(values))
            features.append(np.min(values))
        else:
            features.extend([0, 0, 0, 0])

    features = np.array(features).reshape(1, -1)

    # Ensure feature names are used in the correct order
    if feature_names:
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df = features_df[feature_names]  # Ensure correct order
    else:
        features_df = pd.DataFrame(features)

    return features_df


def connect(url):
    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    ws_url = "ws://192.168.1.36:8080/sensors/connect?types=[\"android.sensor.accelerometer\",\"android.sensor.gyroscope\"]"

    # Start WebSocket connection in a separate thread
    import threading

    ws_thread = threading.Thread(target=connect, args=(ws_url,))
    ws_thread.daemon = True
    ws_thread.start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)