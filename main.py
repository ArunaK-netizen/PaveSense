import websocket
import json
import pickle
import numpy as np


# Load the saved SVC model and threshold
with open('model.p', 'rb') as file:
    pothole_data = pickle.load(file)

classifier = pothole_data['classifier']  # Extract the classifier
threshold = pothole_data['threshold']  # Extract the threshold


def preprocess_sensor_data(raw_sensor_data):
    """
    Calculate statistical features (mean, std, max, min) for each sensor axis.
    """
    features = []

    for axis, values in raw_sensor_data.items():
        if len(values) > 0:
            features.append(np.mean(values))  # Mean
            features.append(np.std(values))  # Standard deviation
            features.append(np.max(values))  # Maximum
            features.append(np.min(values))  # Minimum
        else:
            # Handle empty values by appending default values (e.g., 0)
            features.extend([0, 0, 0, 0])

    return np.array(features).reshape(1, -1)  # Reshape to 2D array


def retrieve_gps_location():
    """
    Retrieve GPS location using a separate WebSocket connection.
    """
    def on_message(ws, message):
        data = json.loads(message)
        lat, long = data["latitude"], data["longitude"]
        lastKnownLocation = data["lastKnowLocation"]
        print(f"Pothole Detected at Location: ({lat}, {long}), Last Known Location: {lastKnownLocation}")
        ws.close()  # Close the WebSocket connection after retrieving the location

    def on_error(ws, error):
        print("Error retrieving GPS location:", error)

    def on_close(ws, close_code, reason):
        print("GPS WebSocket connection closed")

    def on_open(ws):
        print("Retrieving GPS location...")
        ws.send("getLastKnowLocation")  # Trigger location retrieval

    ws = websocket.WebSocketApp(
        "ws://192.168.0.102:8081/gps",  # Update with your GPS WebSocket server URL
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()


# WebSocket callbacks for sensor data
def on_message(ws, message):
    """
    Handle incoming WebSocket messages for sensor data.
    """
    try:
        sensor_data = json.loads(message)['values']

        # Extract data for required sensors
        raw_sensor_data = {
            "gyro_x": sensor_data[0:4],
            "gyro_y": sensor_data[4:8],
            "gyro_z": sensor_data[8:12],
            "acc_x": sensor_data[12:16],
            "acc_y": sensor_data[16:20],
            "acc_z": sensor_data[20:24],
        }

        # Preprocess data to calculate all 24 features
        input_features = preprocess_sensor_data(raw_sensor_data)

        # Predict using the model
        decision_scores = classifier.decision_function(input_features)
        is_pothole = decision_scores[0] > threshold

        # Output results
        print(f"Decision Score: {decision_scores[0]:.2f}")
        # if is_pothole:
        #     print("Pothole Detected")
        #     retrieve_gps_location()
        # else:
        #     print("No Pothole Detected")

    except KeyError as e:
        print("Missing data field:", e)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
    except ValueError as e:
        print("Model prediction error:", e)


def on_error(ws, error):
    print("Error occurred:", error)


def on_close(ws, close_code, reason):
    print("Connection closed")
    print(f"Close code: {close_code}, Reason: {reason}")


def on_open(ws):
    print("WebSocket connection opened")


def connect(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()


if __name__ == "__main__":
    # WebSocket URL for SensorServer
    ws_url = "ws://192.168.1.37:8080/sensors/connect?types=[\"android.sensor.accelerometer\",\"android.sensor.gyroscope\"]"

    # Connect to WebSocket
    connect(ws_url)
