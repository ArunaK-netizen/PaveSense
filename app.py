import websocket
import json
import numpy as np
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

ACCEL_Z_DROP_THRESHOLD = -2.5
ACCEL_Z_RISE_THRESHOLD = 3.0
GYRO_THRESHOLD = 1.5

sensor_data_buffer = {"accelerometer": [], "gyroscope": []}
pothole_locations = []
lat, long = "12.8435778", "80.1548060"

print("✅ Rule-Based Pothole Detection Active!")


def on_accelerometer_event(values, timestamp):
    sensor_data_buffer["accelerometer"].append(values)
    process_sensor_data()


def on_gyroscope_event(values, timestamp):
    sensor_data_buffer["gyroscope"].append(values)
    process_sensor_data()


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
            print(f"Error processing {self.sensor_type} message:", e)

    def on_error(self, ws, error):
        print(f"Error in {self.sensor_type} WebSocket:", error)

    def on_close(self, ws, close_code, reason):
        print(f"Connection closed for {self.sensor_type}: {reason}")

    def on_open(self, ws):
        print(f"Connected to {self.sensor_type} WebSocket!")

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


def process_sensor_data():
    global lat, long

    if len(sensor_data_buffer["accelerometer"]) < 2 or len(sensor_data_buffer["gyroscope"]) < 1:
        return

    accel_current = np.array(sensor_data_buffer["accelerometer"][-1])
    accel_previous = np.array(sensor_data_buffer["accelerometer"][-2])
    gyro_current = np.array(sensor_data_buffer["gyroscope"][-1])

    accel_z_change = accel_current[2] - accel_previous[2]
    accel_magnitude = np.linalg.norm(accel_current)
    accel_change_magnitude = np.linalg.norm(accel_current - accel_previous)
    gyro_magnitude = np.linalg.norm(gyro_current)
    gyro_variance = np.var(gyro_current)

    print("\n--- SENSOR MAGNITUDES ---")
    print(f"Accel XYZ: [{accel_current[0]:.2f}, {accel_current[1]:.2f}, {accel_current[2]:.2f}]")
    print(f"Accel Z Change: {accel_z_change:.2f} (Drop Threshold: {ACCEL_Z_DROP_THRESHOLD})")
    print(f"Gyro Variance: {gyro_variance:.2f} (Threshold: {GYRO_THRESHOLD})")


    pothole_detected = (

            accel_z_change < ACCEL_Z_DROP_THRESHOLD and
            np.abs(accel_current[2]) > ACCEL_Z_RISE_THRESHOLD and
            gyro_variance > GYRO_THRESHOLD
    )

    if pothole_detected:
        print("🚧 POTHOLE DETECTED! Triggered by:")
        print(f" - Z-axis drop: {accel_z_change:.2f} < {ACCEL_Z_DROP_THRESHOLD}")
        print(f" - Z-axis absolute: {np.abs(accel_current[2]):.2f} > {ACCEL_Z_RISE_THRESHOLD}")
        print(f" - Gyro variance: {gyro_variance:.2f} > {GYRO_THRESHOLD}")

        pothole_locations.append((lat, long))
        socketio.emit("update_map", {"latitude": lat, "longitude": long, "type": "pothole"})

    else:
        print("No events detected")

    print("-------------------------")


# Flask Routes
@app.route("/")
def index():
    print("🖥️ Rendering index.html")
    return render_template("./index.html")


# Start WebSocket Connections for Sensors
if __name__ == "__main__":
    sensor_address = "192.168.1.33:8080"

    print("🛠️ Starting WebSocket connections for sensors...")
    Sensor(sensor_address, "android.sensor.accelerometer", on_accelerometer_event).connect()
    Sensor(sensor_address, "android.sensor.gyroscope", on_gyroscope_event).connect()

    print("🚀 Starting Flask server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
