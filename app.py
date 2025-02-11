import websocket
import json
import numpy as np
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Threshold values for pothole detection
ACCEL_Z_DROP_THRESHOLD = -2.5  # Large downward acceleration (falling into pothole)
ACCEL_Z_RISE_THRESHOLD = 3.0   # Large upward acceleration (rising out of pothole)
GYRO_THRESHOLD = 1.5           # High gyroscope variation indicating road impact

# Data buffers for recent sensor values
sensor_data_buffer = {"accelerometer": [], "gyroscope": []}
pothole_locations = []
lat, long = "13.0827", "80.2707"  # Default to Chennai

print("✅ Rule-Based Pothole Detection Active!")

# WebSocket Sensor Event Handlers
def on_accelerometer_event(values, timestamp):
    print(f"📡 Accelerometer Data: {values} | Timestamp: {timestamp}")
    sensor_data_buffer["accelerometer"].append(values)
    process_sensor_data()

def on_gyroscope_event(values, timestamp):
    print(f"📡 Gyroscope Data: {values} | Timestamp: {timestamp}")
    sensor_data_buffer["gyroscope"].append(values)
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

# Rule-Based Pothole Detection
def process_sensor_data():
    """ Detects potholes based on sensor thresholds """
    global lat, long

    if len(sensor_data_buffer["accelerometer"]) < 2 or len(sensor_data_buffer["gyroscope"]) < 1:
        return  # Wait until enough sensor data is collected

    # Get latest sensor readings
    accel_current = np.array(sensor_data_buffer["accelerometer"][-1])
    accel_previous = np.array(sensor_data_buffer["accelerometer"][-2])  # Previous reading for comparison
    gyro_current = np.array(sensor_data_buffer["gyroscope"][-1])

    # Compute acceleration changes
    accel_z_change = accel_current[2] - accel_previous[2]  # Change in Z-axis acceleration
    gyro_variance = np.var(gyro_current)  # Gyroscope variation

    # Check if pothole detected
    pothole_detected = (
        accel_z_change < ACCEL_Z_DROP_THRESHOLD and  # Sudden drop in Z-axis
        np.abs(accel_current[2]) > ACCEL_Z_RISE_THRESHOLD and  # Immediate rise after drop
        gyro_variance > GYRO_THRESHOLD  # Sudden gyroscope movement
    )

    if pothole_detected:
        print("🚧 POTHOLE DETECTED! Using Chennai GPS Data.")
        pothole_locations.append((lat, long))
        socketio.emit("update_map", {"latitude": lat, "longitude": long})
    else:
        print("✅ No pothole detected.")

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
