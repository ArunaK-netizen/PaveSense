import websocket
import json
import numpy as np
import threading
import sqlite3
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Constants for pothole detection
ACCEL_Z_DROP_THRESHOLD = -2.5
ACCEL_Z_RISE_THRESHOLD = 3.0
GYRO_THRESHOLD = 1.5

# Database configuration
DB_FILE = "locations.db"
LOCATION_DISTANCE_THRESHOLD = 0.00005  # Approximately 5 meters in decimal degrees

# Data storage
sensor_data_buffer = {"accelerometer": [], "gyroscope": []}
pothole_locations = []

print("✅ Rule-Based Pothole Detection Active!")


# Initialize database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS locations
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       latitude
                       REAL
                       NOT
                       NULL,
                       longitude
                       REAL
                       NOT
                       NULL,
                       type
                       TEXT
                       NOT
                       NULL,
                       timestamp
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")


# Location handling functions
def add_location(latitude, longitude, location_type="pothole"):
    """Store location in database if it's not a duplicate"""
    if not is_duplicate_location(latitude, longitude):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO locations (latitude, longitude, type, timestamp) VALUES (?, ?, ?, ?)",
            (latitude, longitude, location_type, timestamp)
        )
        conn.commit()
        conn.close()
        print(f"✅ New {location_type} location added: ({latitude}, {longitude})")
        return True
    else:
        print(f"⚠️ Duplicate {location_type} location ignored: ({latitude}, {longitude})")
        return False


def is_duplicate_location(latitude, longitude):
    """Check if location is too close to existing ones"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT 1
                   FROM locations
                   WHERE ABS(latitude - ?) < ?
                     AND ABS(longitude - ?) < ? LIMIT 1
                   """, (latitude, LOCATION_DISTANCE_THRESHOLD, longitude, LOCATION_DISTANCE_THRESHOLD))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def get_all_locations():
    """Retrieve all stored locations"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT latitude, longitude, type FROM locations")
    locations = cursor.fetchall()
    conn.close()
    return locations


# Sensor event handlers
def on_accelerometer_event(values, timestamp):
    sensor_data_buffer["accelerometer"].append(values)
    process_sensor_data()


def on_gyroscope_event(values, timestamp):
    sensor_data_buffer["gyroscope"].append(values)
    process_sensor_data()


# GPS WebSocket handlers
def on_gps_message(ws, message):
    global lat, long
    try:
        data = json.loads(message)
        lat, long = data["latitude"], data["longitude"]
        lastKnownLocation = data.get("lastKnowLocation")
        print(f"📍 GPS: ({lat}, {long}) response to getLastKnownLocation = {lastKnownLocation}")
    except Exception as e:
        print(f"❌ Error processing GPS message: {e}")


def on_gps_error(ws, error):
    print(f"❌ GPS WebSocket error: {error}")


def on_gps_close(ws, close_code, reason):
    print(f"🔒 GPS WebSocket closed: {reason}")


def on_gps_open(ws):
    print("🔗 Connected to GPS WebSocket")
    ws.send("getLastKnowLocation")  # Request GPS location


def connect_gps(url):
    """Establish GPS WebSocket connection"""
    ws = websocket.WebSocketApp(url,
                                on_open=on_gps_open,
                                on_message=on_gps_message,
                                on_error=on_gps_error,
                                on_close=on_gps_close)
    ws.run_forever()


# Sensor class for handling accelerometer and gyroscope
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
            print(f"❌ Error processing {self.sensor_type} message: {e}")

    def on_error(self, ws, error):
        print(f"❌ Error in {self.sensor_type} WebSocket: {error}")

    def on_close(self, ws, close_code, reason):
        print(f"🔒 Connection closed for {self.sensor_type}: {reason}")

    def on_open(self, ws):
        print(f"🔗 Connected to {self.sensor_type} WebSocket!")

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
        thread = threading.Thread(target=self.make_websocket_connection, daemon=True)
        thread.start()


# Pothole detection algorithm
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

        # Add to database and emit to frontend if not a duplicate
        if add_location(lat, long, "pothole"):
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


@app.route("/api/locations")
def get_locations():
    """API endpoint to get all locations"""
    locations = get_all_locations()
    location_data = [
        {"latitude": lat, "longitude": lng, "type": loc_type}
        for lat, lng, loc_type in locations
    ]

    return jsonify(location_data)


# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Send all stored locations to newly connected clients"""
    locations = get_all_locations()
    for lat, lng, loc_type in locations:
        socketio.emit("update_map", {
            "latitude": lat,
            "longitude": lng,
            "type": loc_type,
            "isHistorical": True
        })
    print(f"📊 Sent {len(locations)} historical locations to client")


# Start WebSocket Connections and Flask Server
if __name__ == "__main__":
    # Initialize database
    init_db()

    sensor_address = "192.168.1.33:8080"
    gps_url = "ws://192.168.1.33:8080/gps"

    print("🛠️ Starting WebSocket connections...")

    # Start GPS WebSocket in a separate thread
    gps_thread = threading.Thread(target=connect_gps, args=(gps_url,), daemon=True)
    gps_thread.start()

    # Start sensor WebSockets
    Sensor(sensor_address, "android.sensor.accelerometer", on_accelerometer_event).connect()
    Sensor(sensor_address, "android.sensor.gyroscope", on_gyroscope_event).connect()

    print("🚀 Starting Flask server...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)