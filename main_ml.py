import websocket
import json
import numpy as np
import threading
import sqlite3
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime
import os

# Import ML components
from inference.real_time_predictor import IntegratedPotholeDetector
from utils.constants import *

app = Flask(__name__)
socketio = SocketIO(app)

# Data storage
sensor_data_buffer = {"accelerometer": [], "gyroscope": []}
pothole_locations = []

# Initialize ML detector
try:
    model_path = 'models/trained_pothole_model.pth'
    ml_detector = IntegratedPotholeDetector(model_path, sequence_length=SEQUENCE_LENGTH)
    print("🤖 ML Pothole Detection System Initialized!")
except Exception as e:
    print(f"❌ ML initialization error: {e}")
    ml_detector = None

# Global variables for GPS
lat, long = 0.0, 0.0

# Statistics
detection_stats = {
    "total_readings": 0,
    "ml_detections": 0,
    "session_start": datetime.now()
}


def ml_pothole_detection_callback(result):
    """Callback for ML pothole detections"""
    global lat, long, detection_stats

    detection_stats["ml_detections"] += 1

    print(f"\n🚨 POTHOLE DETECTED!")
    print(f"   Method: {result['message']}")
    print(f"   Confidence: {result['confidence']:.3f} ({result['confidence'] * 100:.1f}%)")
    print(f"   Location: ({lat:.6f}, {long:.6f})")
    print(f"   Total detections: {detection_stats['ml_detections']}")

    # Store in database
    if add_location(lat, long, "pothole"):
        pothole_locations.append((lat, long))

        # Emit to frontend
        socketio.emit("update_map", {
            "latitude": lat,
            "longitude": long,
            "type": "pothole",
            "confidence": result['confidence'],
            "detection_method": "ML" if "ML:" in result['message'] else "Rule",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })


# Set the ML callback
if ml_detector:
    ml_detector.set_detection_callback(ml_pothole_detection_callback)


# Database functions (your existing ones)
def init_db():
    os.makedirs("database", exist_ok=True)
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


def add_location(latitude, longitude, location_type="pothole"):
    """Store location in database"""
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
        print(f"✅ New {location_type} location added: ({latitude:.6f}, {longitude:.6f})")
        return True
    else:
        print(f"⚠️ Duplicate {location_type} location ignored")
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


# Sensor event handlers (your existing ones)
def on_accelerometer_event(values, timestamp):
    sensor_data_buffer["accelerometer"].append(values)
    if len(sensor_data_buffer["accelerometer"]) > 100:
        sensor_data_buffer["accelerometer"].pop(0)
    process_sensor_data()


def on_gyroscope_event(values, timestamp):
    sensor_data_buffer["gyroscope"].append(values)
    if len(sensor_data_buffer["gyroscope"]) > 100:
        sensor_data_buffer["gyroscope"].pop(0)
    process_sensor_data()


# GPS handlers (your existing ones)
def on_gps_message(ws, message):
    global lat, long
    try:
        data = json.loads(message)
        lat, long = data["latitude"], data["longitude"]
        print(f"📍 GPS: ({lat:.6f}, {long:.6f})")
    except Exception as e:
        print(f"❌ Error processing GPS message: {e}")


def on_gps_error(ws, error):
    print(f"❌ GPS WebSocket error: {error}")


def on_gps_close(ws, close_code, reason):
    print(f"🔒 GPS WebSocket closed: {reason}")


def on_gps_open(ws):
    print("🔗 Connected to GPS WebSocket")
    ws.send("getLastKnowLocation")


def connect_gps(url):
    """Establish GPS WebSocket connection"""
    ws = websocket.WebSocketApp(url,
                                on_open=on_gps_open,
                                on_message=on_gps_message,
                                on_error=on_gps_error,
                                on_close=on_gps_close)
    ws.run_forever()


# Sensor class (your existing one)
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


# ML-enhanced sensor data processing
def process_sensor_data():
    """ML-enhanced sensor data processing"""
    global detection_stats

    if len(sensor_data_buffer["accelerometer"]) < 1 or len(sensor_data_buffer["gyroscope"]) < 1:
        return

    # Get current sensor data
    accel_current = np.array(sensor_data_buffer["accelerometer"][-1])
    gyro_current = np.array(sensor_data_buffer["gyroscope"][-1])

    detection_stats["total_readings"] += 1

    # Process with ML model
    if ml_detector:
        result = ml_detector.process_sensor_data(accel_current, gyro_current)
    else:
        # Fallback rule-based detection
        accel_magnitude = np.linalg.norm(accel_current)
        gyro_magnitude = np.linalg.norm(gyro_current)

        if accel_magnitude > ACCEL_MAGNITUDE_THRESHOLD or gyro_magnitude > 2.5:
            confidence = min(0.9, (accel_magnitude + gyro_magnitude) / 20.0)
            result = {
                'pothole_detected': True,
                'confidence': confidence,
                'message': f'Rule: Pothole detected! (accel: {accel_magnitude:.2f})'
            }
            if ml_pothole_detection_callback:
                ml_pothole_detection_callback(result)
        else:
            result = {
                'pothole_detected': False,
                'confidence': 0.1,
                'message': f'Rule: Road OK (accel: {accel_magnitude:.2f})'
            }

    # Display current readings
    accel_mag = np.linalg.norm(accel_current)
    gyro_mag = np.linalg.norm(gyro_current)

    status_icon = "🚨" if result['pothole_detected'] else "✅"
    confidence_bar = "█" * int(result['confidence'] * 20)

    print(f"\r{status_icon} A:{accel_mag:6.2f} G:{gyro_mag:6.2f} "
          f"Conf:{result['confidence']:.3f} {confidence_bar:<20} "
          f"Det:{detection_stats['ml_detections']}/{detection_stats['total_readings']}", end='')


# Flask routes (your existing ones)
@app.route("/")
def index():
    return "<h1>🚗 ML Pothole Detection System</h1><p>Connect your phone sensors to start detection!</p>"


@app.route("/api/locations")
def get_locations():
    """API endpoint to get all locations"""
    locations = get_all_locations()
    location_data = [
        {"latitude": lat, "longitude": lng, "type": loc_type}
        for lat, lng, loc_type in locations
    ]
    return jsonify(location_data)


@app.route("/api/stats")
def get_stats():
    """API endpoint for detection statistics"""
    session_duration = (datetime.now() - detection_stats["session_start"]).total_seconds()
    return jsonify({
        "total_readings": detection_stats["total_readings"],
        "ml_detections": detection_stats["ml_detections"],
        "session_duration_minutes": round(session_duration / 60, 1),
        "detection_rate": round(detection_stats["ml_detections"] / max(session_duration / 60, 1), 2),
        "model_loaded": ml_detector is not None and hasattr(ml_detector, 'model') and ml_detector.model is not None
    })


# SocketIO handlers (your existing ones)
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


if __name__ == "__main__":
    # Initialize database
    init_db()

    # CHANGE THIS TO YOUR PHONE'S IP ADDRESS
    sensor_address = "192.168.1.37:8080"
    gps_url = f"ws://{sensor_address}/gps"

    print("🚗 ML Pothole Detection System Starting...")
    print("=" * 60)
    print(f"🤖 ML Model: {'Loaded' if ml_detector and ml_detector.model else 'Rule-based fallback'}")
    print(f"📱 Phone Address: {sensor_address}")
    print("=" * 60)

    # Start GPS WebSocket
    gps_thread = threading.Thread(target=connect_gps, args=(gps_url,), daemon=True)
    gps_thread.start()

    # Start sensor WebSockets
    Sensor(sensor_address, "android.sensor.accelerometer", on_accelerometer_event).connect()
    Sensor(sensor_address, "android.sensor.gyroscope", on_gyroscope_event).connect()

    print("🚀 Starting Flask server on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
