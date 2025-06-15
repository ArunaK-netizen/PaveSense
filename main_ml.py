import websocket
import json
import numpy as np
import threading
import time
from datetime import datetime
from inference.real_time_predictor import IntegratedPotholeDetector
from utils.constants import *

# === Load model ===
try:
    model_path = 'models/trained_pothole_model.pth'
    ml_detector = IntegratedPotholeDetector(model_path, sequence_length=SEQUENCE_LENGTH)
    print("🤖 ML Pothole Detection System Initialized!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    ml_detector = None

# === Global state ===
sensor_data_buffer = {"accelerometer": [], "gyroscope": []}
lat, long = 0.0, 0.0

def ml_pothole_detection_callback(result):
    print("\n🚨 POTHOLE DETECTED!")
    print(f"   Method: {result['message']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Location: ({lat:.6f}, {long:.6f})")
    print(f"   Timestamp: {datetime.now().strftime('%H:%M:%S')}")

# Register ML callback
if ml_detector:
    ml_detector.set_detection_callback(ml_pothole_detection_callback)

def process_sensor_data():
    if not sensor_data_buffer["accelerometer"] or not sensor_data_buffer["gyroscope"]:
        return

    accel_current = np.array(sensor_data_buffer["accelerometer"][-1])
    gyro_current = np.array(sensor_data_buffer["gyroscope"][-1])

    if ml_detector:
        result = ml_detector.process_sensor_data(accel_current, gyro_current)
    else:
        accel_mag = np.linalg.norm(accel_current)
        gyro_mag = np.linalg.norm(gyro_current)

        if accel_mag > ACCEL_MAGNITUDE_THRESHOLD or gyro_mag > 2.5:
            result = {
                'pothole_detected': True,
                'confidence': min(0.9, (accel_mag + gyro_mag) / 20.0),
                'message': f'Rule: Pothole detected! (accel: {accel_mag:.2f})'
            }
            ml_pothole_detection_callback(result)
        else:
            result = {
                'pothole_detected': False,
                'confidence': 0.1,
                'message': f'Rule: Road OK (accel: {accel_mag:.2f})'
            }

    status = "🚨 POTHOLE" if result['pothole_detected'] else "✅ Road OK"
    print(f"\r{status} | A:{np.linalg.norm(accel_current):.2f} G:{np.linalg.norm(gyro_current):.2f} | Conf: {result['confidence']:.2f}", end='')

# === Sensor Events ===
def on_accel_event(values, timestamp):
    sensor_data_buffer["accelerometer"].append(values)
    if len(sensor_data_buffer["accelerometer"]) > 100:
        sensor_data_buffer["accelerometer"].pop(0)
    process_sensor_data()

def on_gyro_event(values, timestamp):
    sensor_data_buffer["gyroscope"].append(values)
    if len(sensor_data_buffer["gyroscope"]) > 100:
        sensor_data_buffer["gyroscope"].pop(0)
    process_sensor_data()

# === GPS WebSocket ===
def on_gps_message(ws, message):
    global lat, long
    try:
        data = json.loads(message)
        lat = float(data["latitude"])
        long = float(data["longitude"])
        print(f"\n📍 GPS: ({lat:.6f}, {long:.6f})")
    except Exception as e:
        print(f"❌ GPS error: {e}")

def on_gps_error(ws, error): print(f"❌ GPS WebSocket error: {error}")
def on_gps_close(ws, close_code, reason): print(f"🔒 GPS closed: {reason}")
def on_gps_open(ws): print("🔗 Connected to GPS"); ws.send("getLastKnowLocation")

def connect_gps(url):
    ws = websocket.WebSocketApp(
        url,
        on_open=on_gps_open,
        on_message=on_gps_message,
        on_error=on_gps_error,
        on_close=on_gps_close,
    )
    ws.run_forever()

# === Sensor WebSocket Wrapper ===
class Sensor:
    def __init__(self, address, sensor_type, on_event):
        self.address = address
        self.sensor_type = sensor_type
        self.on_event = on_event

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            values = data["values"]
            timestamp = data["timestamp"]
            self.on_event(values, timestamp)
        except Exception as e:
            print(f"❌ {self.sensor_type} error: {e}")

    def on_error(self, ws, error): print(f"❌ {self.sensor_type} WebSocket error: {error}")
    def on_close(self, ws, close_code, reason): print(f"🔒 {self.sensor_type} closed: {reason}")
    def on_open(self, ws): print(f"🔗 Connected to {self.sensor_type}")

    def make_ws_connection(self):
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
        thread = threading.Thread(target=self.make_ws_connection, daemon=True)
        thread.start()

# === Entry Point ===
if __name__ == "__main__":
    sensor_address = "192.168.1.33:8080"
    gps_url = f"ws://{sensor_address}/gps"

    print("🚗 Starting minimal pothole detection system...")
    print("=" * 60)
    print(f"🤖 ML Model: {'Loaded' if ml_detector and ml_detector.model else 'Rule-based'}")
    print(f"📱 Phone Address: {sensor_address}")
    print("=" * 60)

    threading.Thread(target=connect_gps, args=(gps_url,), daemon=True).start()
    Sensor(sensor_address, "android.sensor.accelerometer", on_accel_event).connect()
    Sensor(sensor_address, "android.sensor.gyroscope", on_gyro_event).connect()

    # Prevent CPU overuse
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Exiting...")
