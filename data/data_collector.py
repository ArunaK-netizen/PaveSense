import websocket
import json
import numpy as np
import threading
import pandas as pd
from datetime import datetime
import time
import os
import keyboard  # pip install keyboard
from collections import deque


class PotholeDataCollector:
    def __init__(self, phone_ip="192.168.1.37:8080"):
        self.phone_ip = phone_ip
        self.collecting = False
        self.data_buffer = []
        self.current_session = {
            "timestamp": [],
            "accel_x": [],
            "accel_y": [],
            "accel_z": [],
            "gyro_x": [],
            "gyro_y": [],
            "gyro_z": [],
            "latitude": [],
            "longitude": [],
            "label": [],  # 0 = normal, 1 = pothole
            "speed": []
        }

        # Current sensor values
        self.current_accel = [0, 0, 0]
        self.current_gyro = [0, 0, 0]
        self.current_gps = {"lat": 0.0, "lng": 0.0, "speed": 0.0}

        # Pothole marking
        self.pothole_window = 3.0  # seconds before and after pothole mark
        self.pothole_events = []  # timestamps of pothole events

        # Data collection stats
        self.total_samples = 0
        self.pothole_samples = 0
        self.session_start = None

        print("🚗 Pothole Data Collector Initialized")
        print("📱 Phone IP:", phone_ip)
        print("🔴 Press SPACEBAR to mark a pothole when you feel one!")
        print("🟢 Press 'S' to start/stop data collection")
        print("💾 Press 'Q' to quit and save data")

    def start_collection(self):
        """Start collecting sensor data"""
        self.collecting = True
        self.session_start = time.time()

        # Start keyboard listener
        keyboard.on_press_key("space", self.mark_pothole)
        keyboard.on_press_key("s", self.toggle_collection)
        keyboard.on_press_key("q", self.quit_and_save)

        # Connect to phone sensors
        self.connect_sensors()

        # Start data collection loop
        self.collection_loop()

    def mark_pothole(self, e):
        """Mark current time as pothole event"""
        if self.collecting:
            pothole_time = time.time()
            self.pothole_events.append(pothole_time)
            print(f"\n🚨 POTHOLE MARKED at {datetime.now().strftime('%H:%M:%S')}")
            print(f"   Total potholes marked: {len(self.pothole_events)}")
            print(
                f"   Current accel: [{self.current_accel[0]:.2f}, {self.current_accel[1]:.2f}, {self.current_accel[2]:.2f}]")
            print(
                f"   Current gyro:  [{self.current_gyro[0]:.2f}, {self.current_gyro[1]:.2f}, {self.current_gyro[2]:.2f}]")

    def toggle_collection(self, e):
        """Toggle data collection on/off"""
        self.collecting = not self.collecting
        status = "STARTED" if self.collecting else "STOPPED"
        print(f"\n📊 Data collection {status}")
        if self.collecting:
            self.session_start = time.time()

    def quit_and_save(self, e):
        """Save data and quit"""
        print("\n💾 Saving data and quitting...")
        self.save_dataset()
        exit()

    def connect_sensors(self):
        """Connect to phone sensors"""
        try:
            # Start GPS WebSocket
            gps_thread = threading.Thread(target=self.connect_gps, daemon=True)
            gps_thread.start()

            # Start sensor WebSockets
            accel_thread = threading.Thread(target=self.connect_accelerometer, daemon=True)
            accel_thread.start()

            gyro_thread = threading.Thread(target=self.connect_gyroscope, daemon=True)
            gyro_thread.start()

            print("🔗 Connecting to phone sensors...")
            time.sleep(2)  # Give connections time to establish

        except Exception as e:
            print(f"❌ Error connecting to sensors: {e}")

    def connect_gps(self):
        """Connect to GPS WebSocket"""

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.current_gps = {
                    "lat": data.get("latitude", 0.0),
                    "lng": data.get("longitude", 0.0),
                    "speed": data.get("speed", 0.0)
                }
            except Exception as e:
                pass

        def on_open(ws):
            print("🔗 GPS connected")
            ws.send("getLastKnowLocation")

        gps_url = f"ws://{self.phone_ip}/gps"
        ws = websocket.WebSocketApp(gps_url, on_open=on_open, on_message=on_message)
        ws.run_forever()

    def connect_accelerometer(self):
        """Connect to accelerometer WebSocket"""

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.current_accel = data["values"]
            except Exception as e:
                pass

        def on_open(ws):
            print("🔗 Accelerometer connected")

        accel_url = f"ws://{self.phone_ip}/sensor/connect?type=android.sensor.accelerometer"
        ws = websocket.WebSocketApp(accel_url, on_open=on_open, on_message=on_message)
        ws.run_forever()

    def connect_gyroscope(self):
        """Connect to gyroscope WebSocket"""

        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.current_gyro = data["values"]
            except Exception as e:
                pass

        def on_open(ws):
            print("🔗 Gyroscope connected")

        gyro_url = f"ws://{self.phone_ip}/sensor/connect?type=android.sensor.gyroscope"
        ws = websocket.WebSocketApp(gyro_url, on_open=on_open, on_message=on_message)
        ws.run_forever()

    def collection_loop(self):
        """Main data collection loop"""
        print("\n🚀 Data collection ready!")
        print("📝 Drive around and press SPACEBAR when you hit potholes")
        print("=" * 60)

        while True:
            if self.collecting:
                # Record current sensor data
                current_time = time.time()

                self.current_session["timestamp"].append(current_time)
                self.current_session["accel_x"].append(self.current_accel[0])
                self.current_session["accel_y"].append(self.current_accel[1])
                self.current_session["accel_z"].append(self.current_accel[2])
                self.current_session["gyro_x"].append(self.current_gyro[0])
                self.current_session["gyro_y"].append(self.current_gyro[1])
                self.current_session["gyro_z"].append(self.current_gyro[2])
                self.current_session["latitude"].append(self.current_gps["lat"])
                self.current_session["longitude"].append(self.current_gps["lng"])
                self.current_session["speed"].append(self.current_gps["speed"])

                # Determine label (0 = normal, 1 = pothole)
                label = self.get_current_label(current_time)
                self.current_session["label"].append(label)

                self.total_samples += 1
                if label == 1:
                    self.pothole_samples += 1

                # Display current status
                self.display_status()

            time.sleep(0.02)  # 50 Hz sampling rate

    def get_current_label(self, current_time):
        """Determine if current time should be labeled as pothole"""
        for pothole_time in self.pothole_events:
            if abs(current_time - pothole_time) <= self.pothole_window:
                return 1  # Pothole
        return 0  # Normal road

    def display_status(self):
        """Display real-time collection status"""
        if self.total_samples % 25 == 0:  # Update every 0.5 seconds
            session_time = time.time() - self.session_start if self.session_start else 0
            accel_mag = np.sqrt(sum(x ** 2 for x in self.current_accel))
            gyro_mag = np.sqrt(sum(x ** 2 for x in self.current_gyro))

            pothole_pct = (self.pothole_samples / max(self.total_samples, 1)) * 100

            print(f"\r📊 Time:{session_time:6.1f}s | Samples:{self.total_samples:5d} | "
                  f"Potholes:{self.pothole_samples:3d}({pothole_pct:4.1f}%) | "
                  f"Accel:{accel_mag:6.2f} | Gyro:{gyro_mag:6.2f} | "
                  f"Speed:{self.current_gps['speed']:4.1f}km/h", end='')

    def save_dataset(self):
        """Save collected data to CSV file"""
        if len(self.current_session["timestamp"]) == 0:
            print("❌ No data collected!")
            return

        # Create dataset directory
        os.makedirs("datasets", exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame(self.current_session)

        # Add session metadata
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"datasets/pothole_data_{session_id}.csv"

        # Save to CSV
        df.to_csv(filename, index=False)

        # Print statistics
        print(f"\n💾 Dataset saved: {filename}")
        print(f"📊 Total samples: {len(df)}")
        print(f"🚨 Pothole samples: {self.pothole_samples} ({self.pothole_samples / len(df) * 100:.1f}%)")
        print(
            f"✅ Normal samples: {len(df) - self.pothole_samples} ({(len(df) - self.pothole_samples) / len(df) * 100:.1f}%)")
        print(f"⏱️  Duration: {(df['timestamp'].max() - df['timestamp'].min()) / 60:.1f} minutes")
        print(f"📍 GPS points: {df[df['latitude'] != 0].shape[0]}")

        # Save summary
        summary = {
            "session_id": session_id,
            "total_samples": len(df),
            "pothole_samples": self.pothole_samples,
            "normal_samples": len(df) - self.pothole_samples,
            "duration_minutes": (df['timestamp'].max() - df['timestamp'].min()) / 60,
            "pothole_events": len(self.pothole_events),
            "sampling_rate": "50 Hz",
            "pothole_window_seconds": self.pothole_window
        }

        summary_filename = f"datasets/summary_{session_id}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📋 Summary saved: {summary_filename}")


def main():
    print("🚗 Pothole Dataset Collection System")
    print("=" * 50)

    # Get phone IP
    phone_ip = input("Enter your phone's IP address (e.g., 192.168.1.37:8080): ").strip()
    if not phone_ip:
        phone_ip = "192.168.1.37:8080"

    # Create collector
    collector = PotholeDataCollector(phone_ip)

    print("\n📋 INSTRUCTIONS:")
    print("1. Start your phone sensor app")
    print("2. Press 'S' to start data collection")
    print("3. Drive around normally")
    print("4. Press SPACEBAR immediately when you hit a pothole")
    print("5. Press 'Q' when done to save data")
    print("\n🚨 IMPORTANT: Be safe! Have a passenger mark potholes or pull over!")

    input("\nPress ENTER when ready to start...")

    try:
        collector.start_collection()
    except KeyboardInterrupt:
        print("\n\n💾 Saving data...")
        collector.save_dataset()


if __name__ == "__main__":
    main()
