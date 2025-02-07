import websocket

ws_url = "ws://192.168.1.36:8080/sensor/connect?type=android.sensor.accelerometer"

def on_message(ws, message):
    print("✅ WebSocket Received Data:", message)

def on_error(ws, error):
    print("❌ WebSocket Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("🔌 WebSocket Closed")

def on_open(ws):
    print("🔗 WebSocket Connection Opened - Waiting for Data...")

ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
ws.on_open = on_open
ws.run_forever()
