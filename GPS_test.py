import websocket
import json
from time import sleep
import sys
import threading

sensor_address = "192.168.1.36:8080"
closed = False


def on_message(ws, message):
    data = json.loads(message)
    lat, long = data.get("latitude"), data.get("longitude")
    lastKnownLocation = data.get("lastKnowLocation")
    print(f"({lat}, {long}) response to getLastKnownLocation = {lastKnownLocation}")


def on_error(ws, error):
    print("Error occurred:", error)


def on_close(ws, close_code, reason):
    global closed
    closed = True
    print("Connection closed:", reason)


def on_open(ws):
    print("Connected to sensor")
    thread = threading.Thread(target=send_requests, args=(ws,))
    thread.start()


def send_requests(ws):
    while not closed:
        try:
            ws.send("getLastKnownLocation")
            sleep(1)  # 1 second sleep
        except Exception as e:
            print("Error sending request:", e)
            break


def connect(url):
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()


# Connect to the new sensor address
connect(f"ws://{sensor_address}/gps")
