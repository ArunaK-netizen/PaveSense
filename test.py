import websocket
import json
from time import sleep
import sys
import threading

closed = False


def on_message(ws, message):
    data = json.loads(message)
    lat, long = data["latitude"], data["longitude"]
    lastKnownLocation = data["lastKnowLocation"]
    print(f"({lat},{long}) response to getLastKnownLocation = {lastKnownLocation}")


def on_error(ws, error):
    print("error occurred ", error)


def on_close(ws, close_code, reason):
    global closed
    closed = True
    print("connection closed : ", reason)


def on_open(ws):
    print("connected")
    thread = threading.Thread(target=send_requests, args=(ws,))
    thread.start()


def send_requests(ws):
    while True:
        if not closed:
            ws.send("getLastKnownLocation")
            sleep(1)  # 1 second sleep
        else:
            sys.exit()  # stop this thread


def connect(url):
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()


connect(f"ws://192.168.1.36:8080/gps")
