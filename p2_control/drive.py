import asyncio
import websockets
from PIL import Image
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from lane_line_detection import calculate_control_signal

async def process_image(websocket, path):
    async for message in websocket:
        # Nhận hình ảnh từ giả lập
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Tính toán góc lái và tốc độ
        throttle, steering_angle = calculate_control_signal(image)

        # Gửi tín hiệu điều khiển lên giả lập
        message = json.dumps({"throttle": throttle, "steering": steering_angle})
        print(message)
        await websocket.send(message)
        

async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

asyncio.run(main())