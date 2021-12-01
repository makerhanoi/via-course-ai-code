import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
from PIL import Image

from lane_line_detection import *
from traffic_sign_detection import *

# Khởi tạo mô hình phân loại biển báo
traffic_sign_model = cv2.dnn.readNetFromONNX(
    "traffic_sign_classifier.onnx")

# Biến global để lưu hình ảnh hiện tại
# Chúng ta cần chạy mô hình phân loại biển báo ở process riêng
# Sử dụng biển này như một biến trung gian để trao đổi hình ảnh
g_image_queue = Queue(maxsize=5)

# Process phát hiện biển báo
def process_traffic_sign_loop(g_image_queue):
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()

        # Chuẩn bị 1 hình ảnh để vẽ kết quả
        draw = image.copy()
        # Phát hiện biển báo
        detect_traffic_signs(image, traffic_sign_model, draw=draw)
        # Hiện kết quả
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)


async def process_image(websocket, path):
    async for message in websocket:
        # Nhận hình ảnh từ giả lập
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Chuẩn bị 1 hình ảnh để vẽ các kết quả
        draw = image.copy()

        # Tính toán góc lái và tốc độ
        throttle, steering_angle = calculate_control_signal(image, draw=draw)

        # Cập nhật hình ảnh lên queue g_image_queue - dùng để chạy phát hiện biển báo
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Vẽ kết quả lên màn hình
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Gửi tín hiệu điều khiển lên giả lập
        message = json.dumps(
            {"throttle": throttle, "steering": steering_angle})
        await websocket.send(message)


async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue,))
    p.start()
    asyncio.run(main())
