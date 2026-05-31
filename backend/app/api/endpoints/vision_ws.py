from fastapi import APIRouter, WebSocket
import numpy as np
import cv2
import time

from services.yolo_service import get_yolo_service
from utils.image_pipeline import preprocess_frame
from core.vision_state import vision_state

router = APIRouter()

yolo = get_yolo_service()

prev_gains = None


@router.websocket("/frame")
async def receive_frame(websocket: WebSocket):
    global prev_gains

    await websocket.accept()

    last = 0
    FPS = 10
    interval = 1.0 / FPS

    while True:
        data = await websocket.receive_bytes()

        frame = cv2.imdecode(
            np.frombuffer(data, np.uint8),
            cv2.IMREAD_COLOR
        )

        now = time.time()
        if now - last < interval:
            continue
        last = now

        proc, prev_gains = preprocess_frame(frame, prev_gains)

        results, annotated = yolo.infer(proc)

        with vision_state.lock:
            vision_state.latest_frame = frame
            vision_state.latest_yolo_frame = annotated