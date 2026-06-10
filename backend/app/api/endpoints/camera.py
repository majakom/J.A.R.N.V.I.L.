from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi_utils.cbv import cbv
import cv2
import time
from core.vision_state import vision_state

router = APIRouter()



@cbv(router)
class CameraEndpoints:

    @router.get("/frame")
    def get_frame(self):
        with vision_state.lock:
            frame = vision_state.latest_frame.copy() if vision_state.latest_frame is not None else None

        if frame is None:
            raise HTTPException(404, "No frame available")

        _, buffer = cv2.imencode(".jpg", frame)

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @router.get("/frame/yolo")
    def get_yolo_frame(self):
        with vision_state.lock:
            frame = vision_state.latest_yolo_frame.copy() if vision_state.latest_yolo_frame is not None else None

        if frame is None:
            raise HTTPException(404, "No YOLO frame available")

        _, buffer = cv2.imencode(".jpg", frame)

        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    @router.get("/stream/yolo")
    def stream_yolo(self):
        def generate():
            while True:
                with vision_state.lock:
                    frame = vision_state.latest_yolo_frame.copy() if vision_state.latest_yolo_frame is not None else None

                if frame is None:
                    time.sleep(0.05)
                    continue

                _, buffer = cv2.imencode(".jpg", frame)

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    buffer.tobytes() +
                    b"\r\n"
                )

                time.sleep(0.03)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )