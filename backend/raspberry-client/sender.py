import cv2
import asyncio
import websockets

BACKEND_URL = "ws://{{IP}}/api/ws/frame"
print("CONNECTING TO:", BACKEND_URL)

async def send_frames(camera):
    async with websockets.connect(BACKEND_URL) as ws:
        while True:
            frame = camera.get_frame()

            _, jpg = cv2.imencode(".jpg", frame)

            await ws.send(jpg.tobytes())
            await asyncio.sleep(0.03)  # ~30 FPS max