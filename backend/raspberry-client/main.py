import asyncio
from camera import Camera
from sender import send_frames

cam = Camera()
cam.start()

asyncio.run(send_frames(cam))