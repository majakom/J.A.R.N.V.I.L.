from picamera2 import Picamera2
import cv2

class Camera:
    def __init__(self):
        self.cam = Picamera2()
        self.cam.configure(
            self.cam.create_video_configuration(
                main={"format": "BGR888", "size": (640, 480)}
            )
        )

    def start(self):
        self.cam.start()

    def get_frame(self):
        return self.cam.capture_array()