import threading


class VisionState:
    def __init__(self):
        self.active_class_names = set()
        self.enabled = False

        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_yolo_frame = None

vision_state = VisionState()