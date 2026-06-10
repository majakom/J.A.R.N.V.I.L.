import logging

from ultralytics import YOLO
import cv2

from core.vision_state import vision_state

_yolo_service = None

class YOLOService:
    def __init__(self, model_path="best.pt"):
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        self.model = YOLO(model_path)

    def infer(self, frame):
        results = self.model(frame)

        boxes = results[0].boxes
        names = results[0].names

        if boxes is None:
            return results, results[0].plot()

        filtered_indices = []

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))

            if not vision_state.enabled:
                continue

            if class_name in vision_state.active_class_names:
                filtered_indices.append(i)

        # If nothing matches user intent → return empty view
        if not filtered_indices:
            blank = frame.copy()
            return results, blank

        # rebuild filtered result visualization
        filtered_boxes = boxes[filtered_indices]
        results[0].boxes = filtered_boxes
        return results, results[0].plot()
    
def get_yolo_service() -> YOLOService:
    global _yolo_service
    if _yolo_service is None:
        _yolo_service = YOLOService()
    return _yolo_service