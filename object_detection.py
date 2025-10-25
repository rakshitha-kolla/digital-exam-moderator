from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # 'n' for speed; upgrade to 'm' or 'l' for accuracy
        self.prohibited_objects = {"cell phone", "laptop", "book", "tablet", "phone", "electronic device"}

    def detect_objects(self, frame):
        results = self.model(frame)  # Run YOLOv8 detection
        violations = set()

        for r in results:
            boxes = r.boxes.xyxy  # Bounding boxes
            class_ids = r.boxes.cls  # Class IDs
            confidences = r.boxes.conf  # Confidence scores

            for i in range(len(boxes)):
                class_name = self.model.names[int(class_ids[i])]
                confidence = confidences[i].item()

                if class_name.lower() in self.prohibited_objects and confidence > 0.3:
                    violations.add(class_name)
_productsorted
        return list(violations)
