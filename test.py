from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano version for speed

# Read a sample image
image = cv2.imread("sample.jpg")  # Ensure this file exists

# Run object detection
results = model(image)

# Iterate over results and visualize each one
for r in results:
    r.show()  # Show detected objects
