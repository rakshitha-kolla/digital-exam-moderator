import cv2
import time
import numpy as np
from face_detection import FaceDetection
from head_pose import HeadPose
from object_detection import ObjectDetector


class ExamProctor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetection()
        self.head_pose = HeadPose()
        self.object_detector = ObjectDetector()

        # Warning system
        self.warning_count = 0
        self.last_warning_time = 0
        self.warning_cooldown = 3  # seconds

        # Color codes
        self.colors = {
            0: (0, 255, 0),  # Green
            1: (0, 255, 255),  # Yellow
            2: (0, 165, 255),  # Orange
            3: (0, 0, 255)  # Red
        }

    def draw_face_rect(self, frame, landmarks, color):
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    def process_frame(self, frame):
        violations = []

        # 1. Face detection
        faces = self.face_detector.detect_faces(frame)
        if len(faces) > 1:
            violations.append("Multiple faces detected")
            for face in faces:
                self.draw_face_rect(frame, face, (0, 0, 255))

        # 2. Head pose (only process first face if multiple)
        if faces:
            landmarks = faces[0]
            yaw = self.head_pose.get_yaw(landmarks)
            if abs(yaw) > 0.02:  # Threshold for head turn
                violations.append("Looking away")

            # Draw face rectangle with warning color
            warning_level = min(self.warning_count, 3)
            self.draw_face_rect(frame, landmarks, self.colors[warning_level])

        # 3. Object detection
        prohibited = self.object_detector.detect_objects(frame)
        if prohibited:
            violations.append(f"Prohibited items: {', '.join(prohibited)}")

        return violations

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process frame and check for violations
                violations = self.process_frame(frame)

                # Update warning count if violations found
                if violations and (time.time() - self.last_warning_time) > self.warning_cooldown:
                    self.warning_count += 1
                    self.last_warning_time = time.time()
                    print(f"Warning {self.warning_count}: {', '.join(violations)}")

                # Display warning level
                warning_level = min(self.warning_count, 3)
                cv2.putText(frame, f"Warnings: {self.warning_count}/3",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, self.colors[warning_level], 2)

                # Terminate if 3 warnings
                if self.warning_count >= 3:
                    cv2.putText(frame, "EXAM TERMINATED", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.imshow("Exam Proctor", frame)
                    cv2.waitKey(3000)
                    break

                cv2.imshow("Exam Proctor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    proctor = ExamProctor()
    proctor.run()