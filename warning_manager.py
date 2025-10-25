import cv2

class WarningManager:
    def __init__(self):
        self.warning_count = 0
        self.colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        self.terminated = False
    def get_color(self):
        return self.colors[min(self.warning_count, 3)]

    def issue_warning(self, frame):
        self.warning_count += 1
        msg = f"⚠️ Warning {self.warning_count}!"
        if self.warning_count == 3:
            msg = "❌ Exam Terminated!"
            self.terminated = True

        cv2.putText(frame, msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return frame