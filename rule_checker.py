import time

class RuleChecker:
    def __init__(self):
        self.last_check = time.time()

    def check(self, bboxes, object_flag):
        """Monitors face count and prohibited objects."""
        print(f"[DEBUG] Faces detected: {len(bboxes)}")
        print(f"[DEBUG] Object flag: {object_flag}")

        # Violations
        if len(bboxes) != 1:
            print("[VIOLATION] Invalid number of faces detected!")
            return True

        if object_flag:
            print("[VIOLATION] Suspicious object detected!")
            return True

        return False
