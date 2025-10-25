
class HeadPose:
    def __init__(self):
        pass

    def get_yaw(self, landmarks):
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        nose_tip = landmarks.landmark[1]
        eye_dx = right_eye.x - left_eye.x
        nose_dx = nose_tip.x - (left_eye.x + eye_dx / 2)
        return nose_dx