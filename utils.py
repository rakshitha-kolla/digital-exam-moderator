import cv2

def draw_stylish_text(img, text, position, color=(0, 255, 255), scale=1.0, thickness=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, position, font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, position, font, scale, color, thickness)