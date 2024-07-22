import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = width / height
    print(f"Width: {width}, Height: {height}, Aspect Ratio: {aspect_ratio:.2f}")
else:
    print("Error: Unable to open camera.")
cap.release()
