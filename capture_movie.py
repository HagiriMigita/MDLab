import cv2
import time
from feat import Detector

detector = Detector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした。")
    exit()

fps = 0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Web Camera",frame)

        frame_count += 1
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            with open("frame_rate.txt", 'a') as f:
                print(f"フレームレート : {fps:.2f} FPS", file = f)
            start_time = time.time()
            frame_count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 