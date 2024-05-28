import cv2
import numpy as np 

cap = cv2.VideoCapture(0)

fmt = cv2.VideoWriter_fourcc('m','p','4','v')
fps = 30
size = (1000,600)

write = cv2.VideoWriter('WebCam.m4v', fmt, fps, size)

while True:
    ret, frame = cap.read()

    frame =cv2.resize(frame, size)
    write.write(frame)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == 13:
        break

write.release()
cap.release()
cv2.destroyAllWindows()