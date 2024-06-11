import cv2
from feat import Detector
import matplotlib.pyplot as plt

detector = Detector()

def Recording():
    time = 10

    cap = cv2.VideoCapture(0)

    fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    name = "sample.mp4"

    video = cv2.VideoWriter(name, fourcc, fps, (w,h))

    print("start")
    roop = int(fps * time)
    for i in range(roop):
        ret, frame = cap.read()
        video.write(frame)

    print("stop")
    video.release()
    cap.release()
    cv2.destroyAllWindows()

    return video

Recording()

video_prediction = detector.detect_video("sample.mp4", skip_frames=30)
video_prediction.head()

video_prediction.loc[[30, 60]].plot_detections(faceboxes=False, add_titles=False)

axes = video_prediction.emotions.plot()
plt.show()