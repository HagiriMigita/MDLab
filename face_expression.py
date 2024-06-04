from feat import Detector
from feat.utils.io import get_test_data_path
import matplotlib.pyplot as plt
import cv2
import os

# 検出器の定義
detector = Detector(landmark_model='mobilefacenet',emotion_model='resmasknet',facepose_model='img2pose')

# 公式が用意した画像のパスを取得
# test_data_dir = get_test_data_path()
# single_face_img_path = os.path.join(test_data_dir, "single_face.jpg")

# 画像を指定して表情認識を実行
size = (600,400)
image = cv2.resize('face_image.jpg', size)
result = detector.detect_image(image)
# result = detector.detect_image(single_face_img_path)

# 結果を出力
result.plot_detections()
plt.show()