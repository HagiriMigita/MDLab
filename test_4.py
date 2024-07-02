import cv2
from feat import Detector
import numpy as np
import tempfile
import time
import matplotlib.pyplot as plt


detector = Detector()

cap = cv2.VideoCapture(0)

i = 30

def draw_fireworks(image, num_fireworks=5):
    for _ in range(num_fireworks):
        center = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        for _ in range(20):
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(10, 50)
            end = (int(center[0] + length * np.cos(angle)), int(center[1] + length * np.sin(angle)))
            cv2.line(image, center, end, color, 2)

while cap.isOpened():
  start = time.perf_counter() # 計測開始
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  # 入力画像をRGB形式に変換
  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # 画像を一時ファイルに保存
  with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
      temp_image_path = temp_image.name
      cv2.imwrite(temp_image_path, image)

  if i == 30:
       # 画像を一時ファイルに保存
      with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image_path = temp_image.name
        cv2.imwrite(temp_image_path, image)
      # 入力画像をPy-Featに渡す
      result = detector.detect_image(temp_image_path)
      emotions = result.emotions.iloc[0]
      print(emotions)
      result.plot_detections() # Py-Featの感情推定の結果
      plt.show() # 結果の表示の有無
      i = 0

  i += 1

  # 表情が幸福感であればエフェクトを描画
  if 'happiness' in emotions and emotions['happiness'] > 0.5:  # 0.5は幸福感の閾値
    draw_fireworks(image)
  if 'surprise' in emotions and emotions['surprise'] > 0.5:
     display_surprise(image)

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  height = image.shape[0]
  width = image.shape[1]

  image = cv2.resize(image,(width * 2, height * 2))

  cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
  end = time.perf_counter() #計測終了
  print('{:.2f}'.format((end-start)/60)) # 87.97(秒→分に直し、小数点以下の桁数を指定して出力)

  # キー入力を待ち
  k = cv2.waitKey(1)
  if k == 27:
    break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
