import cv2
import numpy as np
from feat import Detector
import tempfile
import matplotlib.pyplot as plt

detector = Detector()

# 顔検出用のカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 貼り付ける画像を読み込み
overlay = cv2.imread('overlay_image.png', -1)  # -1はアルファチャンネル（透明度）も読み込むため

# カメラをキャプチャ
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 顔検出を実行
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image_path = temp_image.name
        cv2.imwrite(temp_image_path, image)
    # 入力画像をPy-Featに渡す
    result = detector.detect_image(temp_image_path)
    emotions = result.emotions.iloc[0]
    print(emotions)

    for (x, y, w, h) in faces:
        # 貼り付ける画像のサイズを顔のサイズに合わせてリサイズ
        overlay_resized = cv2.resize(overlay, (w, h))

        # 貼り付ける画像のアルファチャンネル（透明度）を分離
        overlay_img = overlay_resized[:, :, :3]
        overlay_mask = overlay_resized[:, :, 3:]

        # 顔の領域を選択
        region = frame[y:y+h, x:x+w]

        # 背景と合成
        img1_bg = cv2.bitwise_and(region.copy(), region.copy(), mask=cv2.bitwise_not(overlay_mask))
        img2_fg = cv2.bitwise_and(overlay_img, overlay_img, mask=overlay_mask)

        # 重ね合わせ
        dst = cv2.add(img1_bg, img2_fg)

        # フレームを更新
        frame[y:y+h, x:x+w] = dst

    # フレームを表示
    cv2.imshow('Face Detection', frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
