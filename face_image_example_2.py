import cv2
import numpy as np

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

    # 顔検出を実行
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # 顔の領域を取得
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # 表情認識を実行
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotions = analysis['emotion']

            # 幸福感の閾値を確認
            if 'happy' in emotions and emotions['happy'] > 50:  # 幸福感の閾値は50%
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

        except Exception as e:
            print(f"Error: {e}")
            continue

    # フレームを表示
    cv2.imshow('Face Detection', frame)

    # 'q'キーが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
