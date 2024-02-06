import cv2
import mediapipe as mp

# Mediapipeの初期化
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# カメラの初期化
cap = cv2.VideoCapture(1)

while cap.isOpened():
    # フレームの読み込み
    ret, frame = cap.read()
    if not ret:
        break

    # 顔検出
    results = face_detection.process(frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # 顔領域にバイラテラルフィルタを適用
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.bilateralFilter(face_roi, d=9, sigmaColor=100, sigmaSpace=100)

            # 元のフレームに処理後の顔領域を反映
            frame[y:y+h, x:x+w] = face_roi

    # フレームの表示
    cv2.imshow('Beauty Filter', frame)

    # 終了条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後片付け
cap.release()
cv2.destroyAllWindows()
