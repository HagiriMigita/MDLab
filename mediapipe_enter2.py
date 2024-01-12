import cv2
import mediapipe as mp
# MediaPipeのFaceMeshクラスを初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
# カメラからの映像をキャプチャ
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # BGRをRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 顔ランドマーク検出を実行
    results = face_mesh.process(rgb_frame)
    # 検出されたランドマークの描画
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for landmark in landmarks.landmark:
                ih, iw, _ = frame.shape
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    # 結果の表示
    cv2.imshow('Face Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 後処理
cap.release()
cv2.destroyAllWindows()