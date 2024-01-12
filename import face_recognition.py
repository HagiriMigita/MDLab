import face_recognition
import cv2
# カメラのインデックス（通常は0）
camera_index = 0
# カメラを起動
video_capture = cv2.VideoCapture(camera_index)
while True:
    # フレームをキャプチャ
    ret, frame = video_capture.read()
    # 顔の位置とランドマークを検出
    face_locations = face_recognition.face_locations(frame)
    face_landmarks_list = face_recognition.face_landmarks(frame)
    # フレームをクリア
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame[:, :, 3] = 0
    # 顔に枠を描画および各ランドマークの座標を表示
    for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 各パーツの座標を表示
        for landmark_name, landmark_points in face_landmarks.items():
            for point in landmark_points:
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
    # 結果を表示
    cv2.imshow('Video', frame)
    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# カメラをリリースしてウィンドウを閉じる
video_capture.release()
cv2.destroyAllWindows()