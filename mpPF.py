import cv2
import mediapipe as mp
from feat import Detector
import matplotlib.pyplot as plt
import os

detector = Detector()

# MediaPipeの関連モジュールのインポート
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
 
# Webカメラから入力
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# FaceMeshモデルを読み込む
with mp_face_mesh.FaceMesh(
    max_num_faces=1,                         # 検出する最大の顔の数
    refine_landmarks=True,                   # ランドマークのリファインを行うかどうか
    min_detection_confidence=0.5,            # 最小の検出信頼度
    min_tracking_confidence=0.5) as face_mesh:  # 最小の追跡信頼度
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    # 入力画像をRGB形式に変換
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #入力画像をPy-Featに渡す
    result = detector.detect_image(image)

    result.plot_detections()
    plt.show()

    # FaceMeshモデルに画像を入力し、顔のメッシュを検出
    results = face_mesh.process(image)
 
    # 検出された顔のメッシュをカメラ画像の上に描画
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # 顔のメッシュを描画
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        # 顔の輪郭を描画
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        # 目の周りのメッシュを描画
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    
    # 画面に表示
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    # キー入力を待ち、ESCが押されたらループを抜ける
    if cv2.waitKey(5) & 0xFF == 27:
      break

# リソースを解放
cap.release()
cv2.destroyAllWindows()
