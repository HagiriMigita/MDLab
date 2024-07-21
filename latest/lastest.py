import cv2
from feat import Detector
import numpy as np
import tempfile
import time  # FPSを測定するためのtimeモジュールをインポート

# Py-FeatのDetectorを初期化
detector = Detector()

# Webカメラから入力
cap = cv2.VideoCapture(0)

# 花火画像の読み込み
fireworks_image = cv2.imread('happiness.png', cv2.IMREAD_UNCHANGED)
sadness_image = cv2.imread('sadness.png', cv2.IMREAD_UNCHANGED)

# 怒りエフェクトの動画読み込み
anger_effect_path = "anger.mp4"
anger_cap = cv2.VideoCapture(anger_effect_path)

if fireworks_image is None:
    print("Error: happiness.png could not be loaded. Please check the file path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

if sadness_image is None:
    print("Error: sadness.png could not be loaded. Please check the file path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

if not anger_cap.isOpened():
    print("Error: anger.mp4 could not be loaded. Please check the file path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# カラーチャンネルを BGR から BGRA に変換
sadness_image = cv2.cvtColor(sadness_image, cv2.COLOR_BGRA2RGBA)

# 画像の貼り付け関数
def paste_image(base_image, overlay_image, x_offset, y_offset):
    h, w = overlay_image.shape[:2]
    base_h, base_w = base_image.shape[:2]
    
    # オーバーレイ画像がベース画像の範囲外に出ないように調整
    if x_offset < 0:
        overlay_image = overlay_image[:, -x_offset:]
        w = overlay_image.shape[1]
        x_offset = 0
    if y_offset < 0:
        overlay_image = overlay_image[-y_offset:, :]
        h = overlay_image.shape[0]
        y_offset = 0
    if x_offset + w > base_w:
        overlay_image = overlay_image[:, :base_w - x_offset]
        w = overlay_image.shape[1]
    if y_offset + h > base_h:
        overlay_image = overlay_image[:base_h - y_offset, :]
        h = overlay_image.shape[0]

    # アルファチャンネルのブレンド処理
    alpha_s = overlay_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        base_image[y_offset:y_offset+h, x_offset:x_offset+w, c] = \
        (alpha_s * overlay_image[:, :, c] + alpha_l * base_image[y_offset:y_offset+h, x_offset:x_offset+w, c])

# 花火画像の貼り付け関数
def paste_fireworks(image, fireworks, num_fireworks=3):
    for _ in range(num_fireworks):
        y_offset = np.random.randint(0, image.shape[0] - fireworks.shape[0])
        x_offset = np.random.randint(0, image.shape[1] - fireworks.shape[1])
        paste_image(image, fireworks, x_offset, y_offset)

frame_count = 0  # フレームカウンター
effect_duration = 30  # エフェクトを表示するフレーム数（約1秒）
happiness_timer = 0  # 幸福感のタイマー
sadness_timer = 0  # 悲しみのタイマー
anger_timer = 0  # 怒りのタイマー
eyes_center = (0, 0)  # 目の中心位置の初期化

# FPS計算用のタイマーとフレームカウンターを追加
start_time = time.time()
fps_frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 入力画像をRGB形式に変換
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    frame_count += 1  # フレームカウンターを増加
    fps_frame_count += 1  # FPS計算用のフレームカウンターを増加

    # 幸福感を表示するタイマーが動いている場合は花火を描画
    if happiness_timer > 0:
        image.flags.writeable = True
        paste_fireworks(image, fireworks_image)
        happiness_timer -= 1

    # 怒りを表示するタイマーが動いている場合はエフェクトを描画
    if anger_timer > 0:
        ret, anger_frame = anger_cap.read()
        if not ret:
            anger_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, anger_frame = anger_cap.read()
        
        # グリーンスクリーンを透過
        anger_frame = cv2.cvtColor(anger_frame, cv2.COLOR_BGR2BGRA)
        hsv = cv2.cvtColor(anger_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        anger_frame[mask != 0, 3] = 0
        
        # アルファチャンネル付きの画像をRGBに変換して表示
        anger_frame = cv2.cvtColor(anger_frame, cv2.COLOR_BGRA2RGBA)
        
        image.flags.writeable = True
        paste_image(image, anger_frame, 0, 0)
        anger_timer -= 1
    
    # 悲しみを表示するタイマーが動いている場合は涙画像を描画
    if sadness_timer > 0:
        image.flags.writeable = True
        paste_image(image, sadness_image, eyes_center[0] - sadness_image.shape[1] // 2, eyes_center[1] - sadness_image.shape[0] // 2)
        sadness_timer -= 1
    
    # 60フレームに1回表情認識を行う
    if frame_count % 60 == 0:
        # 画像を一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            temp_image_path = temp_image.name
            cv2.imwrite(temp_image_path, image)
        
        # 入力画像をPy-Featに渡す
        result = detector.detect_image(temp_image_path)
        emotions = result.emotions.iloc[0]
        landmarks = result.landmarks.iloc[0]
        print(emotions)
        
        # 表情が幸福感であればエフェクトを描画
        if 'happiness' in emotions and emotions['happiness'] > 0.5:  # 0.5は幸福感の閾値
            happiness_timer = effect_duration
        
        # 表情が悲しみであれば目と目の間に画像を貼り付け
        if 'sadness' in emotions and emotions['sadness'] > 0.4:  # 0.4は悲しみの閾値
            sadness_timer = effect_duration
            # 目の中心位置を計算
            left_eye_x = int(landmarks[['x_36', 'x_37', 'x_38', 'x_39', 'x_40', 'x_41']].mean())
            left_eye_y = int(landmarks[['y_36', 'y_37', 'y_38', 'y_39', 'y_40', 'y_41']].mean())
            right_eye_x = int(landmarks[['x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47']].mean())
            right_eye_y = int(landmarks[['y_42', 'y_43', 'y_44', 'y_45', 'y_46', 'y_47']].mean())
            eyes_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)

        # 表情が怒りであればエフェクトを描画
        if 'anger' in emotions and emotions['anger'] > 0.5:  # 0.5は怒りの閾値
            anger_timer = effect_duration

    # FPSの計算と表示
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = fps_frame_count / elapsed_time
        fps_frame_count = 0
        start_time = time.time()
        cv2.putText(image, f"FPS: {fps:.2f}", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))

    # 入力画像をBGR形式に戻す
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 画面に表示
    cv2.imshow('Happiness and Sadness Effects', cv2.flip(image, 1))

    # キー入力を待ち
    if cv2.waitKey(5) & 0xFF == 27:
        break

# リソースを解放
cap.release()
anger_cap.release()
cv2.destroyAllWindows()
