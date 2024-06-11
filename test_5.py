import cv2
from feat import Detector
import numpy as np
import tempfile
# Py-FeatのDetectorを初期化
detector = Detector()
# Webカメラから入力
cap = cv2.VideoCapture(0)
# 画像の読み込み
fireworks_image = cv2.imread('happiness.png', cv2.IMREAD_UNCHANGED)
# sadness_image = cv2.imread('sadness.png', cv2.IMREAD_UNCHANGED)
if fireworks_image is None:
    print("Error: hanabi.png could not be loaded. Please check the file path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()
#if sadness_image is None:
    #print("Error: sadness.png could not be loaded. Please check the file path.")
    #cap.release()
    #cv2.destroyAllWindows()
    #exit()
# 花火画像の貼り付け関数
def paste_fireworks(image, fireworks, num_fireworks=3):
    for _ in range(num_fireworks):
        y_offset = np.random.randint(0, image.shape[0] - fireworks.shape[0])
        x_offset = np.random.randint(0, image.shape[1] - fireworks.shape[1])
        for c in range(0, 3):
            image[y_offset:y_offset+fireworks.shape[0], x_offset:x_offset+fireworks.shape[1], c] = \
            fireworks[:, :, c] * (fireworks[:, :, 3] / 255.0) + \
            image[y_offset:y_offset+fireworks.shape[0], x_offset:x_offset+fireworks.shape[1], c] * \
            (1.0 - fireworks[:, :, 3] / 255.0)
# 悲しみ画像の貼り付け関数
#def paste_sadness(image, sadness):
    #y_center = image.shape[0] // 2 - sadness.shape[0] // 2
    #x_center = image.shape[1] // 2 - sadness.shape[1] // 2
    #for c in range(0, 3):
        #image[y_center:y_center+sadness.shape[0], x_center:x_center+sadness.shape[1], c] = \
        #sadness[:, :, c] * (sadness[:, :, 3] / 255.0) + \
        #image[y_center:y_center+sadness.shape[0], x_center:x_center+sadness.shape[1], c] * \
        #(1.0 - sadness[:, :, 3] / 255.0)
frame_count = 0  # フレームカウンター
is_happy = False  # 笑顔フラグ
is_sad = False  # 悲しみフラグ
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # 入力画像をRGB形式に変換
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_count += 1  # フレームカウンターを増加
    # 10フレームに1回表情認識を行う
    if frame_count % 10 == 0:
        # 画像を一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            temp_image_path = temp_image.name
            cv2.imwrite(temp_image_path, image)
        # 入力画像をPy-Featに渡す
        result = detector.detect_image(temp_image_path)
        emotions = result.emotions.iloc[0]
        print(emotions)
        # 表情が幸福感であればフラグをセット
        if 'happiness' in emotions and emotions['happiness'] > 0.5:  # 0.5は幸福感の閾値
            is_happy = True
        else:
            is_happy = False
        # 表情が悲しみであればフラグをセット
        #if 'sadness' in emotions and emotions['sadness'] > 0.5:  # 0.5は悲しみの閾値
            #is_sad = True
        #else:
            #is_sad = False
    # 笑顔フラグがセットされている場合に花火を描画
    if is_happy:
        paste_fireworks(image, fireworks_image)
    # 悲しみフラグがセットされている場合に悲しみ画像を描画
    #if is_sad:
        #paste_sadness(image, sadness_image)
    # 入力画像をBGR形式に戻す
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 画面に表示
    cv2.imshow('Emotion Detection', cv2.flip(image, 1))
    # キー入力を待ち
    if cv2.waitKey(5) & 0xFF == 27:
        break
# リソースを解放
cap.release()
cv2.destroyAllWindows()

