import cv2
import numpy as np
import openvino.runtime as ov

# ターゲットデバイスの指定
ie = ov.Core()
plugin = ie.get_plugin(device="MYRIAD")

# モデルの読み込み（顔検出）
net = ie.read_network(model='FP16/face-detection-retail-0004.xml', weights='FP16/face-detection-retail-0004.bin')
exec_net = plugin.load(network=net)

# モデルの読み込み（感情分類）
net_emotion = ie.read_network(model='FP16/emotions-recognition-retail-0003.xml', weights='FP16/emotions-recognition-retail-0003.bin')
exec_net_emotion = plugin.load(network=net_emotion)

# カメラ準備
cap = cv2.VideoCapture(0)

# メインループ
while True:
    ret, frame = cap.read()

    # Reload on error
    if ret == False:
        continue

    # 入力データフォーマットへ変換
    img = cv2.resize(frame, (300, 300))   # サイズ変更
    img = img.transpose((2, 0, 1))    # HWC > CHW
    img = np.expand_dims(img, axis=0) # 次元合せ

    # 推論実行
    out = exec_net.infer(inputs={'image': img})

    # 出力から必要なデータのみ取り出し
    out = out['detection_out']
    out = np.squeeze(out) #サイズ1の次元を全て削除

    # 検出されたすべての顔領域に対して１つずつ処理
    for detection in out:
        # conf値の取得
        confidence = float(detection[2])

        # バウンディングボックス座標を入力画像のスケールに変換
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示
        if confidence > 0.5:
            # 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > frame.shape[1]:
                xmax = frame.shape[1]
            if ymax > frame.shape[0]:
                ymax = frame.shape[0]

            # 顔領域のみ切り出し
            frame_face = frame[ymin:ymax, xmin:xmax]

            # 入力データフォーマットへ変換
            img_emotion = cv2.resize(frame_face, (64, 64))   # サイズ変更
            img_emotion = img_emotion.transpose((2, 0, 1))    # HWC > CHW
            img_emotion = np.expand_dims(img_emotion, axis=0) # 次元合せ

            # 推論実行
            out_emotion = exec_net_emotion.infer(inputs={'data': img_emotion})

            # 出力から必要なデータのみ取り出し
            out_emotion = out_emotion['prob_emotion']
            out_emotion = np.squeeze(out_emotion) #不要な次元の削減

            # 出力値が最大のインデックスを得る
            index_max = np.argmax(out_emotion)

            # 各感情の文字列をリスト化
            list_emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger']

            # 文字列描画
            cv2.putText(frame, list_emotion[index_max], (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

            # バウンディングボックス表示
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)

            # １つの顔で終了
            break

    # 画像表示
    cv2.imshow('frame', frame)

    # 何らかのキーが押されたら終了
    key = cv2.waitKey(1)
    if key != -1:
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
