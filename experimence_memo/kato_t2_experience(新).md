# 実験メモ/作業メモ
## 環境構築
- Ubuntu 22.04.1 LTS 
- CUDA 11.4 
- Pytorch 1.12.1
```bush
conda create -n mmdet3 python=3.10
conda activate mmdet3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip3 install openmim 
mim install "mmcv==2.1.0" "mmdet==3.2.0"
pip3 install tensorboard
git clone https://github.com/open-mmlab/mmdetection.git
```

- CUDAのVersionを変更できないので、11.4(11.3)に合わせて各パッケージをインストールしたが以下のエラーが発生し、demo.pyの実行が不可能。
- PyTorchもCUDA11.4に対応したバージョンは作成されておらず、11.3に対応したバージョンがそのまま11.4に互換性がある。
```bush
Traceback (most recent call last):
  File "/work/kato.t2/mmdetection/demo/image_demo.py", line 61, in <module>
    from mmengine.logging import print_log
ModuleNotFoundError: No module named 'mmengine'
```
- mmengineがインストールされていることは確認済であり、mim installやpip installでのインストールも確認したが、依然変わらずエラーが表示される。
-また、別問題で以下のエラーが表示される。
```bush
WARNING: Error parsing dependencies of distro-info: Invalid version: '0.23ubuntu1'
WARNING: Error parsing dependencies of python-debian: Invalid version: '0.1.36ubuntu1'
```
- 検索しても同条件の警告が発生しているものが発見できなかった。

## 環境構築 11/12
- MMDitekutionの公式サイトとスタートアップに従って環境構築中
- 従ってインストールしていると、mmcvのバージョンを指摘された、mmcvのバージョンは2.0.0以上、2.2.0未満。
- ルートディレクトリ上でもmmcvのインストールでエラーが発生、C++が実行され、Python.hが見つからないという旨のエラー。
- mimでインストールすると、上記のエラーが発生するが、pipでインストールすると発生せず、正常なインストールが可能。
1. We need to download config and checkpoint files.
```
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```
2. Verify the inference demo.
```
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
- 実行後、./output/visが作成され、物体の検出画像が入ってる。


## 実験メモ
- mmdetection/toolsのtrain.pyをターミナルで引数付きで実行していく。
````
※最小単位
python ./tools/train.py ../models/yolox/yolox_tiny_8x8_300e_coco.py
````

### 11/19
- INFERENCE WITH EXISING MODELS内のVideo demoを実行した。outputとしてresult.mp4が出力された。
- 研究内容である遠赤外線画像云々へつなげるために何をすればいいのかがわからない。

## コンフィグの解説
```
model = dict(
    type='MaskRCNN',  # 検出器の名前
    data_preprocessor=dict(  # データ前処理の設定。通常、画像の正規化やパディングが含まれる
        type='DetDataPreprocessor',  # データ前処理の種類。詳細は https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.data_preprocessors.DetDataPreprocessor を参照
        mean=[123.675, 116.28, 103.53],  # 事前トレーニングされたバックボーンモデルに使用される平均値（R、G、B の順）
        std=[58.395, 57.12, 57.375],  # 事前トレーニングされたバックボーンモデルに使用される標準偏差（R、G、B の順）
        bgr_to_rgb=True,  # 画像を BGR から RGB に変換するかどうか
        pad_mask=True,  # インスタンスマスクをパディングするかどうか
        pad_size_divisor=32),  # パディングされた画像サイズは `pad_size_divisor` の倍数でなければならない
    backbone=dict(  # バックボーンの設定
        type='ResNet',  # バックボーンネットワークの種類。詳細は https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.backbones.ResNet を参照
        depth=50,  # バックボーンの深さ。通常、ResNet や ResNeXt の場合は 50 または 101
        num_stages=4,  # バックボーンの段階数
        out_indices=(0, 1, 2, 3),  # 各段階で生成される出力特徴マップのインデックス
        frozen_stages=1,  # 最初の段階の重みは固定されている
        norm_cfg=dict(  # 正規化レイヤーの設定
            type='BN',  # 正規化レイヤーの種類。通常は BN（Batch Normalization）または GN（Group Normalization）
            requires_grad=True),  # BN の γ と β を学習するかどうか
        norm_eval=True,  # BN の統計を固定するかどうか
        style='pytorch',  # バックボーンのスタイル。'pytorch' は 3x3 Conv に stride 2 のレイヤーを持ち、'caffe' は 1x1 Conv に stride 2 のレイヤーを持つ
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 事前トレーニング済みの ImageNet バックボーンをロード
    neck=dict(
        type='FPN',  # 検出器のネックは FPN（Feature Pyramid Network）。'NASFPN'、'PAFPN' などもサポートされる。詳細は https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.necks.FPN を参照
        in_channels=[256, 512, 1024, 2048],  # 入力チャンネル。これはバックボーンの出力チャンネルと一致する
        out_channels=256,  # ピラミッド特徴マップの各レベルの出力チャンネル
        num_outs=5),  # 出力スケールの数
    rpn_head=dict(
        type='RPNHead',  # RPN ヘッドの種類は 'RPNHead'。'GARPNHead' などもサポートされる。詳細は https://mmdetection.readthedocs.io/en/latest/api.html#mmdet.models.dense_heads.RPNHead を参照
        in_channels=256,  # 各入力特徴マップの入力チャンネル。これはネックの出力チャンネルと一致する
        feat_channels=256,  # ヘッド内の畳み込みレイヤーの特徴チャンネル
        anchor_generator=dict(  # アンカー生成器の設定
            type='AnchorGenerator',  # 多くの方法で使用される AnchorGenerator。SSD 検出器は `SSDAnchorGenerator` を使用。詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/anchor_generator.py#L18 を参照
            scales=[8],  # アンカーの基本スケール。特徴マップ上のある位置のアンカーの面積は scale * base_sizes
            ratios=[0.5, 1.0, 2.0],  # 高さと幅の比率
            strides=[4, 8, 16, 32, 64]),  # アンカー生成器のストライド。これは FPN の特徴ストライドと一致する。base_sizes が設定されていない場合はストライドが base_sizes として扱われる
        bbox_coder=dict(  # 学習およびテスト中にボックスをエンコードおよびデコードするためのボックスコーダーの設定
            type='DeltaXYWHBBoxCoder',  # ボックスコーダーの種類。'DeltaXYWHBBoxCoder' は多くの方法で適用される。詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py#L13 を参照
            target_means=[0.0, 0.0, 0.0, 0.0],  # ボックスのエンコードおよびデコードに使用されるターゲットの平均
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # ボックスのエンコードおよびデコードに使用される標準偏差
        loss_cls=dict(  # 分類ブランチの損失関数の設定
            type='CrossEntropyLoss',  # 分類ブランチの損失タイプ。FocalLoss などもサポートされる。詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/cross_entropy_loss.py#L201 を参照
            use_sigmoid=True,  # RPN は通常 2 クラス分類を行うので、シグモイド関数を使用する
            loss_weight=1.0),  # 分類ブランチの損失の重み
        loss_bbox=dict(  # 回帰ブランチの損失関数の設定
            type='L1Loss',  # 損失の種類。IoU 損失や smooth L1 損失などもサポートされる。詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/smooth_l1_loss.py#L56 を参照
            loss_weight=1.0)),  # 回帰ブランチの損失の重み
    roi_head=dict(  # RoIHead は 2 段階/カスケード検出器の第 2 段階をカプセル化
        type='StandardRoIHead',
        bbox_roi_extractor=dict(  # バウンディングボックス回帰用の RoI 特徴抽出器
            type='SingleRoIExtractor',  # RoI 特徴抽出器の種類。ほとんどの方法で SingleRoIExtractor を使用。詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py#L13 を参照
            roi_layer=dict(  # RoI レイヤーの設定
                type='RoIAlign',  # RoI レイヤーの種類。DeformRoIPoolingPack や ModulatedDeformRoIPoolingPack もサポートされる。詳細は https://mmcv.readthedocs.io/en/latest/api.html#mmcv.ops.RoIAlign を参照
                output_size=7,  # 特徴マップの出力サイズ
                sampling_ratio=0),  # RoI 特徴抽出時のサンプリング比率。0 は適応比率を意味する
            out_channels=256,  # 抽出された特徴の出力チャンネル
            featmap_strides=[4, 8, 16, 32]),  # マルチスケール特徴マップのストライド。バックボーンの構造と一致する必要がある
        bbox_head=dict(  # RoIHead のボックスヘッドの設定
            type='Shared2FCBBoxHead',  # ボックスヘッドの種類。詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L220 を参照
            in_channels=256,  # RoI 特徴抽出器の出力チャンネルと一致
            fc_out_channels=1024,  # 出力全結合層のチャンネル数
            roi_feat_size=7,  # RoI 特徴抽出器の出力サイズと一致
            num_classes=80,  # COCO のデフォルトクラス数。実行するデータセットに基づいて設定
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,  # ボックスの回帰がクラス非依存かどうか
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,  # 多クラス分類でソフトマックスを使用
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(  # マスクヘッドの RoI 特徴抽出器
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(  # RoIHead のマスクヘッド
            type='FCNMaskHead',  # 詳細は https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L135 を参照
            num_convs=4,  # コンボリューションレイヤーの数
            in_channels=256,  # 入力チャンネル
            conv_out_channels=256,  # 出力チャンネル
            num_classes=80,  # マスクのクラス数
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))  # 損失関数はクロスエントロピー損失。use_mask=True でマスクを適用

```
# 12/10
```
\\aka\work\tanimoto.j\workspace\fir_domain_adaptation\experiments\TFTRAIN03
```
- これが谷本先輩の提案手法、この中のmain.pyで実験ができる。

```
\\aka\data\its\flir
```
- この中にfir_domain_adaptationのデータが入っているので、自分の\\red\kato.t2とかにインストールするとサーバーがやばいので、dataファイル以下が推奨。

```
\\aka\data\its\flir\FLIR_ADAS_1_3\train
```
- ここに実験に使う画像ファイルが存在する。refined_daytime_RGB_annotations.jsonはRGBファイルの中の画像を自動でアノテーションしてくれる優れもの。

# メモ
- red内で作業しているのは、Ubuntu20.04を使用するためですよ。

- mimはpython3.12に非対応なので、また仮想環境を作り直してくださいね

- torchをちゃんとGPU対応版にしてくださいね。
(chatgptが提供してきたAnacondaでのインストールは、Pytorch公式が非推奨としているもの)

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# 2025/04/21 ICTミーティング
- v1、v2はFLIRのデータセット(https://www.flir.com/oem/adas/adas-dataset-form/)
- LOSS関数はMMDetectionの公式でLOSS関数のカスタマイズが提供されているので、それをちゃんと見よう！(https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_losses.html)
- 現在、研究テーマが迷子なので、それを含めて少し話す必要があると思います。
- sum,and,or,IoU,
- とりあえず、谷本先輩の修士論文を読んでみよう！

# 環境構築(2025/04/21)
- 基本的にはMMDetectionの公式サイトに準拠。(https://mmdetection.readthedocs.io/en/latest/get_started.html)
1. 仮想環境の構築
```
conda create --name openmmlab python=3.10 -y
```
2. Pytorchのインストール
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. MIMを使ってMMEingineとMMCVをインストール
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
```
4. MMDetectionのインストール
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
### main.py実行時のエラー
- 谷本先輩のmain.pyを実行した結果、mmengineのコードを変更する必要が発生。
- /home1/kato.t2/miniconda3/envs/openmmlab05/lib/python3.10/site-packages/mmengine/runner/checkpoint.pyの中でload_from_local()関数の中で
```
checkpoint = torch.load(filename, map_location=map_location)
```
となっているので、下記のように修正。
```
checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
```

# 2025/04/22 道満先生GM
- main.pyを実行は出来たが、その検出精度が谷本先輩の修士論文の結果と一致しているのかを比較。
- 人と車の検出精度の偏りは谷本先輩の頃から存在した。またFLIRのv2へのバージョンアップは、谷本先輩も敬遠していたので、何かしらの障壁が存在する。
- main.pyのコンフィグの挿入の仕方で精度が改善する可能性もあるので、各コンフィグの理解が必要←めっちゃ時間かかりそう。

# 2025/05/25 試行錯誤中
- 段々理解できるようになってきたが、まだ頭が痛い。
- mmdetection/mmdet/models/losses/task_modules/assigners/sim_ota_assinger.pyとかいうめちゃくちゃ深いところまで編集した。怖い。
- LOSSの役割も理解してきたがまだ時間が掛かりそう。
- FLIRのV2への移行が一番時間かかりそうだし、滅茶苦茶難しそう。つらい。

# 2025/05/26 
- 提案手法は確かにTFTRAIN01を使用したものだったが、最終的に谷本先輩が実験していたのはTFTRAIN03だった。
- TFTRAIN03はFTを行わない仕組みになっている←提案手法ではない。
- 損失関数を使った方が検出精度が低い\
![TFTRAIN04の検出精度(損失関数使用)](/markdown_image/検出精度TFTRAIN04_FocalLoss.png)

# 2025/05/27
- 損失関数など、変更して実験に使いたい場合はyolox_x,yolox_s.pyの記述のみを変更すれば良い。あくまでyolox_x_8x8_coco.py等はbeseとして使われているだけなので、yolox_x.py,yolox_s.pyの記述が優先されるため。
- TFTRAIN04を新設、損失関数実装したものがTFTRAIN04になります。
- TFTRAIN03の方でエラーが発生、なぜ……
## DomanGMまとめ
- 検出精度結果に関しては、比較のしやすいように並べて表示をする。
- スクリーンショットではなく、表形式で表示する。
- ドローン等、検出する物体が小さい写真での検出を研究している論文を参考にすると良い。Lossに関して見る。
- 月曜日のミーティングに関することもGMで記入する。

# 2025/05/29
- 谷本先輩の使用していた実行ファイルのbaseファイルであるyolox_x_8x8_300e_coco.py、yolox_s_8x8_300e_coco.pyにもすでにbboxにはIoULoss、クラス分類にはCrossEntropyLoss、オブジェクト性にもCrossEntropyLossが使用されていた。これらについてより良い損失関数や、パラメーターを変更することによってモデルの精度向上に繋がるかどうかを研究テーマにするべきか。要相談。

# 2025/06/02
- focal_lossの重みファイルが出力されないエラーが発生中。コードを変更したが改善するかは実行してファイルが生成されるかを確認するしかないです。

# 2025/06/02 ITSミーティング
- 損失関数でもセグメンテーション用だったり、アノテーション用だったり、クラス不均衡だったり、エッジが不明瞭なときの為の損失関数があるので色々調べながら試行錯誤していきましょう。
- クラス数の変更については、80をいきなり2に変えるのはダメ。
- 谷本先輩がやろうとしていたのは損失関数も式から考えていく。既存の損失関数ではない。←頑張ろう

# 2025/06/04
```
exec_list = [
    'test_for_pseudo_label_train_data',                          # ① 既存モデルで RGB 画像に対する予測を実行（疑似ラベル元）
    'generate_pseudo_labeled_train_annotation',                  # ② 出力された予測を JSON に変換（疑似ラベルを COCO 形式に）
    'generate_pseudo_labeled_train_annotation_using_sam2',       # ③ SAM2 でマスク処理と bbox 整形（より正確な疑似ラベルに）
    'train_with_pseudo_label_used_sam2',                         # ④ 整形済み疑似ラベルで Thermal 画像に対するモデル再学習
    'test_pseudo_label_used_sam2_trained_model',                 # ⑤ 学習済みモデルを評価（例：夜間 Thermal 画像など）
]
```
- この状態が好ましい。

# 2025/06/05

- lossかscore_thresholdの影響か、学習が早期(epoc=10,40)でbestになってしまっている。
- generate_pseudo_labeled_train_annotation_using_sam2_0.5に変更。
- FocalLossに関しても、gamma(難易度調整)=1.5に修正し、難しいサンプルに過剰に重みがかからないようにする。

- focal_lossがalphaが2つのクラスに付与することができなかったため、修正。

# 2025/06/06
- 谷本先輩の提案手法＆threshold_score=0.50の検出精度は以下の表の通り。

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l | 
| :--------: | :-----: | :------: | :------: | :-----: | :-----: | :-----: | 
| person   | 0.024 | 0.108  | 0.001  | 0.027 | 0.072 | 0.100 | 
| car      | 0.299 | 0.567  | 0.293  | 0.119 | 0.553 | 0.758 | 

-現在、疑似ラベル作成→SAM2(0.50)でアノテーション→FocalLoss(クラス不均衡に配慮したalpha）＆SmoothLossで学習

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l | 
| :--------: | :-----: | :------: | :------: | :-----: | :-----: | :-----: | 
| person   | 0.37 | 0.771  | 0.3  | 0.245 | 0.452 | 0.601 | 
| car      | 0.304 | 0.647  | 0.238| 0.116 | 0.401 | 0.649 |

- 比較すると一目瞭然だが、personの検出精度が著しく向上している。ただalphaの影響か、carの検出精度が低下している。

- 現在、yolox_xでもyolox_sでも、早期に検出精度のピークが来てしまっているので、score_thresholdやlrなどを編集する必要あり。
score_thresoldは疑似ラベルの閾値なので、0.7以上にして疑似ラベルの信頼性を向上させる。
lrについては0.0025が良いらしい。

- YOLOXを研究してるおっちゃんたちもこう言ってる。
~~~
We use stochastic gradient descent (SGD) for training. We use a learning rate of
lr×BatchSize/64 (linear scaling [8]), with a initial lr = 0.01 and the cosine lr schedule
~~~
```math
Ir = 0.01 × BatchSize / 64
```
がええよ、と。

- score_thresold=0.7、学習率を上に倣っても結局早期で学習のピークに達してしまう現象は変わらなかった。
しかし検出精度はうなぎ登り←いいことなのか？


```
score_thresold=0.7,lr=0.00125,YOLOX_X
    exec_list = [
        'test_for_pseudo_label_train_data',
        'generate_pseudo_labeled_train_annotation',
        'generate_pseudo_labeled_train_annotation_using_sam2_0.7',
        'train_with_pseudo_label_used_sam2_0.7',
        'test_pseudo_label_used_sam2_trained_model',
    ]
```


あなた:
- score_thresold=0.7、学習率を上に倣っても結局早期で学習のピークに達してしまう現象は変わらなかった。
しかし検出精度はうなぎ登り←いいことなのか？


score_thresold=0.7,lr=0.00125,YOLOX_X
    exec_list = [
        'test_for_pseudo_label_train_data',
        'generate_pseudo_labeled_train_annotation',
        'generate_pseudo_labeled_train_annotation_using_sam2_0.7',
        'train_with_pseudo_label_used_sam2_0.7',
        'test_pseudo_label_used_sam2_trained_model',
    ]


| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.404 | 0.819 | 0.345 | 0.276 | 0.491 | 0.655 |
| car      | 0.395 | 0.761 | 0.372 | 0.185 | 0.455 | 0.737 |


# 2025/06/09 ICTミーティング
- 学習率は決して数式で決まるものではないので、繰り返し実験して自分のデータセットにあった学習率にするべき。
- データセットの位置合わせ自体は愛工大の卒業生の方がやってくだっていたので、それを使う。
- 損失関数についても、色々試して、傾向をつかむ。数値だけではなく、描画もして何を拾えていないのかも見て傾向を知る。

- testを実行していなかったので、実行してみた。
- checkpoint='experiments/TFTRAIN04/yolox_x/train_with_pseudo_label_used_sam2_0.7_20250606_054835/best_coco_bbox_mAP_50_epoch_10.pth'

| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|---------|------|--------|--------|-------|-------|-------|
| person  | 0.404 | 0.819 | 0.345 | 0.276 | 0.491 | 0.655 |
| car     | 0.395 | 0.761 | 0.372 | 0.185 | 0.455 | 0.737 |　

# 2025/06/09 ICTミーティング
- 学習率は決して数式で決まるものではないので、繰り返し実験して自分のデータセットにあった学習率にするべき。
- データセットの位置合わせ自体は愛工大の卒業生の方がやってくだっていたので、それを使う。
- 損失関数についても、色々試して、傾向をつかむ。数値だけではなく、描画もして何を拾えていないのかも見て傾向を知る。

- testを実行していなかったので、実行してみた。
- checkpoint='experiments/TFTRAIN04/yolox_x/train_with_pseudo_label_used_sam2_0.7_20250606_054835/best_coco_bbox_mAP_50_epoch_10.pth'

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| person   | 0.069 | 0.236  | 0.21 | 0.058 | 0.167 | 0.118 | 
| car      | 0.294 | 0.638  | 0.253| 0.143 | 0.532 | 0.761 |

# 2025/06/17
- 一週間研究を休んだので改めてやるべきことを洗い出す。
- FLIRのデータセットをv1からv2に変更するのが一番分かりやすい変更点になると思う。
- 損失関数に関しては、まだ触らなくていい気がする。


# 2025/06/24
```
export PYTHONPATH=/work/kato.t2/fir_domain_adaptation/SAM2/sam2
```
- これを実行してパスを追加しないと正常にmain.pyが実行できません。

## MekadaGMまとめ
- MMDetectionでLossがどう使われているかで、loss_clsやloss_bbox、loss_objの数値の評価が変わる。
- loss_objについては距離に関する損失なので、数値が1より大きくなる可能性がある。重みを考えなければならない。故にloss_bboxが支配的な学習を行っている可能性が高い。
- 実験において比較をする際は変更要素を1つに絞り、実験を行った方がどの要素が実験においてプラスになるかわかる。
- 小さいオブジェクトに対して強い方法を調べる。 

# 2025/07/08
- とりあえず、loss_weight=1.0,1.0,0.1の実験結果を出したい。
- MMdetectionのlossesにgfocal_loss.pyがあった。その中にQualityFocalLossとDistributionFocalLossが存在しており、QFocalLossがおすすめらしい。
- QFocalLossの目的:分類スコアにIoUなどの品質（Quality）を取り入れた分類損失
- ターゲットは「クラスラベル」＋「品質スコア（通常IoU）」
- 確信度（score）が高いほど損失を強くし、低いと弱くする（ノイズ耐性あり）
````
# 例：pred = モデルのlogits (N, C)
# target = (class_label: (N,), confidence_score: (N,))
loss = QualityFocalLoss(beta=2.0, loss_weight=1.0)
loss(pred, target)
````

# 2025/07/14
- ハイパーパラメータチューニングを知らなかったので、調べる。
- 清水先輩が谷本先輩に連絡をとってくれるらしいのでv2_dataset_toolsについて進められるかも。そのためにも、現在のv1についての構造も把握しておくべき。
- 損失関数の研究にするならやっぱり独自の損失関数を設計するべき。
- loss関数の式を見て、パラメータを変えて、どの部分が有力なのか。項が増える等、係数が増えたり、損失関数を作る。
- FocalLossの式は場当たり的な設計なので、もっとコンパクトな損失関数も見るべき。
- 0から損失関数を作るのは難しいので、既存の損失関数の美味い部分を参考にして作ったら良い。

#2025/08/04
- 現在のオリジナルのデータセットだと画像サイズが大き過ぎる、全ての画角を一致させる必要がある、時間方向にずれているペアも多いので評価に使えない。
    - 現在やっているエラーとの格闘は無駄
- v2のデータセットを使うためにも矢上さんに連絡する必要アリ
- 清水先輩にdiscord教えてもらう
- 矢神さんから谷本先輩に伝えた内容もそこにある
- データ数が増えるとできること
    - 10枚に対して付ける付けないのと、1000枚に対して付ける付けないのとでは100倍の差がある。
- データ数が増えることによって精度が向上するのか、余地はあるのかの確認→先行研究の問題点
- sam2は遠赤外線画像に使用してる
- 遠赤外線画像の解像度と解像感が問題点だった
    - 先鋭化とかかけたら変わるかもしれない
- 谷本先輩の修士論文をもう一度読んで問題点を洗い流す→slackでまとめて
- データセット拡充できればそれ中間発表で言える
- 検出率が低いシーンが挙げられていれば、方向性を決めるヒントになる

# 2025/08/07
- 現在v2のtrainとvalの写真はclip済
- FLIRのv2フォルダに入っているcoco.jsonをclip済の写真に対応するように、除外された写真のアノテーション情報をcoco.jsonから除外。coco.jsonをrefined_rgb_annotations.json(thermalも)に改造する。
- testの写真はclipする必要はあるのか？(おそらくない)
- FLIR_ADAS_v2について
    - trainとvalについてはclip済のdataが揃いました。
    - v1のときにあったrefined_rgb_annotations.jsonはv2におけるcoco.jsonをclip後のファイルに合わせて作ったcoco形式のjsonファイルだと推測できるので、coco.jsonを改造します。

# 2025/09/09
testの検出精度比較

### 谷本手法 (v1)
| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.034 | 0.149 | 0.002 | 0.034 | 0.076 | 0.101 |
| car      | 0.274 | 0.514 | 0.268 | 0.096 | 0.538 | 0.734 |


### 谷本手法 (v2)
| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.045 | 0.148 | 0.020 | 0.036 | 0.146 | 0.004 |
| car      | 0.242 | 0.471 | 0.223 | 0.123 | 0.419 | 0.454 |


### 加藤手法 (v1)
| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.065 | 0.236 | 0.018 | 0.056 | 0.181 | 0.119 |
| car      | 0.321 | 0.668 | 0.269 | 0.146 | 0.542 | 0.802 |


### 加藤手法 (v2)
| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.101 | 0.303 | 0.041 | 0.091 | 0.247 | 0.098 |
| car      | 0.424 | 0.754 | 0.401 | 0.311 | 0.533 | 0.847 |


# 2025/10/21
- 道満GMにて、データセット拡張の方向性が決定。v1+v2で、アノテーションファイルの形式をv1に合わせる。
- "occluded"はそもそも使ってなさそうなので、削除します

# 2025/11/03
- v2の画像の中で、ペアが存在して、なおかつそれが日中の写真であるもの。RGBが645枚、Thermalが139枚。
- 枚数が一致しない理由は不明。対応した画像ファイルが存在しない場合、谷本先輩のコードだとエラーが起きる可能性。

# 2025/11/05 目加田GM
- 従来手法(v1)、提案手法(v1)→損失関数の有用性
- 従来手法(v1)、提案手法(v1+v2)→データ拡張の有用性
- 提案手法(v1+v2)→損失関数とデータ拡張の組み合わせの有用性

- loss_clsに関して、FocalLossを使っても車と人で形状も何も違うので、重み的にもFocalLossの性質的にもそこまで重要じゃない
- それよりも背景と物体、bboxの精度に関する損失関数を重要視した方が良い

- v1のRGB→FIRの射影変換行列とv2のRGB→FIRの射影変換行列は違うので、変更する必要あり

- v1で学習→v2でテスト等でデータセットの傾向の明示→v1+v2への理由付け

- optunaでハイパーパラメーター探索

- データセットの詳しい数字をまとめる
    - person、carのアノテーションの数、比率
    - RGBとFIRの枚数、比率

# 2025/11/14

### v1+v2 → v1
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.044 | 0.156 | 0.015 | 0.039 | 0.138 | 0.128 |
| car      | 0.249 | 0.540 | 0.201 | 0.100 | 0.514 | 0.732 |


### v1+v2 → v2
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.180 | 0.374 | 0.149 | 0.144 | 0.542 | nan |
| car      | 0.125 | 0.306 | 0.092 | 0.059 | 0.457 | 0.641 |


### v1+v2 → v1+v2
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.077 | 0.224 | 0.040 | 0.074 | 0.212 | 0.143 |
| car      | 0.159 | 0.368 | 0.122 | 0.065 | 0.474 | 0.672 |


どのデータセットにおいてもv1+v2の学習の結果はひどいもの  

### v2 → v1
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.072 | 0.229 | 0.026 | 0.063 | 0.185 | 0.110 |
| car      | 0.401 | 0.749 | 0.372 | 0.282 | 0.541 | 0.761 |


### v2 → v2
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.235 | 0.433 | 0.226 | 0.199 | 0.593 | nan |
| car      | 0.261 | 0.509 | 0.247 | 0.184 | 0.561 | 0.652 |


### v2 → v1+v2
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.115 | 0.292 | 0.070 | 0.106 | 0.287 | 0.105 |
| car      | 0.300 | 0.573 | 0.282 | 0.203 | 0.547 | 0.702 |


### v1 → v1
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.036 | 0.135 | 0.008 | 0.034 | 0.100 | 0.172 |
| car      | 0.274 | 0.552 | 0.245 | 0.121 | 0.534 | 0.721 |
 

### v1 → v2
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.174 | 0.368 | 0.139 | 0.139 | 0.536 | nan |
| car      | 0.138 | 0.308 | 0.115 | 0.070 | 0.506 | 0.547 |


### v1 → v1+v2
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.070 | 0.204 | 0.033 | 0.066 | 0.187 | 0.171 |
| car      | 0.179 | 0.379 | 0.154 | 0.082 | 0.513 | 0.621 |



# 2025/11/16
- 結局v2が動かなくなるのはpairs.jsonとconverter.pyがけんかするから。
- 結局エラーが解消されたと思ったらmAPが著しく低い現象が再発。
- yolox_sは9月5日時点のgenerate_pseudo_label_train_annotation_using_sam2_0.5_20250905_020253を使うことによって学習は可能→損失関数によるmAPの向上の実験はできるが、データセットによる精度の改善の実験はできない状況になってしまった。


# 2025/11/17
- score_thresould = 0.7 の場合、class毎の総数はperson 531個、car 1696個 で2227個
- ![score_threshould=0.7の場合の各bboxのスコア分布](/experimence_memo/markdown_image/score__Histgram.png)
```
bbox_head=dict(
        act_cfg=dict(type='Swish'),
        feat_channels=128,
        in_channels=128,
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
)
```
| Loss                    | タスク             | 役割                                | 主なパラメータ                              | パラメータの解説                                                                 |
|------------------------|------------------|-------------------------------------|--------------------------------------------|----------------------------------------------------------------------------------|
| **loss_bbox (IoULoss)** | bbox regression   | IoU ベースで bbox の重なりを最大化       | `eps`, `mode`, `loss_weight`, `reduction`  | **eps**：ゼロ除算防止の微小値（学習安定化）。<br>**mode='square'**：IoU 差を二乗し微調整性能UP。<br>**loss_weight=5.0**：bbox の重要度を強調。<br>**reduction='sum'**：ミニバッチ合計。 |
| **loss_cls (BCE)**      | クラス分類          | person/car のクラス分類                | `use_sigmoid`, `loss_weight`, `reduction`  | **use_sigmoid=True**：softmax でなく BCE。YOLO 系定番。<br>**loss_weight=1.0**：分類タスクの重み。<br>**reduction='sum'**：分類誤差を合計。 |
| **loss_obj (BCE)**      | objectness       | 物体の有無（背景 / 前景）を判定            | `use_sigmoid`, `loss_weight`, `reduction`  | **use_sigmoid=True**：obj も BCE。<br>**loss_weight=1.0**：objectness の重要度。<br>**reduction='sum'**：ミニバッチ全体の obj 誤差。 |
| **loss_l1 (L1Loss)**    | bbox 補助回帰       | IoU の弱点（高 IoU 時の微調整）を補う       | `loss_weight`, `reduction`                 | **loss_weight=1.0**：補助回帰の重み。<br>**reduction='sum'**：微調整用の L1 を合計。                         |


- 現在使用している損失関数のパラメータ
```
model = dict(
    bbox_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.60,
            loss_weight=1.0),
        loss_obj=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.60,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(
            type='CIoULoss',
            loss_weight=5.0)
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
    )
)
```

| Loss                     | タスク             | 役割                                   | 主なパラメータ                                         | パラメータの解説                                                                                         |
|-------------------------|------------------|----------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **loss_bbox (CIoULoss)** | bbox regression   | 距離・重なり・縦横比のズレを総合的に最適化 | `loss_weight`                                         | **CIoU**：IoU＋中心距離＋アスペクト比を同時最適化。高精度。<br>**loss_weight=5.0**：bbox 回帰を強調。 |
| **loss_cls (FocalLoss)** | クラス分類          | クラス不均衡の補正 / hard example 強調     | `gamma`, `alpha`, `use_sigmoid`, `loss_weight`        | **gamma=4.0**：簡単なサンプルの影響を強く減衰（難例に焦点）。<br>**alpha=0.60**：正例の重み付け。<br>**use_sigmoid=True**：BCE型分類。 |
| **loss_obj (FocalLoss)** | objectness       | 前景 / 背景を判定（物体がいるか）           | `gamma`, `alpha`, `reduction`, `loss_weight`          | **gamma=4.0**：背景の圧倒的多数を抑制し、前景の学習を強化。<br>**alpha=0.60**：正例（物体）のウェイト増強。<br>**reduction='mean'**：ロス平均化。 |
| **loss_l1 (L1Loss)**     | bbox 補助回帰       | CIoU の微調整能力を補完                     | `loss_weight`, `reduction`                            | **L1**：bbox の微細な位置合わせの補助用。<br>**reduction='sum'**：ミニバッチ全体の L1 誤差を合算。     |

- 目加田先生のアドバイスから、loss_clsの重みをもっと減らして良い。またFocalLossを使用する必要性はない。
- v1とv2の射影変換についての問題も解消した。v1の除外リストがv2にも適用されてしまう状態も解決し、アノテーションの個数も正常になった。
- ただv1+v2では、v1とv2の画像データを区別できない問題で、精度がそれぞれ学習したときよりも著しく低下している。
- 現在optunaで最適なパラメータの探索を行っている。
- 先行研究(谷本先輩)では、アノテーションのスコアで足切りを行っており、その値は0.5になっていた。
- 時間はないが、自分なりの損失関数を作ってみたい。

- /work/kato.t2/fir_domain_adaptation/experiments/TFTRAIN03_loss/yolox_s/generate_pseudo_labeled_train_annotation_using_sam2_0.5_20251114_023001/pred_train_data_annotation_fir_bbox_arranged.jsonにおけるクラス毎のannnotatation数
- person:4909、car:13453

- /work/kato.t2/fir_domain_adaptation/experiments/TFTRAIN04/yolox_s/generate_pseudo_labeled_train_annotation_using_sam2_0.5_20251117_024253/pred_train_data_annotation_fir_bbox_arranged_0.5.jsonにおけるクラス毎のannotation数
- person:1026、car:2422


# 2025/11/18 LT
- annotationにおけるscoreというものは尤度なのか確信度なのか
- 今まで、thermalの画像データをhours:dayのものだけをデータセットとして読み込んでいたが、その場合欠落しているものが拾われずにthermalだけ枚数が少ない状況だった→pairs.jsonとrgb_annotations.jsonからrgbとの対応付けとhoursの情報を抽出→rgbと同じく645枚の画像データ取得に成功

- optunaでパラメータ探索をした結果、以下の結果になった。
- /work/kato.t2/fir_domain_adaptation/experiments/TFTRAIN04/optuna_trials/251118_013323/trial_0026
```
lr          : 0.008899016425117203
weight_decay: 0.00031475782484270535
cls_gamma   : 3.4280302119685513
cls_alpha   : 0.6948062368002415
cls_loss_w  : 1.9549504884157618
obj_gamma   : 3.3649360915495468
obj_alpha   : 0.6224107342709038
obj_loss_w  : 2.5924295633290706
bbox_loss_w : 2.9726769921601677
```
- 最良mAPの計算式としては、personとcarのmAP_50の平均
- それが最も大きいもののパラメータを選択

# 2025/11/21
- converter.pyと依存関係にあるプログラム
````
tools/convert_annotation_fir_to_vis.py
tools/convert_annotation_vis_to_fir.py
tools/convert_annotation_vis_to_fir_v1+v2.py
tools/convert_annotation_vis_to_fir_v2.py
tools/merge_result_annotaiton.py
tools/refine_fir_annotation.py
tools/symbolic_link.py
````

- bbox_converter.pyと依存関係にあるプログラム
````
converter.py
````

- 学習時のmAPとして、v1はやはりpersonのmAP_sが小さく、v2はpersonのmAP_sが大きい(比較的)
- v1も決してpersonのSmallのannotaitonの数が少ないわけではない
- carのmAPはv1の方が圧倒的に大きい

- v2のサイズ別アノテーションの数
````
| category_id | Small | Medium | Large | Total |
|-------------|-------|--------|-------|-------|
| 1 | 9504 | 1135 | 20 | 10659 |
| 3 | 14967 | 2600 | 417 | 17984 |
| all | 24471 | 3735 | 437 | 28643 |
annotation_count_to_get: 27000
annotation_count: 27000
score_to_set: 0.011462477967143059
````

- v1のサイズ別アノテーションの数
````
| category_id | Small | Medium | Large | Total |
|-------------|-------|--------|-------|-------|
| 3 | 60336 | 13957 | 1968 | 76261 |
| 1 | 23170 | 5083 | 367 | 28620 |
| all | 83506 | 19040 | 2335 | 104881 |
annotation_count_to_get: 27000
annotation_count: 27000
score_to_set: 0.25729888677597046
````

- v1のsam2整形後

| category_id | Small | Medium | Large | Total |
|-------------|-------|--------|-------|-------|
| car | 5611 | 6425 | 1417 | 13453 |
| person | 1762 | 2784 | 363 | 4909 |
| all | 7373 | 9209 | 1780 | 18362 |

annotation_count_to_get: 27000  
annotation_count: 18362  
score_to_set: 0.5000089406967163  

- v2のsam2整形後

| category_id | Small | Medium | Large | Total |
|-------------|-------|--------|-------|-------|
| car | 951 | 1121 | 350 | 2422 |
| person | 452 | 551 | 23 | 1026 |
| all | 1403 | 1672 | 373 | 3448 |

annotation_count_to_get: 27000  
annotation_count: 3448  
score_to_set: 0.5003814101219177  

- v1+v2のsam2整形後

| category_id | Small | Medium | Large | Total |
|-------------|-------|--------|-------|-------|
| car | 7254 | 6612 | 1466 | 15332 |
| person | 2801 | 2534 | 334 | 5669 |
| all | 10055 | 9146 | 1800 | 21001 |

annotation_count_to_get: 100000  
annotation_count: 21001  
score_to_set: 0.5000089406967163  

# 2025/11/26
- 現在は先行研究のTFTRAIN03の損失関数のパラメータ探索を実行中。
- 私の卒論のコアはデータセットの更新&拡張、損失関数の更新。損失関数の最適化。これらのあらゆる結果を比較する。
- 先行研究と提案手法の比較
    - TFTRAIN03(谷本手法,default_loss)におけるv1,v2,v1+v2の結果比較→データセットの比較
    - TFTRAIN03(optimized_default_loss),TFTRAIN03_loss(optimized_FocalLoss,CIoULoss)の比較→損失関数の比較
    - TFTRAIN03(default_loss),TFTRAIN03(optimized_default_loss)の比較→損失関数の最適化の影響の比較
    - TFTRAIN03(v1,谷本手法,default_loss)と、TFTRAIN04(v2,optimized_FocalLoss,CIoULoss)orTFTRAIN05(v1+v2,optimized_FocalLoss,CIoULoss)(提案手法)の精度の比較→先行研究と提案手法の比較

- 現在すべきこと
    - v1に最適化されたdefault_lossのパラメータ探索(実行中)
    - v1+v2における射影変換行列の切り替えの対策

- 問題点
    - v1+v2における射影変換行列の切り替えをどうするか
        - 新規のconverter.py、bbox_converter.pyを作成して、v1+v2を実行するときのみ、それらにパスを繋げる
        - 既存のプログラムに変更を加えて、v1、v2、v1+v2の切り替えを自動で行う。

# 2025/11/28
- convert_annotation_vis_to_firをv1,v2,v1+v2用にそれぞれ用意する。
- TFTRAIN03,04,05はそれぞれv1,v2,v1+v2学習用の実行ファイルと位置付ける。
- またconvert_annotaion_vis_to_firを3つに増やすことに併せて、converter.pyやbbox_converter.pyも3つに増やす必要がある。
- 上記の理由としては、v1,v2,v1+v2においてそれぞれ別の射影変換を行う必要があるからである。
    - v1のみ→射影変換が必要。
    - v2のみ→pair.pyにbbox情報があるので、converter.py内に射影変換を行う処理が必要ない。
    - v1+v2→v1のデータにのみ射影変換、v2のデータには何も行わない。

# 2025/11/29
- TFTRAIN05のoptunaによるパラメータ探索が可能になった。(結果が出るのにはちょっと時間かかる)
- TFTRAIN05の一連の実験が可能になった。

- v1+v2→v1+v2(パラメータチューニング前)

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.071 | 0.257  | 0.016  | 0.064 | 0.222 | 0.177 |
| car      | 0.112 | 0.318  | 0.067  | 0.024 | 0.348 | 0.629 |

- v1+v2→v2

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.133 | 0.377  | 0.052  | 0.097 | 0.473 | nan   |
| car      | 0.095 | 0.276  | 0.052  | 0.025 | 0.374 | 0.61  |

- v1+v2→v1

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.045 | 0.194  | 0.006  | 0.041 | 0.11  | 0.177 |
| car      | 0.162 | 0.439  | 0.108  | 0.029 | 0.329 | 0.67  |

- Todo
    - v1のbaseloss(best)のtrain、testを行う
    - v1+v2のパラメータ探索、train、testを行う
- Done
    - v1のbeselossのパラメータ探索
    - v1のnewlossすべて

### v1（デフォルト損失）→ v1テスト
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.027 | 0.127 | 0.002 | 0.032 | 0.080 | 0.089 |
| car      | 0.278 | 0.554 | 0.255 | 0.112 | 0.520 | 0.726 |

### v1(デフォルト損失・チューニング後)　→ v1テスト

| category | mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.04 | 0.149  | 0.007  | 0.038 | 0.125 | 0.174 |
| car      | 0.32 | 0.61   | 0.292  | 0.134 | 0.56  | 0.778 |

### v1(デフォルト損失・チューニング後) → v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.17  | 0.346  | 0.147  | 0.135 | 0.515 | nan   |
| car      | 0.171 | 0.362  | 0.156  | 0.082 | 0.501 | 0.719 |


### v1(デフォルト損失・チューニング後) → v1+v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.072 | 0.205  | 0.035  | 0.072 | 0.201 | 0.171 |
| car      | 0.216 | 0.432  | 0.198  | 0.095 | 0.532 | 0.74  |

### v1（新損失）→ v1テスト
| category |  mAP  | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.036 | 0.135 | 0.008 | 0.034 | 0.100 | 0.172 |
| car      | 0.274 | 0.552 | 0.245 | 0.121 | 0.534 | 0.721 |

### v2(新損失) → v1テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|:--------:|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| person   | 0.073 | 0.226  | 0.03   | 0.062 | 0.203 | 0.079 |
| car      | 0.397 | 0.734  | 0.379  | 0.263 | 0.549 | 0.786 |


### v1（新損失・チューニング後）→ v1テスト
| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.077 | 0.247  | 0.026  | 0.066 | 0.168 | 0.285 |
| car      | 0.359 | 0.671  | 0.346  | 0.226 | 0.513 | 0.774 |

### v1（新損失・チューニング後) → v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.215 | 0.441  | 0.171  | 0.179 | 0.573 | nan   |
| car      | 0.224 | 0.461  | 0.201  | 0.145 | 0.522 | 0.675 |

### v1 （新損失・チューニング後） → v1+v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.113 | 0.304  | 0.059  | 0.104 | 0.264 | 0.3   |
| car      | 0.263 | 0.517  | 0.244  | 0.164 | 0.515 | 0.706 |

### v2 （新損失・チューニング後） → v1テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.108 | 0.31   | 0.045  | 0.096 | 0.265 | 0.228 |
| car      | 0.376 | 0.697  | 0.359  | 0.239 | 0.526 | 0.787 |

### v2 （新損失・チューニング後） → v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.233 | 0.411  | 0.233  | 0.197 | 0.618 | nan   |
| car      | 0.25  | 0.484  | 0.234  | 0.166 | 0.561 | 0.684 |

### v2 （新損失・チューニング後） → v1+v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.12  | 0.29   | 0.081  | 0.109 | 0.334 | 0.042 |
| car      | 0.292 | 0.546  | 0.276  | 0.187 | 0.551 | 0.724 |

### v1+v2(新損失・チューニング後) → v1テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.054 | 0.215  | 0.012  | 0.046 | 0.133 | 0.191 |
| car      | 0.16  | 0.475  | 0.09   | 0.038 | 0.331 | 0.674 |

### v1+v2(新損失・チューニング後) → v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.136 | 0.387  | 0.052  | 0.098 | 0.475 | nan   |
| car      | 0.086 | 0.284  | 0.033  | 0.027 | 0.341 | 0.655 |

### v1+v2(新損失・チューニング後) → v1+v2テスト

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.08  | 0.274  | 0.021  | 0.066 | 0.231 | 0.19  |
| car      | 0.106 | 0.333  | 0.05   | 0.027 | 0.332 | 0.653 |



|データセット|総数 (Count)|平均スコア (Mean)|標準偏差 (Std Dev)|安定性 (Std Dev)|
|-----|-----|-----|-----|-----|
|V1|18,362|0.747385|0.123170|最も安定|
|V2|3,448|0.748813|0.125183|やや不安定|
|V1+V2|21,001|0.747532|0.123631|V1に次いで安定|


- パラメータ一覧
1. TFTRAIN03(v1学習・デフォルト損失)
    - lr=1.068213756942332e-05
    - weight decay=7.431072761843534e-05
    - loss_cls:CrossEntoropyLoss
        - loss_weight=3.003815407821085
    - loss_obj:CrossEntoropyLoss(use_sigmoid=True=BinaryCE) # BCEは二値分類→前景,背景
        - loss_weight=3.035053838639484
    - loss_bbox:IoULoss
        - eps=5.8231429152995255e-09 # オーバーフロー,アンダーフロー対策
        - loss_weight=1.9993266871961406
    - loss_l1:L1Loss
        - loss_weight=1.0

2. TFTRAIN03_loss(v1学習・新損失)
    - lr:1.77624618498687e-05
    - wd:1.597200655882034e-06
    - loss_cls:FocalLoss
        - gamma=3.244636272823687
        - alpha=0.6731822518615557
        - loss_weight=2.751252206378675
    - loss_obj:FocalLoss
        - gamma=2.8026441139840594
        - alpha=0.5444941291038482
        - loss_weight=1.2896376748450793
    - loss_bbox:CIoUloss
        - eps=1e-6
        - loss_weight=3.394385434435689
    - loss_L1:L1Loss
        - loss_weight=1.0

3. TFTRAIN04(v2学習・新損失)
    - lr:0.007072671327811989
    - wd:6.841163741638921e-06
    - loss_cls:FocalLoss
        - gamma=3.5064929481205923
        - alpha=0.21917302497098703
        - loss_weight=2.731299202759068
    - loss_obj:FocalLoss
        - gamma=1.057832042857803
        - alpha=0.15323777830922947
        - loss_weight=2.8762737277483446
    - loss_bbox:CIoUloss
        - eps=1e-6
        - loss_weight=8.827897927967747
    - loss_L1:L1Loss
        - loss_weight=1.0

4. TFTRAIN05(v1+v2学習・新損失)
    - 探索中

# 2025/12/03
- 目加田GMでの問題点
    - GTのannotationが正しいのか、疑似ラベルが正しいのか
        - 疑似ラベルにおいてスコアが著しく低いもので、元画像を参照したところ、infoにhourでdayと書かれているのにも関わらず、夜の画像だった。
        - 改めてRGBとThermalのGTを比較、bbox座標はズレなし。
        - 疑似ラベルは大きく外れているものもある。(hour:dayにも関わらず)
        - sam2整形前の疑似ラベルで比較したので、sam2は悪くない。
        - やはり事前学習モデルが原因。
    - scoreの理解
        - 結局のところ、scoreはあくまで事前学習モデルの確信度合い(感想)であり、事前学習モデルのannotationとGTのannotaitonのIoUスコアではない。すなわち、scoreが高ければ高いほど精度が高くなるわけではない。IoUスコアであれば、検出精度は高くなるでしょうねぇ。
        - しかし、研究の前提上、可視光学習済みモデルから疑似ラベルを作成するという流れなので、何とも言えない。
        - 疑似ラベル作成までにGTが介入するタイミングはない→疑似ラベルに入っているbboxはGTのbboxとは全く別物になる。(GTでアノテーションされていない物体も疑似ラベルではアノテーションされている)
        - 例としては、GTではpersonとして認識されておらず、アノテーションされていないかったキックボードに載っている人を疑似ラベルではアノテーションされている。
        - 誤検出かと言われると微妙なライン。キックボードに載っている人間を検出できるくらいには事前学習モデルがしっかりと人を人として認識できていると解釈できる。
        それがロバスト性に繋がるかもしれない。（ただGTとの方針が異なるので、最終的な検出精度は低くなる）

# 2025/12/25
- 最終ミーティング
    - v1からv2に移行する理由が弱い
        - 一応v1のときはRGBのannotaitonファイルが存在しなかった。v2にはオフィシャルのannotationファイルが存在する。
        - あくまで私たちの実験には関係ない。事前学習モデルが作成した可視光のラベルの精度評価には使えるが、最終的な検出結果には何も関係ない。
        - v1のときはRGBのannotaitonファイルが存在しなかったので、可視光のラベルの評価が行えなかった。
        - 評価を行えば、v1からv2に移行した正当性が生まれる。
        - 可視光のラベルの検出精度と、最終的な遠赤外線の検出精度を比較

- test_for_pseudo_label_train_data_20251204_084443の結果

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.121 | 0.260  | 0.100  | 0.116 | 0.277 | 0.100 |
| car      | 0.405 | 0.644  | 0.422  | 0.227 | 0.692 | 0.828 |

- test_for_pseudo_label_train_data_20251129_032641の結果

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.121 | 0.260  | 0.100  | 0.116 | 0.277 | 0.100 |
| car      | 0.405 | 0.644  | 0.422  | 0.227 | 0.692 | 0.828 |

- train_with_pseudo_label_used_sam2_0.5_20251129_032641のテスト結果

| category | mAP   | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
|----------|-------|--------|--------|-------|-------|-------|
| person   | 0.227 | 0.431  | 0.21   | 0.189 | 0.595 | nan   |
| car      | 0.251 | 0.511  | 0.219  | 0.163 | 0.566 | 0.723 |

- Thermalにはpersonのlargeが存在しないのでnanになります。

# 2025/01/04
- pHashでv1とv2の画像を比較したところ
    - v1のtestとv2のtrainで、約20枚ほど完全一致の画像が存在した。

- FLIR_ADAS_v1
    - daytime_RGB_train = 4,358枚
    - daytime_FIR_train = 4,358枚
    - daytime_FIR_val = 755枚
    - FIR_test = 4,224枚