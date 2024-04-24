# LYT-Net-ONNX-Sample
Low-Light Image Enhancementモデルである[albrateanu/LYT-Net](https://github.com/albrateanu/LYT-Net)のONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。変換自体を試したい方は[LYT-Net-Convert2ONNX.ipynb](LYT-Net-Convert2ONNX.ipynb)を使用ください。

https://github.com/Kazuhito00/LYT-Net-ONNX-Sample/assets/37477845/8ad2245d-51e0-44bc-bd9e-a08551feeac2

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.13.0 or later

# Demo
デモの実行方法は以下です。
```bash
python sample.py --movie=night01.mp4
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/lyt_net_lolv2_real_320x240.onnx

# Reference
* [albrateanu/LYT-Net](https://github.com/albrateanu/LYT-Net)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
LYT-Net-ONNX-Sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[雨イメージ　夜の道路を走る車](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002161702_00000)を使用しています。
