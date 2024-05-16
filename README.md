# 2024年度 5月 雑誌interface向けコード 現場プロの画像処理100（仮）

## 要件

画像処理，AI処理を分けてカウントしたいと思っています。

1. [姿勢推定情報の抽出](#姿勢推定情報の抽出)
2. 特徴量の抽出
3. 人物照合
4. 上記の際に，どこかで利用した画像補正　（あれば）

# 動作環境セットアップ

## python依存関係の解決
```
pip install -r requriements.txt
```

# 姿勢推定情報の抽出

[MoveNet-Python-Example](https://github.com/Kazuhito00/MoveNet-Python-Example)を修正して利用しています。

## モデル配備
```
$ mkdir onnx
$ cd onnx
$ wget https://github.com/Kazuhito00/MoveNet-Python-Example/raw/main/onnx/movenet_singlepose_lightning_4.onnx
```

