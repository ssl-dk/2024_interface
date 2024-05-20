# 2024年度 5月 雑誌interface向けコード 現場プロの画像処理100（仮）

## 要件

画像処理，AI処理を分けてカウントしたいと思っています。

1. [姿勢推定情報の抽出](#姿勢推定情報の抽出)
2. [特徴量の抽出](#特徴量の抽出)
3. 人物照合
4. 上記の際に，どこかで利用した画像補正　（あれば）

# 動作環境セットアップ

## python依存関係の解決
```
pip install -r requirements.txt
```
動作確認バージョンを念のため付記していますが、特にバージョン依存等はありません（試行時取得バージョン）。

# 姿勢推定情報の抽出

[MoveNet-Python-Example](https://github.com/Kazuhito00/MoveNet-Python-Example)を修正して利用しています。

## モデル配備
```
mkdir onnx
cd onnx
wget https://github.com/Kazuhito00/MoveNet-Python-Example/raw/main/onnx/movenet_singlepose_lightning_4.onnx
```

## 重畳動画とCSVファイルの取得
重畳動画とCSVファイルはコード実行パス配下の`debugs`、`csv`にそれぞれ出力されます。

### コード実行例
```
python movenet_singlepose_onnx.py --file interface_videos/A_1.mp4 --keypoint_score 0.0 --debug_output --csv
```

### 重畳動画参考

![img.png](img.png)

### CSVファイル内容参考
```
cat csv/A_1.csv | more
frame_number,nose_x,nose_y,nose_conf,left_eye_x,left_eye_y,left_eye_conf,right_eye_x,right_eye_y,right_eye_conf,left_ear_x,left_ear_y,left_ear_conf,right_ear_x,right_ear_y,right_ear_conf,left_shoulder_x,left_shoulder_y,left_shoulder_conf,right_shoulder_x,right_shoulder_y,right_shoulder_conf,left_elbow_x,left_elbow_y,left_elbow_conf,right_elbow_x,right_elbow_y,right_elbow_conf,left_wrist_x,left_wrist_y,left_wrist_conf,right_wrist_x,right_wrist_y,right_wrist_conf,left_hip_x,left_hip_y,left_hip_conf,right_hip_x,right_hip_y,right_hip_conf,left_knee_x,left_knee_y,left_knee_conf,right_knee_x,right_knee_y,right_knee_conf,left_ankle_x,left_ankle_y,left_ankle_conf,right_ankle_x,right_ankle_y,right_ankle_conf
0,1721,341,0.42245978,1754,326,0.58175915,1712,327,0.57963926,1813,340,0.53507954,1718,339,0.34348637,1860,441,0.46571508,1676,438,0.47710404,1847,576,0.38557935,1638,550,0.4466851,1835,677,0.20385751,1607,635,0.46187034,1774,676,0.49937767,1697,670,0.42545134,1686,841,0.5424488,1723,835,0.39171275,1650,1020,0.54304403,1855,951,0.45132017
1,1695,341,0.54838777,1730,324,0.6398781,1689,325,0.58653307,1792,334,0.56611574,1699,333,0.36779,1837,439,0.4850201,1666,425,0.3702311,1846,581,0.42863387,1637,543,0.34322596,1795,695,0.29519293,1606,633,0.43505612,1754,677,0.5251442,1672,671,0.5427353,1709,849,0.58842784,1703,837,0.3515209,1667,1032,0.4859239,1796,955,0.48823303
2,1681,336,0.35007125,1711,318,0.45976126,1672,318,0.58253175,1768,332,0.6261512,1674,326,0.28464645,1798,434,0.3731395,1660,423,0.58527696,1856,574,0.35121065,1624,538,0.24173298,1803,697,0.3472525,1591,617,0.47504622,1737,677,0.5850502,1636,667,0.5085149,1692,843,0.5882794,1652,832,0.4899592,1675,1017,0.45458704,1759,943,0.46876293
:
```

# 特徴量の抽出

CSVファイルより以下のような特徴を抽出します。

- 歩行速度
- 歩幅（右足）
- 歩幅（左足）
- 手のふり幅（左手）
- 歩行1サイクルのフレーム数
- 頭の上下運動

