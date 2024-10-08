# [interface](https://interface.cqpub.co.jp/) 2024年 10月号向けコード

1. [姿勢推定情報の抽出](#姿勢推定情報の抽出)
2. [特徴量の抽出と人物照合](#特徴量の抽出と人物照合)
3. [モデル別類似度参考](experiments.md)
4. [指定推定速度参考](estimate_pose_speed.md)

# 動作環境セットアップ

## python依存関係の解決
```
pip install -r requirements.txt
```
動作確認バージョンを念のため付記していますが、特にバージョン依存等はありません（試行時取得バージョン）。

# 姿勢推定情報の抽出

以下のコードおよびモデルを利用しています。
- [MoveNet-Python-Example](https://github.com/Kazuhito00/MoveNet-Python-Example) : estimate_pose.py の母体コードとして修正。onnxモデルの取得

## モデル配備

### MoveNetモデル取得
```
mkdir onnx
cd onnx
wget https://github.com/Kazuhito00/MoveNet-Python-Example/raw/main/onnx/movenet_singlepose_lightning_4.onnx
wget https://github.com/Kazuhito00/MoveNet-Python-Example/raw/main/onnx/movenet_singlepose_thunder_4.onnx
```

## 重畳動画とCSVファイルの取得
重畳動画とCSVファイルはコード実行パス配下の`debugs`、`csv`にそれぞれ出力されます。

### コード実行例

個別ファイル処理（`interface_videos/A_1.mp4`からのCSVファイルと重畳動画の出力をmovenet lightningで実行する場合）
```
python estimate_pose.py --file interface_videos/A_1.mp4 --keypoint_score 0.0 --debug_output --csv --model_select 0
```

一括処理個別ファイル処理（`interface_videos/`配下の動画ファイル群をmovenet thunderでCSV出力する場合）
```
python estimate_pose.py --file interface_videos --keypoint_score 0.0 --csv --model_select 1
```


### 重畳動画参考

![img.png](img.png)

### CSVファイル内容参考
```
cat csv/A_1.csv | more
frame_number,nose_x,nose_y,nose_conf,left_eye_x,left_eye_y,left_eye_conf,right_eye_x,right_eye_y,right_eye_conf,left_ear_x,left_ear_y,left_ear_conf,right_ear_x,right_ear_y,right_ear_conf,left_shoulder_x,left_shoulder_y,left_shoulder_conf,right_shoulder_x,right_shoulder_y,right_shoulder_conf,left_elbow_x,left_elbow_y,left_elbow_conf,right_elbow_x,right_elbow_y,right_elbow_conf,left_wrist_x,left_wrist_y,left_wrist_conf,right_wrist_x,right_wrist_y,right_wrist_conf,left_hip_x,left_hip_y,left_hip_conf,right_hip_x,right_hip_y,right_hip_conf,left_knee_x,left_knee_y,left_knee_conf,right_knee_x,right_knee_y,right_knee_conf,left_ankle_x,left_ankle_y,left_ankle_conf,right_ankle_x,right_ankle_y,right_ankle_conf
0,1707.044906616211,347.13303565979004,0.5577347,1740.3987121582031,331.7494297027588,0.5438053,1705.4011917114258,330.67259788513184,0.5129675,1810.3191375732422,344.8819434642792,0.72115636,1716.6682434082031,341.0364603996277,0.20362222,1850.3754043579102,452.4284613132477,0.32734326,1675.806770324707,440.48311471939087,0.5521757,1881.3070678710938,576.907045841217,0.28962392,1658.0922317504883,552.0480537414551,0.33124334,1866.544532775879,686.0744762420654,0.37768215,1613.5624694824219,640.2298808097839,0.41503298,1796.254005432129,679.5848393440247,0.6814122,1674.1096115112305,675.563714504242,0.45122808,1690.6593704223633,854.6500897407532,0.6855285,1707.575798034668,838.4335613250732,0.4098196,1662.7059173583984,1022.9281067848206,0.60179585,1867.5423431396484,936.2403774261475,0.39444983
1,1693.2710266113281,339.3857431411743,0.5227892,1719.704246520996,325.92512011528015,0.6958655,1685.0078201293945,325.1396405696869,0.49133983,1786.003761291504,335.5732190608978,0.48921174,1696.6831970214844,333.51189851760864,0.33621696,1835.9571075439453,442.78119921684265,0.4933002,1659.327049255371,431.2903583049774,0.73642945,1869.0966796875,567.4644255638123,0.28256628,1639.5893096923828,544.279453754425,0.36098075,1857.0172119140625,673.7013387680054,0.24937102,1602.751579284668,630.2297258377075,0.4217462,1770.3433227539062,669.0708589553833,0.4987685,1659.5563888549805,665.8269095420837,0.723692,1689.6710586547852,847.5977683067322,0.68911743,1676.5421676635742,832.9516196250916,0.45346546,1663.7420654296875,1018.1538820266724,0.59359264,1844.1403198242188,925.3553509712219,0.47075364
2,1673.4619903564453,334.35235261917114,0.5627546,1701.5557479858398,321.4990246295929,0.46223894,1671.1626434326172,317.3306465148926,0.5142007,1769.234275817871,330.8448922634125,0.60693914,1681.1811447143555,326.2068486213684,0.203102,1810.041618347168,440.5553412437439,0.4920682,1647.1482467651367,425.4498589038849,0.6510734,1865.4904174804688,579.2385721206665,0.17245033,1629.8583984375,539.8716723918915,0.32584858,1831.0900497436523,681.2336897850037,0.1805256,1587.1317672729492,626.8554854393005,0.35964042,1740.406265258789,669.6361827850342,0.5150162,1634.7118377685547,664.6569299697876,0.7307576,1687.2341537475586,840.3052711486816,0.48294976,1634.7976684570312,822.8045654296875,0.6378468,1655.416259765625,1015.7953834533691,0.4894848,1782.1746826171875,913.1846022605896,0.4745463
3,1654.6905899047852,331.4412760734558,0.5958133,1683.1229782104492,316.35674715042114,0.5854769,1650.40283203125,314.18126106262207,0.4561541,1747.7503967285156,323.6445128917694,0.63235223,1664.5801162719727,318.9341139793396,0.28798142,1783.2415008544922,437.08261013031006,0.5739326,1632.707862854004,420.7133889198303,0.56784976,1830.300636291504,582.6969480514526,0.20550892,1614.6986389160156,539.6674489974976,0.3772845,1790.9864044189453,693.0182647705078,0.26382068,1567.6762390136719,627.6965832710266,0.38687187,1715.0405502319336,673.5465860366821,0.5621389,1612.5452041625977,665.2123403549194,0.71649873,1687.2699737548828,847.190158367157,0.5825044,1613.1024169921875,826.7172861099243,0.63352615,1669.5453643798828,1016.6966700553894,0.41682708,1706.451644897461,949.4786238670349,0.23366505
:
```

## 外れ値の補正とスムージング（参考）
CSVファイルに出力された姿勢推定情報を更新します。

以下実行では`smooth`配下にスコア下限値を下回る点群を線形補間したCSVが出力されます（今回は利用していません）。

スムージングアルゴリズムは`--filter`オプションで`butterr`または`suvgol`を指定できます（デフォルトは`None`で補正なし）。

### コード実行例
```
python smooth.py --file csv/ --output smooth/
```

## 姿勢推定の点群可視化（参考）
`visualize_pose.ipynb`を`jupyter`より実行し、任意のCSVファイルを選択して可視化ができます。

![animation.gif](animation.gif)

# 特徴量の抽出と人物照合

## 歩容特徴

以下の特徴量を用いて照合します。

- 左腕の振り（横）：左手首の腰原点にしたときのX方向の最大と最小の差の最大値
- 左腕の振り（縦）：左手首の腰原点にしたときのY方向の最大と最小の差の最大値
- 右腕の振り（横）：右手首の腰原点にしたときのX方向の最大と最小の差の最大値
- 右腕の振り（縦）：右手首の腰原点にしたときのY方向の最大と最小の差の最大値 
- 足のひざ上がり方（左）：左ひざの腰原点にしたときのY方向の最大と最小の差の最大値- 足のひざ移動（右）：右ひざの腰原点にしたときのX方向の最大と最小の差の最大値 
- 足のひざ移動（左）：左ひざの腰原点にしたときのX方向の最大と最小の差の最大値
- 足のひざ上がり方（右）：右ひざの腰原点にしたときのY方向の最大と最小の差の最大値 
- 足のひざ移動（右）：右ひざの腰原点にしたときのX方向の最大と最小の差の最大値
- 歩幅（左）：左足首の腰原点にしたときのX方向の最大と最小の差の最大値
- 足の上がり方（左）：左足首の腰原点にしたときのY方向の最大と最小の差の最大値- 足の上がり方（左）：左足首の腰原点にしたときのY方向の最大と最小の差の最大値
- 歩幅（右）：右足首の腰原点にしたときのX方向の最大と最小の差の最大値
- 足の上がり方（右）：右足首の腰原点にしたときのY方向の最大と最小の差の最大値 
- 頭の上下：左目のY方向の最大と最小の差の最大値
- 頭の移動：左目のX方向の最大と最小の差の最大値

## 処理概要

- 任意のディレクトリより、任意のCSVファイル接尾（デフォルト値:`_1.csv`）を持たないCSVファイルをすべて姿勢推定結果の点群として読み込みます。
- 姿勢推定結果の点群から、左右の足首距離を計算し、フーリエ変換し周期（整数に繰り上げ）を得ます。 
- 得た周期を用いて各特長（最大値）を計算し、取得したすべての値を特徴として格納します。
- 格納した特徴を任意のCSVから得られる特徴を用いて人物照合（cos類似度計算）し出力します。

## コード実行例
```
python gait_feature.py --csv_path csv --query_path csv/A_1.csv            
| label | similarity | file_path |
| - | - | - |                                           
| A | 0.7450044234012938 | csv\A_2.csv |   
| A | 0.4521558599142875 | csv\A_4.csv |   
| D | 0.29694326155269396 | csv\D_5.csv |  
| A | 0.20545524107553953 | csv\A_3.csv |  
| B | 0.19316869080360857 | csv\B_2.csv |  
| B | 0.05426617801114495 | csv\B_4.csv |  
| D | 0.024470875058635266 | csv\D_2.csv | 
| D | 0.008471934158318227 | csv\D_4.csv |
| A | -0.011680579291395527 | csv\A_5.csv |
| D | -0.04707788577775721 | csv\D_3.csv | 
```
