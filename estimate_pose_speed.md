# 参考処理速度

| Platform                                         | model                         | FPS     | provider |
|--------------------------------------------------|-------------------------------|---------|----------|
| ThinkPad T480 i5-8250U (Ubuntu 22.04)            | movenet_singlepose_lighting_4 | 42.509  | CPU      |
| ThinkPad T480 i5-8250U (Ubuntu 22.04)            | movenet_singlepose_thunder_4  | 16.596  | CPU      |
| ThinkPad T480 GeForce MX150 (Ubuntu 22.04)       | movenet_singlepose_lighting_4 | 67.222  | CUDA     |
| ThinkPad T480 GeForce MX150 (Ubuntu 22.04)       | movenet_singlepose_thunder_4  | 38.682  | CUDA     |
| LifeBook U9311 i5-1145G7  (Windows10 22H2)       | movenet_singlepose_lighting_4 | 66.159  | CPU      |
| LifeBook U9311 i5-1145G7  (Windows10 22H2)       | movenet_singlepose_thunder_4  | 22.490  | CPU      |
| Raspberry Pi5 (RaspberryPi OS March 15th 2024)   | movenet_singlepose_lighting_4 | 19.866  | CPU      |
| Raspberry Pi5 (RaspberryPi OS March 15th 2024)   | movenet_singlepose_thunder_4  | 7.129   | CPU      |
| Jetson AGX Orin (L4T 36.3/Jetpack 6.0)           | movenet_singlepose_lighting_4 | 93.443  | CUDA     |
| Jetson AGX Orin (L4T 36.3/Jetpack 6.0)           | movenet_singlepose_thunder_4  | 92.897  | CUDA     |
| IA Server Intel(R) Xeon(R) W-2125 (Ubuntu 22.04) | movenet_singlepose_lighting_4 | 182.382 | CPU      |
| IA Server Intel(R) Xeon(R) W-2125 (Ubuntu 22.04) | movenet_singlepose_thunder_4  | 73.271  | CPU      |
| IA Server Quadro RTX 6000 (Ubuntu 22.04)         | movenet_singlepose_lighting_4 | 236.902 | CUDA     |
| IA Server Quadro RTX 6000 (Ubuntu 22.04)         | movenet_singlepose_thunder_4  | 161.967 | CUDA     |


## 対象動画

FPS:30 / width 1920 / height 1080 の動画の処理時間を計測。フレームは以下の通り。

| video path                | frames   |
|---------------------------|----------|
| interface_videos/A_1.mp4  | 73       |
| interface_videos/A_2.mp4  | 70       | 
| interface_videos/A_3.mp4  | 88       | 
| interface_videos/A_4.mp4  | 86       |
| interface_videos/A_5.mp4  | 79       |
| interface_videos/B_1.mp4  | 93       |
| interface_videos/B_2.mp4  | 84       |
| interface_videos/B_3.mp4  | 79       | 
| interface_videos/B_4.mp4  | 81       | 
| interface_videos/B_5.mp4  | 71       |
| interface_videos/C_1.mp4  | 99       |
| interface_videos/C_2.mp4  | 107      | 
| interface_videos/C_3.mp4  | 106      | 
| interface_videos/C_4.mp4  | 100      |
| interface_videos/C_5.mp4  | 97       |
| interface_videos/D_1.mp4  | 86       | 
| interface_videos/D_2.mp4  | 95       |
| interface_videos/D_3.mp4  | 89       |
| interface_videos/D_4.mp4  | 87       |
| interface_videos/D_5.mp4  | 86       |

