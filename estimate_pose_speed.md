# 参考処理速度

| Platform                                        | model                         | sec/frame (avg)      | onnx-provider |
|-------------------------------------------------|-------------------------------|----------------------|---------------|
| ThinkPad T480 i5-8250U (Ubuntu 22.04)           | movenet_singlepose_lighting_4 | 0.06233941911293325  | CPU           |
| ThinkPad T480 i5-8250U (Ubuntu 22.04)           | movenet_singlepose_thunder_4  | 0.09349038348382589  | CPU           |
| ThinkPad T480 Geforce mx150 (Ubuntu 22.04)      | movenet_singlepose_lighting_4 | 0.02970061174557801  | CUDA          |
| ThinkPad T480 Geforce mx150 (Ubuntu 22.04)      | movenet_singlepose_thunder_4  | 0.04620538526352554  | CUDA          |
| Raspberry Pi5 (RaspberryPi OS March 15th 2024)  | movenet_singlepose_lighting_4 | 0.06858045405841906  | CPU           |
| Raspberry Pi5 (RaspberryPi OS March 15th 2024)  | movenet_singlepose_thunder_4  | 0.14750172119748892  | CPU           |
| Jetson AGX Orin (L4T 36.3/Jetpack 6.0)          | movenet_singlepose_lighting_4 | 0.016188874331585095 | CUDA          |
| Jetson AGX Orin (L4T 36.3/Jetpack 6.0)          | movenet_singlepose_thunder_4  | 0.016434536843745202 | CUDA          |


## 対象動画

| video path               | fps               | width | height | frames |
|--------------------------|-------------------|-------|--------|--------|
| interface_videos/A_1.mp4 | 29.97002997002997 | 1920  | 1080   | 73     |
| interface_videos/A_2.mp4 | 29.97002997002997 | 1920  | 1080   | 70     | 
| interface_videos/A_3.mp4 | 29.97002997002997 | 1920  | 1080   | 88     | 
| interface_videos/A_4.mp4 | 29.97002997002997 | 1920  | 1080   | 86     |
| interface_videos/A_5.mp4 | 29.97002997002997 | 1920  | 1080   | 79     |
| interface_videos/B_1.mp4 | 29.97002997002997 | 1920  | 1080   | 93     |
| interface_videos/B_2.mp4 | 29.97002997002997 | 1920  | 1080   | 84     |
| interface_videos/B_3.mp4 | 29.97002997002997 | 1920  | 1080   | 79     | 
| interface_videos/B_4.mp4 | 29.97002997002997 | 1920  | 1080   | 81     | 
| interface_videos/B_5.mp4 | 29.97002997002997 | 1920  | 1080   | 71     |
| interface_videos/C_1.mp4 | 29.97002997002997 | 1920  | 1080   | 99     |
| interface_videos/C_2.mp4 | 29.97002997002997 | 1920  | 1080   | 107    | 
| interface_videos/C_3.mp4 | 29.97002997002997 | 1920  | 1080   | 106    | 
| interface_videos/C_4.mp4 | 29.97002997002997 | 1920  | 1080   | 100    |
| interface_videos/C_5.mp4 | 29.97002997002997 | 1920  | 1080   | 97     |
| interface_videos/D_1.mp4 | 29.97002997002997 | 1920  | 1080   | 86     | 
| interface_videos/D_2.mp4 | 29.97002997002997 | 1920  | 1080   | 95     |
| interface_videos/D_3.mp4 | 29.97002997002997 | 1920  | 1080   | 89     |
| interface_videos/D_4.mp4 | 29.97002997002997 | 1920  | 1080   | 87     |
| interface_videos/D_5.mp4 | 29.97002997002997 | 1920  | 1080   | 86     |

