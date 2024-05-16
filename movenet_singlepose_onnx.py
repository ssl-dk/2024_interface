#!/usr/bin/env python
# -*- coding: utf-8 -*-

# original source: https://github.com/Kazuhito00/MoveNet-Python-Example
# original author: 高橋かずひと(https://twitter.com/KzhtTkhs)
# modify : Toshihiko Aoki

import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument("--model_select", type=int, default=0)
    parser.add_argument("--keypoint_score", type=float, default=0.4)

    # add
    parser.add_argument('--debug_output', action='store_true')
    parser.add_argument('--csv', action='store_true')

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = input_image.astype('int32')  # int32へキャスト

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # キーポイント、スコア取り出し
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.file is not None:
        cap_device = args.file

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    if model_select == 0:
        model_path = "onnx/movenet_singlepose_lightning_4.onnx"
        input_size = 192
    elif model_select == 1:
        model_path = "onnx/movenet_singlepose_thunder_4.onnx"
        input_size = 256
    else:
        sys.exit(
            "*** model_select {} is invalid value. Please use 0-1. ***".format(
                model_select))

    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    while True:
        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv.flip(frame, 1)  # ミラー表示
        debug_image = copy.deepcopy(frame)

        # 検出実施 ##############################################################
        keypoints, scores = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            keypoint_score_th,
            keypoints,
            scores,
        )

        # キー処理(ESC：終了) ##################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('MoveNet(singlepose) Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


# デバッグ動画色
LEFT_LINE_COLOR_RED = (0, 0, 255)
RIGHT_LINE_COLOR_BLUE = (255, 0, 0)
CENTER_LINE_COLOR_WHITE = (255, 255, 255)
JOINT_COLOR_WHITE = (255, 255, 255)


LINE_PALLET = [
    # 右の繋がり
    [16, 14, RIGHT_LINE_COLOR_BLUE],    # 右足首-右膝
    [14, 12, RIGHT_LINE_COLOR_BLUE],    # 右膝-右尻
    [12, 6, RIGHT_LINE_COLOR_BLUE],     # 右尻-右肩
    [6, 8, RIGHT_LINE_COLOR_BLUE],      # 右肩-右肘
    [8, 10, RIGHT_LINE_COLOR_BLUE],     # 右肘-右手首
    # 左の繋がり
    [15, 13, LEFT_LINE_COLOR_RED],      # 左足首-左膝
    [13, 11, LEFT_LINE_COLOR_RED],      # 左膝-左尻
    [11, 5, LEFT_LINE_COLOR_RED],       # 左尻-左肩
    [5, 7, LEFT_LINE_COLOR_RED],        # 左肩-左肘
    [7, 9, LEFT_LINE_COLOR_RED],        # 左肘-左手首
    # 左右結線
    [11, 12, CENTER_LINE_COLOR_WHITE],  # 左尻-右尻
    [5, 6, CENTER_LINE_COLOR_WHITE],    # 左肩-右肩
]


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    for keypoint_index, (keypoint, score) in enumerate(zip(keypoints, scores)):
        if score > keypoint_score_th:
            cv.circle(
                debug_image,
                keypoint, 2,
                tuple(JOINT_COLOR_WHITE), 2)

    for (i, j, color) in LINE_PALLET:
        if scores[i] > keypoint_score_th and scores[j] > keypoint_score_th:
            kp_from = keypoints[i]
            kp_to = keypoints[j]
            cv.line(debug_image, kp_from, kp_to, color, 2)

    # 処理時間
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()