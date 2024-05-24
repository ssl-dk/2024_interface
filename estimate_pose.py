#!/usr/bin/env python
# -*- coding: utf-8 -*-

# original code: https://github.com/Kazuhito00/MoveNet-Python-Example
# original author: 高橋かずひと(https://twitter.com/KzhtTkhs)
# modify : Toshihiko Aoki

import sys
import os.path
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
    parser.add_argument("--debug_output",  action='store_true')
    parser.add_argument('--csv', action='store_true')

    parser.add_argument('--detection', action='store_true')
    parser.add_argument('--score_th', type=float, default=0.4)
    parser.add_argument('--nms_th', type=float, default=0.85)

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
        keypoint_x = image_width * keypoints_with_scores[index][1]
        keypoint_y = image_height * keypoints_with_scores[index][0]
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

    is_debug_output = False
    is_csv = False

    if args.debug_output:
        os.makedirs('debugs', exist_ok=True)
        is_debug_output = True
    if args.csv:
        os.makedirs('csv', exist_ok=True)
        is_csv = True

    cap_devices = [cap_device]
    if args.file is not None:
        if not os.path.exists(args.file):
            raise ValueError("file not found")
        if os.path.isdir(args.file):
            from pathlib import Path
            directory_path = Path(args.file)
            video_paths = []
            for ext in ['.mp4', '.avi', '.mkv', '.mov']:
                video_paths.extend(directory_path.glob(f'*{ext}'))
            cap_devices = [str(path) for path in video_paths]
        else:
            cap_devices = [args.file]

    mirror = args.mirror
    model_select = args.model_select
    keypoint_score_th = args.keypoint_score

    for cap_device in cap_devices:
        if is_debug_output or is_csv:
            basename = os.path.splitext(os.path.basename(cap_device))[0]

        # カメラ準備 ###############################################################
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        if is_debug_output:
            # 重畳動画出力準備 ######################################################
            debug_file_path = 'debugs/' + basename + '_debug.mp4'
            debug_writer = cv.VideoWriter(debug_file_path,
                                          cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        if is_csv:
            # CSVヘッダ出力 #######################################################
            csv_file_path = 'csv/' + basename + '.csv'
            csv_writer = open(csv_file_path, 'w', newline='\n', encoding='utf-8')
            csv_writer.write(','.join(KEYPOINTS_LABELS) + '\n')

        # モデルロード #############################################################
        input_size = -1
        if model_select == 0:
            model_path = "onnx/movenet_singlepose_lightning_4.onnx"
            yolo_path = "onnx/damoyolo_tinynasL20_T_418.onnx"
            input_size = 192
        elif model_select == 1:
            model_path = "onnx/movenet_singlepose_thunder_4.onnx"
            yolo_path = "onnx/damoyolo_tinynasL20_T_418.onnx"
            input_size = 256
        elif model_select == 2:
            model_path = "onnx/litehrnet_18_coco_Nx256x192.onnx"
            yolo_path = "onnx/damoyolo_tinynasL20_T_418.onnx"
        elif model_select == 3:
            model_path = "onnx/hrnet_coco_w48_384x288.onnx"
            yolo_path = "onnx/damoyolo_tinynasL20_T_418.onnx"
        else:
            sys.exit(
                "*** model_select {} is invalid value. Please use 0-1. ***".format(
                    model_select))

        if args.detection:
            from damoyolo_onnx import DAMOYOLO
            damo_yolo = DAMOYOLO(model_path=yolo_path)

        if model_select < 2:
            onnx_session = onnxruntime.InferenceSession(
                model_path,
                providers=[
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                ],
            )
        else:
            from HRNET import HRNET
            hrnet = HRNET(model_path)

        try:
            frame_number = 0
            while True:
                start_time = time.time()

                # カメラキャプチャ #####################################################
                ret, frame = cap.read()
                if not ret:
                    break
                if mirror:
                    frame = cv.flip(frame, 1)  # ミラー表示

                if args.detection:
                    bboxes, scores, class_ids = damo_yolo(frame, score_th=args.score_th, nms_th=args.nms_th)
                    # 動画には一人のみ写っているものとし、最大スコアを持つ矩形を処理対象とする
                    max_index = np.argmax(scores)

                # 検出実施 ############################################################
                if model_select < 2:
                    if args.detection:
                        # HRNETと同等の処理を記載（関心矩形切出）
                        x1, y1, x2, y2 = bboxes[max_index]
                        box_width, box_height = x2 - x1, y2 - y1

                        x1 = max(int(x1 - box_width * 0.1), 0)
                        x2 = min(int(x2 + box_width * 0.1), frame_width)
                        y1 = max(int(y1 - box_height * 0.1), 0)
                        y2 = min(int(y2 + box_height * 0.1), frame_height)

                        crop = frame[y1:y2, x1:x2]
                    else:
                        crop = frame
                    keypoints, scores = run_inference(
                        onnx_session,
                        input_size,
                        crop,
                    )
                    if args.detection:
                        # Fix the body pose to the original image
                        keypoints = [[x + x1, y + y1] for [x, y] in keypoints]
                else:
                    # HRNETはモデルに矩形入力に応じる引数がある
                    deteciton = None
                    if args.detection:
                        deteciton = [[bboxes[max_index]], [scores[max_index]], [class_ids[max_index]]]
                    _, keypoints, scores = hrnet(frame, deteciton)
                    if args.detection:
                        keypoints = keypoints[0]

                elapsed_time = time.time() - start_time

                if is_debug_output or not (is_debug_output or is_csv):
                    debug_image = copy.deepcopy(frame)
                    # デバッグ描画
                    debug_image = draw_debug(
                        debug_image,
                        elapsed_time,
                        keypoint_score_th,
                        keypoints,
                        scores,
                    )

                if is_debug_output:
                    # 重畳動画の出力 #####################################################
                    debug_writer.write(debug_image)
                if is_csv:
                    # CSV出力 ##########################################################
                    output_csv(csv_writer, keypoints, scores, frame_number=frame_number)

                frame_number += 1

                # キー処理(ESC：終了) ####################################################
                key = cv.waitKey(1)
                if key == 27:  # ESC
                    break

                if not (is_debug_output or is_csv):
                    # 画面反映 ##########################################################
                    cv.imshow('Pose estimation Demo', debug_image)

        finally:
            if is_debug_output:
                debug_writer.release()
            if is_csv:
                csv_writer.close()
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
    int_keypoints = [[int(x) for x in origin_xy] for origin_xy in keypoints]
    debug_image = copy.deepcopy(image)

    for keypoint_index, (keypoint, score) in enumerate(zip(int_keypoints, scores)):
        if score > keypoint_score_th:
            cv.circle(
                debug_image,
                keypoint, 2,
                tuple(JOINT_COLOR_WHITE), 2)

    for (i, j, color) in LINE_PALLET:
        if scores[i] > keypoint_score_th and scores[j] > keypoint_score_th:
            kp_from = int_keypoints[i]
            kp_to = int_keypoints[j]
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


KEYPOINTS_LABELS = [
    "frame_number",
    "nose_x", "nose_y", "nose_conf",
    "left_eye_x", "left_eye_y", "left_eye_conf",
    "right_eye_x", "right_eye_y", "right_eye_conf",
    "left_ear_x", "left_ear_y", "left_ear_conf",
    "right_ear_x", "right_ear_y", "right_ear_conf",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_conf",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_conf",
    "left_elbow_x", "left_elbow_y", "left_elbow_conf",
    "right_elbow_x", "right_elbow_y", "right_elbow_conf",
    "left_wrist_x", "left_wrist_y", "left_wrist_conf",
    "right_wrist_x", "right_wrist_y", "right_wrist_conf",
    "left_hip_x", "left_hip_y", "left_hip_conf",
    "right_hip_x", "right_hip_y", "right_hip_conf",
    "left_knee_x", "left_knee_y", "left_knee_conf",
    "right_knee_x", "right_knee_y", "right_knee_conf",
    "left_ankle_x", "left_ankle_y", "left_ankle_conf",
    "right_ankle_x", "right_ankle_y", "right_ankle_conf"
]


def output_csv(
    file_descriptor,
    keypoints,
    scores,
    frame_number=-1,
):
    csv_line = str(frame_number)
    for keypoint_index, (keypoint, score) in enumerate(zip(keypoints, scores)):
        csv_line = csv_line + ',' + str(keypoint[0]) + ',' + str(keypoint[1]) + ',' + str(score)
    file_descriptor.write(csv_line + '\n')


if __name__ == '__main__':
    main()