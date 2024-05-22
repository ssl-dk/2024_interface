#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author : Shuji Awai, Toshihiko Aoki

import os.path
from pathlib import Path
import csv
import math
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

import argparse


def from_csv(file_path):
    pose_history = []
    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pose = {}
            for key in row:
                if '_x' in key or '_y' in key or '_conf' in key:
                    base_key = key.rsplit('_', 1)[0]
                    if base_key not in pose:
                        pose[base_key] = [None, None, None]
                    if '_x' in key:
                        pose[base_key][0] = float(row[key])
                    elif '_y' in key:
                        pose[base_key][1] = float(row[key])
                    elif '_conf' in key:
                        # 使ってない
                        pose[base_key][2] = float(row[key])
                if 'frame_number' == key:
                    # 使ってない
                    pose[key] = int(row[key])
            pose_history.append(pose)

    # [{"nose" :[100, 200, 1.0] ...} ...
    return pose_history


def lr_ankle_distance(pose_history):
    # 足首距離を求める
    return [math.sqrt((pose['right_ankle'][0] - pose['left_ankle'][0]) ** 2 +
                      (pose['right_ankle'][1] - pose['left_ankle'][1]) ** 2) for pose in pose_history]


def walk_cycle(pose_history, fps=30):
    # 足首距離より周期を求める
    distance = lr_ankle_distance(pose_history)
    n = len(distance)
    fft_result = np.fft.fft(distance)
    max_index = max(range(1, n // 2), key=lambda i: abs(fft_result[i]))
    most = max_index / (n / fps)
    return int(fps / most)


def max_diff_in_window(arr, k):
    # kサイズのwindowをスライド、window内の最大-最小を計算しarr中の最大を求める
    if not arr or k <= 0:
        return 0.

    n = len(arr)
    if k > n:
        k = n

    max_diff = 0.
    for i in range(n - k + 1):
        window = arr[i:i + k]
        current_max = max(window)
        current_min = min(window)
        current_diff = current_max - current_min
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff


def center(a, b):
    return [(a[0] + b[0])/2, (a[1] + b[1])/2]


def center_hip(pose):
    return center(pose['left_hip'], pose['right_hip'])


def shoulder_to_hip_length(pose_history):
    shoulder_to_hip_lengths = []
    for pose in pose_history:
        center_shoulder = center(pose['left_shoulder'], pose['right_shoulder'])
        center_hip = center(pose['left_hip'], pose['right_hip'])
        shoulder_to_hip_lengths.append(center_hip[1] - center_shoulder[1])
    return sum(shoulder_to_hip_lengths)/len(shoulder_to_hip_lengths)


def left_elbow_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_elbow'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def left_elbow_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_elbow'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def right_elbow_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_elbow'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def right_elbow_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_elbow'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def left_wrist_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_wrist'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def left_wrist_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_wrist'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def right_wrist_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_wrist'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def right_wrist_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_wrist'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def left_ankle_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_ankle'][0] - center_hip(pose)[0]) / length  for pose in pose_history], cycle)


def left_ankle_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_ankle'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def right_ankle_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_ankle'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def right_ankle_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_ankle'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def left_knee_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_knee'][0] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def left_knee_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_knee'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def right_knee_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_knee'][0] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def right_knee_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['right_knee'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def left_eye_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_eye'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def left_eye_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['left_eye'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


def nose_x_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['nose'][0] - center_hip(pose)[0]) / length for pose in pose_history], cycle)


def nose_y_max(pose_history, cycle, length):
    return max_diff_in_window([(pose['nose'][1] - center_hip(pose)[1]) / length for pose in pose_history], cycle)


class GaitFeature:

    def __init__(self, csv_filepath, fps=30):
        self.csv_filepath = csv_filepath
        self.fps = fps
        self.pose_history = from_csv(csv_filepath)
        self.walk_cycle = walk_cycle(self.pose_history, self.fps)

    @property
    def label(self):
        return os.path.splitext(os.path.basename(self.csv_filepath))[0].split('_')[0]

    @property
    def feature(self):
        length = shoulder_to_hip_length(self.pose_history)
        return [
            # right_wrist_x_max(self.pose_history, self.walk_cycle, length),
            # right_wrist_y_max(self.pose_history, self.walk_cycle, length),
            left_wrist_x_max(self.pose_history, self.walk_cycle, length),
            left_wrist_y_max(self.pose_history, self.walk_cycle, length),
            # right_elbow_x_max(self.pose_history, self.walk_cycle, length),
            # right_elbow_y_max(self.pose_history, self.walk_cycle, length),
            left_elbow_x_max(self.pose_history, self.walk_cycle, length),
            left_elbow_y_max(self.pose_history, self.walk_cycle, length),
            left_ankle_x_max(self.pose_history, self.walk_cycle, length),
            left_ankle_y_max(self.pose_history, self.walk_cycle, length),
            right_ankle_x_max(self.pose_history, self.walk_cycle, length),
            right_ankle_y_max(self.pose_history, self.walk_cycle, length),
            left_knee_x_max(self.pose_history, self.walk_cycle, length),
            left_knee_y_max(self.pose_history, self.walk_cycle, length),
            right_knee_x_max(self.pose_history, self.walk_cycle, length),
            right_knee_y_max(self.pose_history, self.walk_cycle, length),
            # left_eye_x_max(self.pose_history, self.walk_cycle, length),
            left_eye_y_max(self.pose_history, self.walk_cycle, length),
            # nose_x_max(self.pose_history, self.walk_cycle, length),
            # nose_y_max(self.pose_history, self.walk_cycle, length),
        ]


class Database:

    def __init__(
        self,
        csv_path=None,
        norm_model_path='norm_model.pkl',
        save_path='./database.npy',
        exclude_suffix='_1.csv',
        fps=30,
    ):
        self.labels = []
        self.file_paths = []
        self.vectors = []
        self.save_path = save_path
        self.norm_model_path = norm_model_path

        self.norm_model = None
        if os.path.exists(self.norm_model_path) and csv_path is None:
            # 生成済
            with open(self.norm_model_path, 'rb') as file:
                self.norm_model = pickle.load(file)

        if os.path.exists(save_path) and self.norm_model is not None:
            self.load_db()
        elif csv_path is not None and self.norm_model_path is not None and self.save_path is not None:
            self.init_db(csv_path, exclude_suffix=exclude_suffix, fps=fps)
        else:
            raise ValueError('require save_path or csv_path')

    def init_db(self, csv_path, exclude_suffix='_1.csv', fps=30):
        if not os.path.isdir(csv_path):
            raise ValueError('require dir: ' + csv_path)
        gait_features = []
        for filepath in [str(path) for path in Path(csv_path).glob('*.csv')]:
            if filepath.endswith(exclude_suffix):
                continue
            gait_features.append(GaitFeature(filepath, fps=fps))
        if self.norm_model is None:
            norm_model = StandardScaler()
            norm_model.fit([gait_feature.feature for gait_feature in gait_features])
            with open(self.norm_model_path, 'wb') as file:
                pickle.dump(norm_model, file)
            self.norm_model = norm_model

        self.labels = [gait_feature.label for gait_feature in gait_features]
        self.file_paths = [gait_feature.csv_filepath for gait_feature in gait_features]
        self.vectors = self.norm_model.transform([gait_feature.feature for gait_feature in gait_features])
        self.save_db()

    def save_db(self):
        np.save(self.save_path, {'labels': self.labels, 'vectors': self.vectors, 'file_paths': self.file_paths})

    def load_db(self):
        data = np.load(self.save_path, allow_pickle=True).item()
        self.labels = data['labels']
        self.vectors = data['vectors']
        self.file_paths = data['file_paths']  # 判別用

    def cosine_similarity(self, query_vector):
        query_vector = query_vector / np.linalg.norm(query_vector)
        vectors_norm = np.linalg.norm(self.vectors, axis=1)
        dot_products = np.dot(self.vectors, query_vector)
        similarities = dot_products / vectors_norm
        return similarities

    def search_similar(self, gait_feature, top_n=5):
        # 類似検索
        query_vector = self.norm_model.transform([gait_feature.feature])[0]
        similarities = self.cosine_similarity(query_vector)
        top_similar_indices = np.argsort(similarities)[::-1][:top_n]
        results = [(self.labels[i], similarities[i], self.file_paths[i]) for i in top_similar_indices]
        return results

    def add_vector(self, gait_feature):
        # 未使用（正規化を再計算した方がいい）
        if self.norm_model is None:
            raise ValueError('require norm_model')
        self.labels.append(gait_feature.label)
        self.file_paths.append(gait_feature.csv_filepath)
        self.vectors.append(self.norm_model.transform([gait_feature.feature])[0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--norm_model_path", type=str, default='./norm_model.pkl')
    parser.add_argument("--save_path", type=str, default='./database.npy')
    parser.add_argument("--exclude_suffix", type=str, default='_1.csv')
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--query_path", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    fps = args.fps

    db = Database(
        csv_path=args.csv_path,                 # 姿勢推定の結果CSVファイルが存在するディレクトリを入力します（与えた場合再初期化されます）
        norm_model_path=args.norm_model_path,   # modelファイルが出力されます
        save_path=args.save_path,               # databaseファイルが出力されます
        exclude_suffix=args.exclude_suffix,     # CSVファイルより入力を除く接尾を指定します
        fps=fps
    )

    if args.query_path is not None and os.path.isfile(args.query_path):
        query = GaitFeature(csv_filepath=args.query_path, fps=fps)
        similar_results = db.search_similar(query, top_n=10)
        print("Most similar:")
        for label, similarity, file_path in similar_results:
            print(f"Label: {label}, Similarity: {similarity}, FilePath :{file_path}")


if __name__ == "__main__":
    main()
