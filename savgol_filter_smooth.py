#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author : Toshihiko Aoki

import os
import argparse
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='csv/')
    parser.add_argument("--output_dir", type=str, default='smooth')
    parser.add_argument("--keypoint_score", type=float, default=0.3)
    parser.add_argument("--window_length", type=float, default=15)
    parser.add_argument("--polyorder", type=float, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    csv_files = []
    if args.file is not None:
        if not os.path.exists(args.file):
            raise ValueError("file not found")
        if os.path.isdir(args.file):
            from pathlib import Path
            directory_path = Path(args.file)
            csv_paths = []
            for ext in ['.csv']:
                csv_paths.extend(directory_path.glob(f'*{ext}'))
            csv_files = [str(path) for path in csv_paths]
        else:
            csv_files = [args.file]

    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in csv_files:
        basename_with_ext = os.path.basename(csv_path)
        original = pd.read_csv(csv_path)
        columns = original.columns
        joints = set(col.rsplit('_', 1)[0] for col in columns if col != 'frame_number')
        filtered = apply_threshold(original.copy(), joints, args.keypoint_score)
        data_interpolated = linear_interpolation(filtered.copy())
        data_smoothed = smooth_data(data_interpolated.copy(),
                                    window_length=args.window_length, polyorder=args.polyorder)
        data_smoothed.to_csv(os.path.join(args.output_dir, basename_with_ext), index=True)


def apply_threshold(df, joint_names, threshold):
    for joint in joint_names:
        conf_column = f'{joint}_conf'
        x_column = f'{joint}_x'
        y_column = f'{joint}_y'
        mask = df[conf_column] < threshold
        df.loc[mask, [x_column, y_column]] = np.nan
    return df


def linear_interpolation(df):
    for column in df.columns:
        if column.endswith('_x') or column.endswith('_y'):
            mask = np.isnan(df[column])
            df.loc[mask, column] = interp1d(
                df.loc[~mask, 'frame_number'], df.loc[~mask, column], kind='linear', fill_value="extrapolate")(
                df.loc[mask, 'frame_number'])
    return df


def smooth_data(df, window_length=15, polyorder=1):
    for column in df.columns:
        if column.endswith('_x') or column.endswith('_y') :
            df[column] = savgol_filter(df[column], window_length=window_length, polyorder=polyorder)
    return df


if __name__ == '__main__':
    main()
