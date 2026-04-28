#!/usr/bin/env python3
"""
Train a lightweight activity classifier from collected pose CSVs.

Usage:
  python3 train_activity_model.py --data pose_data --out models/pose_activity_model.pkl

Assumptions:
- Each CSV contains rows per frame with columns like lm0_x,lm0_y,...lm32_v (as your sample)
- Directory layout:
    pose_data/
      standing/
        standing_2025....csv
      walking/
      waving/
      falling/
      laying/
      sitting/
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---------------- Feature helpers (adapted from your feature code) ----------------
import math

def landmarks_to_xy_row(row, n_landmarks=33):
    """From a DataFrame row -> Nx2 array of landmark x,y (ignore z/v for now)."""
    xy = []
    for i in range(n_landmarks):
        xi = row.get(f"lm{i}_x", np.nan)
        yi = row.get(f"lm{i}_y", np.nan)
        xy.append([float(xi), float(yi)])
    return np.array(xy, dtype=np.float32)

def normalize_landmarks_xy(xy):
    """Translate & scale normalize: center on mid-hip and scale by torso length."""
    # Mediapipe: left_hip=23, right_hip=24, left_shoulder=11, right_shoulder=12
    try:
        hip_mid = (xy[23] + xy[24]) / 2.0
        shoulder_mid = (xy[11] + xy[12]) / 2.0
    except Exception:
        hip_mid = xy.mean(axis=0)
        shoulder_mid = hip_mid + np.array([0.0, -0.2])
    centered = xy - hip_mid
    torso_len = np.linalg.norm(shoulder_mid - hip_mid)
    scale = torso_len if torso_len > 1e-6 else 1.0
    normalized = centered / scale
    return normalized.flatten()

def angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosv = np.dot(ba, bc) / denom
    cosv = np.clip(cosv, -1.0, 1.0)
    return math.acos(cosv)

def joint_angles(xy):
    """Compute left/right shoulder-elbow-wrist angle approximations."""
    try:
        ls = xy[11]; le = xy[13]; lw = xy[15]
        rs = xy[12]; re = xy[14]; rw = xy[16]
        l_sh_el = angle(ls, le, lw)
        r_sh_el = angle(rs, re, rw)
    except Exception:
        l_sh_el = r_sh_el = 0.0
    return np.array([l_sh_el, r_sh_el], dtype=np.float32)

def extract_features_from_sequence(lm_seq, timestamps, fps=25):
    """
    lm_seq: list of Nx2 arrays, newest last
    timestamps: list of float times corresponding to lm_seq
    Returns feature vector: [normalized_flatten, angles(2), mean_vel_mean, mean_vel_std]
    """
    latest = lm_seq[-1]
    norm_latest = normalize_landmarks_xy(latest)
    angles_latest = joint_angles(latest)
    velocities = []
    for i in range(1, len(lm_seq)):
        dt = max(1e-3, timestamps[i] - timestamps[i-1])
        v = (lm_seq[i] - lm_seq[i-1]) / dt
        vel_mag = np.linalg.norm(v, axis=1)
        velocities.append(vel_mag)
    if velocities:
        vel_stack = np.stack(velocities, axis=0)
        vel_mean = vel_stack.mean(axis=0)
        vel_std = vel_stack.std(axis=0)
        mean_vel_mean = float(np.mean(vel_mean))
        mean_vel_std = float(np.mean(vel_std))
    else:
        mean_vel_mean = 0.0
        mean_vel_std = 0.0
    feat = np.concatenate([norm_latest, angles_latest, np.array([mean_vel_mean, mean_vel_std], dtype=np.float32)], axis=0)
    return feat

# ---------------- Data loader ----------------

def load_csv_to_frames(csv_path, n_landmarks=33):
    """Read csv and return list of frames (Nx2 arrays) and timestamps list"""
    df = pd.read_csv(csv_path)
    frames = []
    timestamps = []
    # Ensure sorted by frame_idx or timestamp
    if 'frame_idx' in df.columns:
        df = df.sort_values('frame_idx')
    elif 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    for _, row in df.iterrows():
        try:
            xy = landmarks_to_xy_row(row, n_landmarks=n_landmarks)
            frames.append(xy)
            if 'timestamp' in row:
                try:
                    ts = pd.to_datetime(row['timestamp']).timestamp()
                except Exception:
                    ts = float(row.get('frame_idx', len(frames))/25.0)
            else:
                ts = float(row.get('frame_idx', len(frames))/25.0)
            timestamps.append(ts)
        except Exception:
            # skip bad rows
            continue
    return frames, timestamps

# ---------------- Create dataset ----------------

def build_dataset_from_folder(data_root, window_size=30, step=5):
    """
    data_root: pose_data folder containing activity subfolders
    window_size: number of frames per window (approx window_seconds * fps)
    step: step between windows
    Returns X (n_samples, feat_dim), y (n_samples,)
    """
    X = []
    y = []
    activities = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print("Found activity classes:", activities)
    for activity in activities:
        folder = os.path.join(data_root, activity)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Processing activity '{activity}' → {len(csv_files)} files")
        for csv in csv_files:
            frames, timestamps = load_csv_to_frames(csv)
            if len(frames) < 2:
                continue
            # sliding windows
            for start in range(0, max(1, len(frames) - window_size + 1), step):
                end = start + window_size
                if end > len(frames):
                    break
                window_frames = frames[start:end]
                window_timestamps = timestamps[start:end]
                feat = extract_features_from_sequence(window_frames, window_timestamps)
                X.append(feat)
                y.append(activity)
    if not X:
        raise RuntimeError("No training samples generated. Check pose_data content and window_size.")
    X = np.vstack(X)
    y = np.array(y)
    return X, y

# ---------------- Train & save ----------------

def train_and_save(X, y, out_path, test_size=0.2, random_state=42):
    print("Dataset:", X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    print("Training RandomForest...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")
    print("Classification report:\n", classification_report(y_test, y_pred))
    # save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": clf, "classes": np.unique(y)}, out_path)
    print("Saved model to:", out_path)
    return clf

# ---------------- CLI ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="pose_data", help="pose_data root folder")
    p.add_argument("--out", type=str, default="models/pose_activity_model.pkl", help="output model path")
    p.add_argument("--window", type=float, default=1.5, help="window size in seconds (approx)")
    p.add_argument("--fps", type=float, default=25.0, help="assumed fps for recorded pose data")
    p.add_argument("--step", type=float, default=0.4, help="sliding step in seconds")
    args = p.parse_args()

    window_frames = max(3, int(round(args.window * args.fps)))
    step_frames = max(1, int(round(args.step * args.fps)))

    print("Building dataset with window_frames=", window_frames, "step_frames=", step_frames)
    X, y = build_dataset_from_folder(args.data, window_size=window_frames, step=step_frames)
    train_and_save(X, y, args.out)

if __name__ == "__main__":
    main()
