#!/usr/bin/env python3
"""
Module 14 — Train Activity Model (v2)
---------------------------------------
Production training pipeline for human activity recognition.

What this fixes vs the old train_activity_model.py:
  - OLD: 1 sample per CSV file (mean of all rows) → 6 total samples → useless
  - NEW: sliding window → ~500-800 samples from same data
  - NEW: rich feature vector (87 features vs raw mean)
  - NEW: data augmentation (3x-5x more samples)
  - NEW: class_weight='balanced' for fall/waving minority classes
  - NEW: saves both model AND class list in joblib dict
  - NEW: compatible with activity_node_v2.py inference

Usage:
  cd modules/module14_human_activity
  python3 train_activity_model_v2.py

  # With custom paths:
  python3 train_activity_model_v2.py --data pose_data --out models/activity_model_v2.pkl
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import joblib
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Import shared feature extractor (same file used by inference node)
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import (
    landmarks_to_xy_from_row,
    extract_features,
    augment_sequence,
    FEATURE_DIM,
)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "pose_data")
DEFAULT_OUT  = os.path.join(BASE_DIR, "models", "activity_model_v2.pkl")

WINDOW_SECONDS = 1.5   # sliding window duration
ASSUMED_FPS    = 15.0  # your collect_pose_data.py uses FPS=15
STEP_SECONDS   = 0.4   # step between windows
WINDOW_FRAMES  = max(3, int(round(WINDOW_SECONDS * ASSUMED_FPS)))  # 22 frames
STEP_FRAMES    = max(1, int(round(STEP_SECONDS   * ASSUMED_FPS)))  # 6 frames

# Minimum frames needed to form one window
MIN_FRAMES = WINDOW_FRAMES

# Fall gets higher weight because it's safety-critical
CLASS_WEIGHTS = {
    "falling":  3.0,
    "laying":   2.0,
    "waving":   1.5,
    "sitting":  1.0,
    "standing": 1.0,
    "walking":  1.0,
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_csv_to_frames(csv_path: str):
    """
    Read a pose CSV and return (frames, timestamps).
    frames     : list of (33, 2) float32 arrays
    timestamps : list of float (unix seconds)
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ⚠️  Cannot read {csv_path}: {e}")
        return [], []

    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    frames, timestamps = [], []
    for _, row in df.iterrows():
        try:
            xy = landmarks_to_xy_from_row(row)
            frames.append(xy)
            if "timestamp" in row:
                try:
                    ts = pd.to_datetime(row["timestamp"]).timestamp()
                except Exception:
                    ts = float(len(frames)) / ASSUMED_FPS
            else:
                ts = float(len(frames)) / ASSUMED_FPS
            timestamps.append(ts)
        except Exception:
            continue

    return frames, timestamps


def build_dataset(data_root: str, augment: bool = True):
    """
    Build (X, y, sample_weights) from all pose CSVs under data_root.

    data_root layout:
        data_root/
          standing/  *.csv
          walking/   *.csv
          sitting/   *.csv
          waving/    *.csv
          falling/   *.csv
          laying/    *.csv
    """
    X, y, weights = [], [], []

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    activities = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    if not activities:
        raise RuntimeError(f"No activity subfolders found in {data_root}")

    print(f"\n📂 Loading data from: {data_root}")
    print(f"   Activities found: {activities}")
    print(f"   Window: {WINDOW_FRAMES} frames ({WINDOW_SECONDS}s) | Step: {STEP_FRAMES} frames\n")

    for activity in activities:
        folder    = os.path.join(data_root, activity)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        w         = CLASS_WEIGHTS.get(activity, 1.0)

        if not csv_files:
            print(f"  ⚠️  No CSV files for '{activity}' — skipping")
            continue

        samples_before = len(X)

        for csv_path in csv_files:
            frames, timestamps = load_csv_to_frames(csv_path)
            if len(frames) < MIN_FRAMES:
                print(f"  ⚠️  {os.path.basename(csv_path)}: only {len(frames)} frames, need {MIN_FRAMES} — skipping")
                continue

            # Generate windows
            windows = []
            for start in range(0, len(frames) - WINDOW_FRAMES + 1, STEP_FRAMES):
                end = start + WINDOW_FRAMES
                windows.append((frames[start:end], timestamps[start:end]))

            # Apply augmentation to each window
            for base_seq, base_ts in windows:
                if augment:
                    versions = augment_sequence(base_seq, base_ts, n_augments=2)
                else:
                    versions = [(base_seq, base_ts)]

                for seq, ts in versions:
                    try:
                        feat = extract_features(seq, ts)
                        if feat.shape[0] != FEATURE_DIM:
                            continue
                        X.append(feat)
                        y.append(activity)
                        weights.append(w)
                    except Exception as e:
                        continue

        samples_added = len(X) - samples_before
        print(f"  ✅ {activity:12s}: {len(csv_files)} files → {samples_added} samples")

    if not X:
        raise RuntimeError("No training samples generated. Check pose_data content.")

    return np.vstack(X), np.array(y), np.array(weights)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray):
    """
    Train RandomForest with cross-validation.
    Returns trained classifier and cross-val scores.
    """
    print(f"\n🧠 Training RandomForest...")
    print(f"   Dataset shape : {X.shape}")
    print(f"   Class distribution: {dict(Counter(y))}\n")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # handles imbalanced classes automatically
        n_jobs=-1,
        random_state=42,
    )

    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"   Cross-val accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Final fit on all data with sample weights
    clf.fit(X, y, sample_weight=sample_weights)

    return clf, cv_scores


def evaluate_model(clf, X: np.ndarray, y: np.ndarray):
    """Print per-class metrics and confusion matrix."""
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n📊 Training-set accuracy: {acc*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    classes = clf.classes_
    cm = confusion_matrix(y, y_pred, labels=classes)
    print("Confusion Matrix:")
    header = f"{'':12s}" + "".join(f"{c[:6]:>8s}" for c in classes)
    print(header)
    for i, row_class in enumerate(classes):
        row_str = f"{row_class:12s}" + "".join(f"{cm[i,j]:8d}" for j in range(len(classes)))
        print(row_str)

    # Per-class fall recall — most important metric
    fall_idx = list(classes).index("falling") if "falling" in classes else None
    if fall_idx is not None:
        fall_recall = cm[fall_idx, fall_idx] / (cm[fall_idx].sum() + 1e-8)
        print(f"\n🚨 Fall Detection Recall: {fall_recall*100:.1f}%")
        if fall_recall < 0.80:
            print("   ⚠️  Below 80% target — collect more fall data or adjust window size")
        else:
            print("   ✅ Above 80% threshold")

    return acc


def save_model(clf, classes, out_path: str, metadata: dict):
    """Save model + metadata in joblib dict format (required by activity_node_v2.py)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "model":          clf,
        "classes":        classes,
        "feature_dim":    FEATURE_DIM,
        "window_frames":  WINDOW_FRAMES,
        "step_frames":    STEP_FRAMES,
        "assumed_fps":    ASSUMED_FPS,
        "metadata":       metadata,
    }
    joblib.dump(payload, out_path)
    print(f"\n💾 Model saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# Feature importance report
# ─────────────────────────────────────────────────────────────

def print_top_features(clf, n=15):
    """Print top N most important features for interpretability."""
    feature_names = (
        [f"norm_lm{i}_{c}" for i in range(33) for c in ("x", "y")] +
        ["angle_left_knee", "angle_right_knee",
         "angle_left_hip", "angle_right_hip",
         "angle_left_elbow", "angle_right_elbow",
         "shoulder_elev_L", "shoulder_elev_R"] +
        ["hip_y", "shoulder_y", "torso_len", "body_height", "wrist_above_shoulder"] +
        ["shoulder_asym", "wrist_asym"] +
        ["vel_mean", "vel_std", "hip_vy_mean", "hip_vy_std"] +
        ["upper_body_energy", "lower_body_energy"]
    )

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n]

    print(f"\n🔍 Top {n} Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"   {rank:2d}. {name:35s} {importances[idx]*100:.2f}%")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train activity model v2")
    p.add_argument("--data",    type=str, default=DEFAULT_DATA, help="pose_data root folder")
    p.add_argument("--out",     type=str, default=DEFAULT_OUT,  help="output .pkl path")
    p.add_argument("--no-aug",  action="store_true",            help="disable data augmentation")
    args = p.parse_args()

    print("=" * 60)
    print("  Module 14 — Activity Model Training (v2)")
    print("=" * 60)

    # Step 1: Load and build dataset
    X, y, sample_weights = build_dataset(args.data, augment=not args.no_aug)

    # Step 2: Train
    clf, cv_scores = train_model(X, y, sample_weights)

    # Step 3: Evaluate
    acc = evaluate_model(clf, X, y)

    # Step 4: Feature importance
    print_top_features(clf, n=10)

    # Step 5: Save
    metadata = {
        "cv_mean":  float(cv_scores.mean()),
        "cv_std":   float(cv_scores.std()),
        "n_samples": len(X),
        "augmented": not args.no_aug,
    }
    save_model(clf, clf.classes_, args.out, metadata)

    print("\n" + "=" * 60)
    print(f"  Training complete. CV accuracy: {cv_scores.mean()*100:.1f}%")
    if cv_scores.mean() < 0.70:
        print("  ⚠️  Accuracy below 70% — collect more data per class (target: 10 CSVs each)")
    elif cv_scores.mean() < 0.85:
        print("  ⚠️  Accuracy 70-85% — collect 5+ more CSVs for low-recall classes")
    else:
        print("  ✅ Accuracy ≥85% — model ready for deployment")
    print("=" * 60)


if __name__ == "__main__":
    main()
