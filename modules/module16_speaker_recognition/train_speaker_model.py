#!/usr/bin/env python3
"""
Module 16 — train_speaker_model.py
=====================================
Trains the speaker recognition SVM model from voice samples.

Usage:
  python3 train_speaker_model.py
  python3 train_speaker_model.py --data known_voices --out model/speaker_model.pkl
  python3 train_speaker_model.py --evaluate    # cross-validation report

Data layout:
  known_voices/
    Dheeraj/
      sample_1.wav
      sample_2.wav
      ...
    Kuber/
      sample_1.wav
      ...

Output: model/speaker_model.pkl  (joblib dict with model + metadata)
"""

import os
import sys
import glob
import argparse
import logging
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_from_file, FEATURE_DIM

# ── Configuration ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "known_voices")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "speaker_model.pkl")
MIN_SAMPLES_PER_SPEAKER = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TrainSpeaker")


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_dataset(data_dir: str = DATA_DIR):
    """
    Load all WAV samples from known_voices/ structure.
    Returns (X, y, speaker_names).
    """
    X, y = [], []
    speaker_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not speaker_names:
        raise RuntimeError(f"No speaker folders found in {data_dir}")

    logger.info(f"Found {len(speaker_names)} speakers: {speaker_names}")

    for speaker in speaker_names:
        speaker_dir = os.path.join(data_dir, speaker)
        wav_files   = glob.glob(os.path.join(speaker_dir, "*.wav"))

        if len(wav_files) < MIN_SAMPLES_PER_SPEAKER:
            logger.warning(
                f"  {speaker}: only {len(wav_files)} samples "
                f"(need {MIN_SAMPLES_PER_SPEAKER}) — skipping"
            )
            continue

        loaded = 0
        for wav in wav_files:
            feat = extract_from_file(wav)
            if np.all(feat == 0):
                logger.warning(f"  Skipping empty features: {os.path.basename(wav)}")
                continue
            X.append(feat)
            y.append(speaker)
            loaded += 1

        logger.info(f"  {speaker:20s}: {loaded} samples loaded")

    if len(X) == 0:
        raise RuntimeError("No valid samples found. Check known_voices/ directory.")

    return np.array(X, dtype=np.float32), np.array(y)


# ─────────────────────────────────────────────────────────────
# Model building
# ─────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Build sklearn Pipeline:
      StandardScaler → SVC(RBF, probability=True)

    Why SVC with RBF kernel?
      - Excellent for high-dimensional MFCC features
      - probability=True enables confidence scores
      - Works well with small datasets (20-50 samples)
      - No GPU needed
      - Fast inference (<1ms per prediction)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(
            kernel      = "rbf",
            C           = 10.0,       # regularisation (tune if overfitting)
            gamma       = "scale",    # auto-scale by feature variance
            probability = True,       # needed for confidence scores
            class_weight= "balanced", # handles imbalanced speaker counts
            random_state= 42,
        ))
    ])


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray, evaluate: bool = True):
    """
    Train the SVM pipeline and optionally run cross-validation.
    Returns fitted Pipeline.
    """
    logger.info(f"Training on {X.shape[0]} samples, {X.shape[1]} features, "
                f"{len(set(y))} speakers")

    pipeline = build_pipeline()

    # Cross-validation (only if enough samples per speaker)
    if evaluate and X.shape[0] >= 6:
        n_splits = min(3, min(np.bincount([list(set(y)).index(yi) for yi in y])))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            logger.info(f"Cross-val accuracy: {cv_scores.mean()*100:.1f}% "
                        f"± {cv_scores.std()*100:.1f}%")
        else:
            logger.info("Not enough samples per speaker for cross-val")

    # Final fit on all data
    pipeline.fit(X, y)
    train_acc = accuracy_score(y, pipeline.predict(X))
    logger.info(f"Training accuracy: {train_acc*100:.1f}%")

    if evaluate:
        print("\nClassification Report (training set):")
        print(classification_report(y, pipeline.predict(X), zero_division=0))

    return pipeline


# ─────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, speakers: list,
               model_path: str = MODEL_PATH):
    """Save model + metadata to joblib file."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        "model":       pipeline,
        "speakers":    speakers,
        "feature_dim": FEATURE_DIM,
        "n_speakers":  len(speakers),
    }
    joblib.dump(payload, model_path)
    logger.info(f"Model saved → {model_path}")
    logger.info(f"  Speakers : {speakers}")
    logger.info(f"  Features : {FEATURE_DIM}")


def load_model(model_path: str = MODEL_PATH) -> dict:
    """Load model from joblib file. Returns payload dict."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python3 train_speaker_model.py"
        )
    payload = joblib.load(model_path)
    logger.info(f"Model loaded: {payload['speakers']}")
    return payload


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train speaker recognition model")
    p.add_argument("--data",     type=str, default=DATA_DIR,   help="known_voices dir")
    p.add_argument("--out",      type=str, default=MODEL_PATH, help="output model path")
    p.add_argument("--evaluate", action="store_true",          help="Run cross-validation")
    p.add_argument("--no-eval",  action="store_true",          help="Skip evaluation")
    args = p.parse_args()

    print("=" * 55)
    print("  Module 16 — Speaker Recognition Training")
    print("=" * 55)

    # Load data
    X, y = load_dataset(args.data)
    speakers = sorted(set(y))

    # Train
    pipeline = train(X, y, evaluate=not args.no_eval)

    # Save
    save_model(pipeline, speakers, args.out)

    print(f"\n✅ Training complete")
    print(f"   Speakers  : {speakers}")
    print(f"   Samples   : {len(X)}")
    print(f"   Model     : {args.out}")


if __name__ == "__main__":
    main()
