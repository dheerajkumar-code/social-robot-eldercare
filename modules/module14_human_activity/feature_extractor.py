#!/usr/bin/env python3
"""
Module 14 — Feature Extractor (v2)
------------------------------------
Shared feature extraction used by BOTH training and inference.
This ensures zero mismatch between training-time and inference-time features.

Feature vector layout (total ~87 features):
  [0:66]   normalized landmark x,y (33 landmarks × 2)
  [66:74]  joint angles (8 angles: knees, hips, elbows)
  [74:79]  body geometry (hip_y, shoulder_y, torso_len, body_height, wrist_above_shoulder)
  [79:81]  asymmetry (shoulder diff, wrist height diff)
  [81:85]  velocity stats (mean_vel, std_vel, hip_vy_mean, hip_vy_std)
  [85:87]  motion energy (upper_body, lower_body)

Key design decisions:
- hip_y absolute position is the #1 fall discriminator
- knee angles separate sitting from standing
- wrist_above_shoulder separates waving
- hip vertical velocity catches the fall motion arc
- All features are scale-normalized to be camera-distance invariant
"""

import math
import numpy as np

# MediaPipe landmark indices (standard 33-point model)
LM_NOSE           = 0
LM_LEFT_SHOULDER  = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_ELBOW     = 13
LM_RIGHT_ELBOW    = 14
LM_LEFT_WRIST     = 15
LM_RIGHT_WRIST    = 16
LM_LEFT_HIP       = 23
LM_RIGHT_HIP      = 24
LM_LEFT_KNEE      = 25
LM_RIGHT_KNEE     = 26
LM_LEFT_ANKLE     = 27
LM_RIGHT_ANKLE    = 28

N_LANDMARKS = 33


# ─────────────────────────────────────────────────────────────
# Low-level geometry helpers
# ─────────────────────────────────────────────────────────────

def _safe_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by rays b→a and b→c (radians)."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_v = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(math.acos(cos_v))


def landmarks_to_xy(landmark_list) -> np.ndarray:
    """
    Convert MediaPipe landmark list OR a dict-keyed row to (33, 2) float32 array.
    Accepts:
      - list of mediapipe landmark objects (with .x, .y attributes)
      - numpy (33, 2) array (pass-through)
    """
    if isinstance(landmark_list, np.ndarray):
        return landmark_list.astype(np.float32)
    return np.array([[lm.x, lm.y] for lm in landmark_list], dtype=np.float32)


def landmarks_to_xy_from_row(row, n_landmarks=N_LANDMARKS) -> np.ndarray:
    """Convert a pandas DataFrame row to (33, 2) array."""
    xy = []
    for i in range(n_landmarks):
        xi = float(row.get(f"lm{i}_x", 0.0))
        yi = float(row.get(f"lm{i}_y", 0.0))
        xy.append([xi, yi])
    return np.array(xy, dtype=np.float32)


def normalize_landmarks(xy: np.ndarray) -> np.ndarray:
    """
    Translate to mid-hip center, scale by torso length.
    Returns flattened (66,) vector.
    This makes features invariant to person distance from camera.
    """
    hip_mid       = (xy[LM_LEFT_HIP] + xy[LM_RIGHT_HIP]) / 2.0
    shoulder_mid  = (xy[LM_LEFT_SHOULDER] + xy[LM_RIGHT_SHOULDER]) / 2.0
    centered      = xy - hip_mid
    torso_len     = np.linalg.norm(shoulder_mid - hip_mid)
    scale         = torso_len if torso_len > 1e-6 else 1.0
    return (centered / scale).flatten().astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Per-frame static features
# ─────────────────────────────────────────────────────────────

def compute_joint_angles(xy: np.ndarray) -> np.ndarray:
    """
    8 joint angles (radians):
      left_knee, right_knee,
      left_hip, right_hip,
      left_elbow, right_elbow,
      left_shoulder_elevation, right_shoulder_elevation
    """
    angles = []
    # Knee angles: hip - knee - ankle
    angles.append(_safe_angle(xy[LM_LEFT_HIP],  xy[LM_LEFT_KNEE],  xy[LM_LEFT_ANKLE]))
    angles.append(_safe_angle(xy[LM_RIGHT_HIP], xy[LM_RIGHT_KNEE], xy[LM_RIGHT_ANKLE]))
    # Hip angles: shoulder - hip - knee
    angles.append(_safe_angle(xy[LM_LEFT_SHOULDER],  xy[LM_LEFT_HIP],  xy[LM_LEFT_KNEE]))
    angles.append(_safe_angle(xy[LM_RIGHT_SHOULDER], xy[LM_RIGHT_HIP], xy[LM_RIGHT_KNEE]))
    # Elbow angles: shoulder - elbow - wrist
    angles.append(_safe_angle(xy[LM_LEFT_SHOULDER],  xy[LM_LEFT_ELBOW],  xy[LM_LEFT_WRIST]))
    angles.append(_safe_angle(xy[LM_RIGHT_SHOULDER], xy[LM_RIGHT_ELBOW], xy[LM_RIGHT_WRIST]))
    # Shoulder elevation: wrist_y vs shoulder_y (negative = wrist above shoulder)
    angles.append(float(xy[LM_LEFT_WRIST][1]  - xy[LM_LEFT_SHOULDER][1]))
    angles.append(float(xy[LM_RIGHT_WRIST][1] - xy[LM_RIGHT_SHOULDER][1]))
    return np.array(angles, dtype=np.float32)


def compute_body_geometry(xy: np.ndarray) -> np.ndarray:
    """
    5 body geometry features:
      hip_y               — absolute vertical position (fall: >1.1)
      shoulder_y          — absolute vertical position
      torso_length        — hip_y - shoulder_y
      body_height         — hip_y - nose_y
      wrist_above_shoulder — fraction of wrists above shoulder line (waving: >0.5)
    """
    hip_y        = float((xy[LM_LEFT_HIP][1]      + xy[LM_RIGHT_HIP][1])      / 2.0)
    shoulder_y   = float((xy[LM_LEFT_SHOULDER][1]  + xy[LM_RIGHT_SHOULDER][1])  / 2.0)
    nose_y       = float(xy[LM_NOSE][1])
    torso_len    = float(hip_y - shoulder_y)
    body_height  = float(hip_y - nose_y)
    # wrist above shoulder = wrist_y < shoulder_y (y increases downward)
    l_above = 1.0 if xy[LM_LEFT_WRIST][1]  < shoulder_y else 0.0
    r_above = 1.0 if xy[LM_RIGHT_WRIST][1] < shoulder_y else 0.0
    wrist_above = (l_above + r_above) / 2.0
    return np.array([hip_y, shoulder_y, torso_len, body_height, wrist_above],
                    dtype=np.float32)


def compute_asymmetry(xy: np.ndarray) -> np.ndarray:
    """
    2 asymmetry features (waving raises one arm, breaking symmetry):
      shoulder_height_diff  — |left_shoulder_y - right_shoulder_y|
      wrist_height_diff     — |left_wrist_y    - right_wrist_y|
    """
    shoulder_diff = abs(float(xy[LM_LEFT_SHOULDER][1] - xy[LM_RIGHT_SHOULDER][1]))
    wrist_diff    = abs(float(xy[LM_LEFT_WRIST][1]    - xy[LM_RIGHT_WRIST][1]))
    return np.array([shoulder_diff, wrist_diff], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Temporal (sequence-level) features
# ─────────────────────────────────────────────────────────────

def compute_velocity_features(lm_seq: list, timestamps: list) -> np.ndarray:
    """
    4 velocity features over the sliding window:
      mean_vel_mag     — mean per-keypoint velocity magnitude (walking > standing)
      std_vel_mag      — std of velocity (erratic motion = waving/falling)
      hip_vy_mean      — mean vertical hip velocity (fall: large positive)
      hip_vy_std       — std of vertical hip velocity
    """
    if len(lm_seq) < 2:
        return np.zeros(4, dtype=np.float32)

    vel_mags    = []
    hip_vy_list = []

    for i in range(1, len(lm_seq)):
        dt = max(1e-3, timestamps[i] - timestamps[i - 1])
        delta = (lm_seq[i] - lm_seq[i - 1]) / dt          # (33, 2)
        vel_mags.append(np.linalg.norm(delta, axis=1))     # (33,)
        # Hip vertical velocity: positive = moving down = potential fall
        hip_vy = ((lm_seq[i][LM_LEFT_HIP][1]  - lm_seq[i-1][LM_LEFT_HIP][1]) +
                  (lm_seq[i][LM_RIGHT_HIP][1] - lm_seq[i-1][LM_RIGHT_HIP][1])) / (2.0 * dt)
        hip_vy_list.append(hip_vy)

    vel_stack    = np.stack(vel_mags, axis=0)               # (T-1, 33)
    mean_vel_mag = float(np.mean(vel_stack))
    std_vel_mag  = float(np.std(vel_stack))
    hip_vy_arr   = np.array(hip_vy_list, dtype=np.float32)
    hip_vy_mean  = float(np.mean(hip_vy_arr))
    hip_vy_std   = float(np.std(hip_vy_arr))

    return np.array([mean_vel_mag, std_vel_mag, hip_vy_mean, hip_vy_std],
                    dtype=np.float32)


def compute_motion_energy(lm_seq: list, timestamps: list) -> np.ndarray:
    """
    2 motion energy features (upper vs lower body):
      upper_body_energy — shoulders, elbows, wrists
      lower_body_energy — hips, knees, ankles
    Distinguishes waving (high upper) from walking (high lower).
    """
    upper_indices = [LM_LEFT_SHOULDER, LM_RIGHT_SHOULDER,
                     LM_LEFT_ELBOW, LM_RIGHT_ELBOW,
                     LM_LEFT_WRIST, LM_RIGHT_WRIST]
    lower_indices = [LM_LEFT_HIP, LM_RIGHT_HIP,
                     LM_LEFT_KNEE, LM_RIGHT_KNEE,
                     LM_LEFT_ANKLE, LM_RIGHT_ANKLE]

    if len(lm_seq) < 2:
        return np.zeros(2, dtype=np.float32)

    upper_e, lower_e = [], []
    for i in range(1, len(lm_seq)):
        dt    = max(1e-3, timestamps[i] - timestamps[i - 1])
        delta = (lm_seq[i] - lm_seq[i - 1]) / dt
        upper_e.append(np.mean(np.linalg.norm(delta[upper_indices], axis=1)))
        lower_e.append(np.mean(np.linalg.norm(delta[lower_indices], axis=1)))

    return np.array([float(np.mean(upper_e)), float(np.mean(lower_e))],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Master feature extraction function
# ─────────────────────────────────────────────────────────────

def extract_features(lm_seq: list, timestamps: list) -> np.ndarray:
    """
    Full feature vector from a sliding window of landmark frames.

    Args:
        lm_seq    : list of (33, 2) numpy arrays, newest last
        timestamps: list of float unix timestamps matching lm_seq

    Returns:
        np.ndarray of shape (87,) — dtype float32
    """
    latest = lm_seq[-1]  # (33, 2)

    norm_lm   = normalize_landmarks(latest)        # (66,)
    angles    = compute_joint_angles(latest)        # (8,)
    geometry  = compute_body_geometry(latest)       # (5,)
    asymmetry = compute_asymmetry(latest)           # (2,)
    velocity  = compute_velocity_features(lm_seq, timestamps)  # (4,)
    energy    = compute_motion_energy(lm_seq, timestamps)      # (2,)

    return np.concatenate([norm_lm, angles, geometry, asymmetry, velocity, energy])


FEATURE_DIM = 87  # expected output size


# ─────────────────────────────────────────────────────────────
# Data augmentation (used only during training)
# ─────────────────────────────────────────────────────────────

def augment_sequence(lm_seq: list, timestamps: list,
                     noise_sigma: float = 0.008,
                     n_augments: int = 3) -> list:
    """
    Generate augmented versions of a landmark sequence.
    Used during training to multiply effective dataset size.

    Augmentations applied:
      1. Gaussian noise on landmark positions
      2. Horizontal mirror (simulate person facing other direction)
      3. Small scale jitter (simulate camera distance variation)

    Returns list of (lm_seq, timestamps) tuples including the original.
    """
    results = [(lm_seq, timestamps)]  # always include original

    for _ in range(n_augments):
        augmented = []
        for frame in lm_seq:
            noisy = frame + np.random.normal(0, noise_sigma, frame.shape).astype(np.float32)
            augmented.append(noisy)
        results.append((augmented, timestamps))

    # Horizontal mirror: flip x coordinate (1 - x)
    mirrored = []
    for frame in lm_seq:
        m = frame.copy()
        m[:, 0] = 1.0 - m[:, 0]
        mirrored.append(m)
    results.append((mirrored, timestamps))

    # Scale jitter: multiply all positions by random scale [0.9, 1.1]
    scale = np.random.uniform(0.9, 1.1)
    scaled = [frame * scale for frame in lm_seq]
    results.append((scaled, timestamps))

    return results
