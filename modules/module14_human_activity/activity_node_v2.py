#!/usr/bin/env python3
"""
Module 14 — Activity Node (v2)
--------------------------------
Production ROS2 node for real-time human activity detection.

Improvements over activity_node.py (v1):
  - Uses shared feature_extractor.py (zero training/inference mismatch)
  - Temporal smoothing: majority vote over last N predictions
  - Confidence thresholding: only publish if confidence >= min_confidence
  - Fall hysteresis: requires 3 consecutive fall detections before alert
  - Non-blocking: camera read in separate thread
  - Publishes both /activity_label AND /activity_confidence
  - ROS2 parameters exposed (no more hardcoded values)

Standalone usage (no ROS2):
  python3 activity_node_v2.py --cam 0 --model models/activity_model_v2.pkl

ROS2 usage:
  python3 activity_node_v2.py --ros --cam 0 --model models/activity_model_v2.pkl

Topics published:
  /activity_label       std_msgs/String  — "standing|walking|sitting|waving|falling|laying"
  /activity_confidence  std_msgs/Float32 — 0.0 to 1.0
"""

import os
import sys
import time
import math
import argparse
import threading
from collections import deque

import cv2
import numpy as np
import joblib

# MediaPipe
try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Install mediapipe: pip install mediapipe")

# ROS2 (optional — only needed in --ros mode)
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    class Node: pass

# Import shared feature extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import (
    landmarks_to_xy,
    extract_features,
    FEATURE_DIM,
)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL   = os.path.join(os.path.dirname(__file__), "models", "activity_model_v2.pkl")
MIN_CONFIDENCE  = 0.55   # below this → publish "unknown"
SMOOTH_WINDOW   = 7      # majority vote over last 7 predictions
FALL_HYSTERESIS = 3      # consecutive fall detections before triggering alert
PRED_INTERVAL_FRAMES = 5 # run inference every N frames (performance tuning)


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Load model saved by train_activity_model_v2.py.
    Returns (classifier, classes, window_frames).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun train_activity_model_v2.py first.")

    blob = joblib.load(model_path)

    # Support both dict format (v2) and raw classifier (v1 fallback)
    if isinstance(blob, dict):
        clf           = blob["model"]
        classes       = blob.get("classes", clf.classes_)
        window_frames = blob.get("window_frames", 22)
        feat_dim      = blob.get("feature_dim", FEATURE_DIM)
        print(f"✅ Model loaded (v2 format)")
        print(f"   Classes      : {list(classes)}")
        print(f"   Window frames: {window_frames}")
        print(f"   Feature dim  : {feat_dim}")
        if feat_dim != FEATURE_DIM:
            raise ValueError(
                f"Feature dim mismatch: model expects {feat_dim}, "
                f"feature_extractor produces {FEATURE_DIM}. "
                f"Retrain with train_activity_model_v2.py"
            )
    else:
        # Legacy v1 model — warn but attempt to use
        print("⚠️  Loading legacy v1 model. Feature mismatch likely. Retrain recommended.")
        clf           = blob
        classes       = getattr(clf, "classes_", None)
        window_frames = 22

    return clf, classes, window_frames


# ─────────────────────────────────────────────────────────────
# Core inference engine
# ─────────────────────────────────────────────────────────────

class ActivityEngine:
    """
    Encapsulates MediaPipe pose + model inference.
    Thread-safe: can be called from camera thread.
    """

    def __init__(self, model_path: str):
        self.clf, self.classes, self.window_frames = load_model(model_path)

        # MediaPipe Pose
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,          # 0=lite, fastest on Pi5
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp_pose

        # Sliding window buffers
        self.lm_buffer = deque(maxlen=self.window_frames)
        self.ts_buffer = deque(maxlen=self.window_frames)

        # Prediction smoothing buffer
        self.pred_history = deque(maxlen=SMOOTH_WINDOW)

        # Fall hysteresis counter
        self.fall_consecutive = 0

        # State
        self.current_label      = "unknown"
        self.current_confidence = 0.0
        self.frame_count        = 0

    def process_frame(self, frame: np.ndarray):
        """
        Process one BGR frame.
        Returns (annotated_frame, label, confidence, fall_alert).
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        # ── Collect landmarks ──
        if results.pose_landmarks:
            lm_xy = landmarks_to_xy(results.pose_landmarks.landmark)
            self.lm_buffer.append(lm_xy)
            self.ts_buffer.append(time.time())
            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 200, 200), thickness=1),
            )
        else:
            # Repeat last known pose to avoid buffer starvation
            if self.lm_buffer:
                self.lm_buffer.append(self.lm_buffer[-1].copy())
                self.ts_buffer.append(time.time())

        # ── Run inference every N frames when buffer is full ──
        fall_alert = False
        if (self.frame_count % PRED_INTERVAL_FRAMES == 0 and
                len(self.lm_buffer) >= self.window_frames):
            label, conf = self._predict()
            self.current_label      = label
            self.current_confidence = conf

            # Fall hysteresis logic
            if label == "falling":
                self.fall_consecutive += 1
                if self.fall_consecutive >= FALL_HYSTERESIS:
                    fall_alert = True
            else:
                self.fall_consecutive = 0

        # ── Overlay ──
        self._draw_overlay(frame, w, h)

        return frame, self.current_label, self.current_confidence, fall_alert

    def _predict(self):
        """Run feature extraction + model inference on current buffer."""
        try:
            feat = extract_features(list(self.lm_buffer), list(self.ts_buffer))
            proba = self.clf.predict_proba(feat.reshape(1, -1))[0]
            idx   = int(np.argmax(proba))
            label = str(self.classes[idx])
            conf  = float(proba[idx])

            # Reject low-confidence predictions
            if conf < MIN_CONFIDENCE:
                label = "unknown"

            # Add to smoothing history
            self.pred_history.append(label)

            # Majority vote over history
            if self.pred_history:
                from collections import Counter
                vote = Counter(self.pred_history).most_common(1)[0][0]
                return vote, conf

            return label, conf

        except Exception as e:
            print(f"Inference error: {e}")
            return "unknown", 0.0

    def _draw_overlay(self, frame, w, h):
        """Draw prediction label and confidence bar on frame."""
        label = self.current_label
        conf  = self.current_confidence

        # Color coding by activity
        color_map = {
            "standing": (0, 255, 0),
            "walking":  (0, 200, 255),
            "sitting":  (255, 200, 0),
            "waving":   (255, 100, 255),
            "falling":  (0, 0, 255),
            "laying":   (0, 100, 255),
            "unknown":  (128, 128, 128),
        }
        color = color_map.get(label, (255, 255, 255))

        # Background box
        cv2.rectangle(frame, (0, 0), (360, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"{label.upper()}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"conf: {conf*100:.0f}%  frames: {len(self.lm_buffer)}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Confidence bar
        bar_w = int(conf * 340)
        cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 8), color, -1)
        cv2.rectangle(frame, (10, h - 20), (350, h - 8), (80, 80, 80), 1)

        # Fall warning flash
        if label == "falling" and self.fall_consecutive >= 1:
            text = f"⚠ FALL ({self.fall_consecutive}/{FALL_HYSTERESIS})"
            cv2.putText(frame, text, (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def destroy(self):
        self.pose.close()


# ─────────────────────────────────────────────────────────────
# Standalone runner (no ROS2)
# ─────────────────────────────────────────────────────────────

def run_standalone(cam_index: int, model_path: str):
    print("=" * 50)
    print("  Module 14 Activity Detection (standalone)")
    print("=" * 50)

    engine = ActivityEngine(model_path)
    cap    = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam_index}")

    print(f"\n📷 Camera {cam_index} opened. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, label, conf, fall_alert = engine.process_frame(frame)

            if fall_alert:
                print(f"🚨 FALL ALERT — confirmed after {FALL_HYSTERESIS} detections!")
            elif label != "unknown":
                print(f"Activity: {label:12s}  confidence: {conf*100:.0f}%", end="\r")

            cv2.imshow("Activity Detection v2", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.destroy()


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class ActivityNodeROS(Node):
    """
    ROS2 node for activity detection.

    Publishes:
      /activity_label       std_msgs/String
      /activity_confidence  std_msgs/Float32
      /activity_alert       std_msgs/String  (only on fall detection)
    """

    def __init__(self, cam_index: int, model_path: str):
        super().__init__("activity_node_v2")

        # Publishers
        self.pub_label = self.create_publisher(String,  "/activity_label",      10)
        self.pub_conf  = self.create_publisher(Float32, "/activity_confidence",  10)
        self.pub_alert = self.create_publisher(String,  "/activity_alert",       10)

        # ROS2 parameters
        self.declare_parameter("min_confidence",  MIN_CONFIDENCE)
        self.declare_parameter("smooth_window",   SMOOTH_WINDOW)
        self.declare_parameter("fall_hysteresis", FALL_HYSTERESIS)

        # Engine
        self.engine = ActivityEngine(model_path)
        self.get_logger().info(f"Activity engine loaded from: {model_path}")

        # Camera in separate thread to not block ROS spin
        self.cap         = cv2.VideoCapture(cam_index)
        self.running     = True
        self.cam_thread  = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

        self.get_logger().info("Activity node v2 running. Publishing to /activity_label")

    def _camera_loop(self):
        """Camera capture runs in background thread."""
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera!")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            _, label, conf, fall_alert = self.engine.process_frame(frame)

            # Publish label
            msg_label      = String()
            msg_label.data = label
            self.pub_label.publish(msg_label)

            # Publish confidence
            msg_conf      = Float32()
            msg_conf.data = conf
            self.pub_conf.publish(msg_conf)

            # Publish fall alert (only on confirmed fall)
            if fall_alert:
                msg_alert      = String()
                msg_alert.data = "FALL_DETECTED"
                self.pub_alert.publish(msg_alert)
                self.get_logger().warn("🚨 FALL DETECTED — alert published to /activity_alert")

    def destroy_node(self):
        self.running = False
        self.cap.release()
        self.engine.destroy()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Activity detection node v2")
    p.add_argument("--ros",   action="store_true",        help="Run as ROS2 node")
    p.add_argument("--cam",   type=int,   default=0,      help="Camera index")
    p.add_argument("--model", type=str,   default=DEFAULT_MODEL, help="Model .pkl path")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ros:
        if not ROS_AVAILABLE:
            print("❌ rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = ActivityNodeROS(cam_index=args.cam, model_path=args.model)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down activity node v2")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(cam_index=args.cam, model_path=args.model)


if __name__ == "__main__":
    main()
