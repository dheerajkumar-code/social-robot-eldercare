#!/usr/bin/env python3
"""
Module 12 — Emotion Detection Node (v2)
-----------------------------------------
Real-time emotion detection for elderly healthcare robot.

What this fixes vs emotion_subtitle_node.py (v1):
  - v1: model.predict() blocks camera loop for ~200ms every frame
  - v2: Inference in background thread — camera always runs at full FPS

  - v1: No ROS2 integration (standalone only)
  - v2: ROS2 node publishes /emotion_label, /emotion_confidence, /emotion_active

  - v1: Probability averaging — drifts toward neutral under noise
  - v2: Majority vote over last N predictions — more stable, less jitter

  - v1: Crashes immediately if .h5 model file is missing
  - v2: Clear error message + instructions, then graceful exit

  - v1: Runs inference every single frame (~5 FPS max on Pi5)
  - v2: Configurable inference interval (default: every 3rd frame → 15+ FPS display)

  - v1: All 7 emotions published with equal weight
  - v2: Confidence threshold — only publishes when confident (default 0.40)
       Elderly-context mapping: disgusted/surprised → neutral if below threshold

  - v1: face-recognition incorrectly listed in requirements (that is M11)
  - v2: Corrected — M12 only needs opencv, tensorflow, numpy

Topics published:
  /emotion_label      std_msgs/String  — "neutral|happy|sad|fearful|angry|..."
  /emotion_confidence std_msgs/Float32 — 0.0 to 1.0
  /emotion_active     std_msgs/Bool    — True when a face is detected in frame

Standalone:
  python3 emotion_node_v2.py --model models/fer_model.h5 --src 0
  python3 emotion_node_v2.py --model models/fer_model.h5 --src 0 --debug

ROS2:
  python3 emotion_node_v2.py --ros --model models/fer_model.h5 --src 0
"""

import os
import sys
import time
import queue
import logging
import threading
import argparse
from collections import deque, Counter

import cv2
import numpy as np

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32, Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# TensorFlow (required)
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")       # suppress TF startup spam
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Emotions that matter for elderly care decisions
# disgusted / surprised are less actionable → mapped to neutral if low confidence
ACTIONABLE_EMOTIONS   = {"angry", "fearful", "happy", "neutral", "sad"}
MARGINAL_EMOTIONS     = {"disgusted", "surprised"}
MARGINAL_TO_NEUTRAL_THRESHOLD = 0.45   # if marginal emotion conf < this → remap to neutral

MIN_CONFIDENCE        = 0.38    # below this → publish "neutral" regardless
VOTE_WINDOW           = 10      # majority vote over last N predictions
INFER_EVERY_N_FRAMES  = 3       # run inference once every N camera frames
MIN_FACE_SIZE         = 60      # minimum face pixel size for detection
DEFAULT_TEMPERATURE   = 2.0     # temperature scaling (higher = softer distribution)

# Calibration weights for FER2013 class imbalance
# Based on the fix docs: subtle adjustments proved better than extreme biases
CLASS_WEIGHTS = np.array([
    1.00,   # angry
    0.85,   # disgusted  ← slightly penalised (common false positive)
    1.00,   # fearful
    1.15,   # happy      ← slightly boosted
    0.95,   # neutral    ← slightly penalised (overrepresented in FER2013)
    1.00,   # sad
    1.05,   # surprised
], dtype=np.float32)

COLOR_MAP = {
    "angry":     (0,   0,   255),
    "disgusted": (0,   128, 128),
    "fearful":   (128, 0,   128),
    "happy":     (0,   200, 0  ),
    "neutral":   (180, 180, 180),
    "sad":       (255, 100, 100),
    "surprised": (0,   200, 200),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("emotion_v2")


# ─────────────────────────────────────────────────────────────
# Model loader — with clear error if file missing
# ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Load TF/Keras FER model. Exits with clear instructions if not found.
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not installed. Run: pip install tensorflow>=2.14.0")
        sys.exit(1)

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("")
        logger.error("To get the model, run from module12 directory:")
        logger.error("  python3 rebuild_improved.py")
        logger.error("  (generates fer_rebuilt_v2.h5 in current directory)")
        logger.error("")
        logger.error("Or specify path: --model /path/to/your_model.h5")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Model loaded: {model_path}")
        # Warmup pass to JIT-compile graph before real-time inference
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)
        logger.info("✅ Model warmed up")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Face detector (Haar Cascade)
# ─────────────────────────────────────────────────────────────

def build_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        logger.error("Failed to load Haar Cascade. Reinstall OpenCV.")
        sys.exit(1)
    return cascade


def detect_largest_face(frame: np.ndarray, cascade):
    """
    Detect faces and return the largest one (primary person).
    Returns (x, y, w, h) or None.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    if len(faces) == 0:
        return None
    # Return largest face by area
    return max(faces, key=lambda f: f[2] * f[3])


# ─────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """BGR face crop → (1, 224, 224, 3) float32 batch."""
    face_rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    return np.expand_dims(face_resized.astype(np.float32) / 255.0, axis=0)


def predict_emotion(model, face_bgr: np.ndarray, temperature: float):
    """
    Run inference on a face crop.
    Returns (label: str, confidence: float, all_probs: np.ndarray)
    """
    inp       = preprocess_face(face_bgr)
    raw_preds = model.predict(inp, verbose=0)[0]

    # Temperature scaling (softer distribution at T>1)
    scaled = (raw_preds / temperature).astype(np.float64)
    probs  = np.exp(scaled - scaled.max())
    probs /= probs.sum()

    # Class-weight calibration
    corrected = (probs * CLASS_WEIGHTS).astype(np.float64)
    corrected /= corrected.sum()

    idx        = int(np.argmax(corrected))
    label      = CLASSES[idx]
    confidence = float(corrected[idx])

    return label, confidence, corrected.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Temporal smoother — majority vote
# ─────────────────────────────────────────────────────────────

class EmotionSmoother:
    """
    Majority vote over the last N predictions.

    More stable than probability averaging because it:
    - Ignores single-frame noise spikes
    - Requires sustained evidence before changing output
    - Naturally handles the happy↔disgusted confusion
    """

    def __init__(self, window: int = VOTE_WINDOW):
        self._history   = deque(maxlen=window)
        self._conf_hist = deque(maxlen=window)

    def update(self, label: str, confidence: float) -> tuple:
        """
        Add a new prediction. Returns (smoothed_label, smoothed_confidence).
        """
        self._history.append(label)
        self._conf_hist.append(confidence)

        # Majority vote
        vote      = Counter(self._history).most_common(1)[0]
        top_label = vote[0]
        vote_frac = vote[1] / len(self._history)

        # Smooth confidence: mean of recent confidences for winning label
        matching_confs = [c for l, c in zip(self._history, self._conf_hist)
                          if l == top_label]
        smooth_conf = float(np.mean(matching_confs)) if matching_confs else confidence

        # Apply elderly-context remapping:
        # Marginal emotions with low confidence → neutral
        if top_label in MARGINAL_EMOTIONS and smooth_conf < MARGINAL_TO_NEUTRAL_THRESHOLD:
            return "neutral", smooth_conf

        return top_label, smooth_conf

    @property
    def history(self):
        return list(self._history)


# ─────────────────────────────────────────────────────────────
# Core engine (handles threading)
# ─────────────────────────────────────────────────────────────

class EmotionEngine:
    """
    Non-blocking emotion inference engine.

    Usage:
        engine = EmotionEngine(model_path, on_result=callback)
        engine.submit_frame(frame)   # non-blocking, call every camera frame
        engine.shutdown()
    """

    def __init__(self,
                 model_path : str,
                 on_result  = None,
                 on_active  = None,
                 temperature: float = DEFAULT_TEMPERATURE):
        """
        Args:
            model_path : path to .h5 model file
            on_result  : callback(label: str, confidence: float, all_probs: np.ndarray)
            on_active  : callback(face_detected: bool)
            temperature: temperature scaling value
        """
        self.on_result   = on_result
        self.on_active   = on_active
        self.temperature = temperature

        self._model   = load_model(model_path)
        self._cascade = build_face_detector()
        self._smoother = EmotionSmoother(window=VOTE_WINDOW)

        self._frame_q  = queue.Queue(maxsize=5)
        self._stop     = threading.Event()
        self._is_face  = False

        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="emotion_worker")
        self._thread.start()
        logger.info(f"EmotionEngine ready (temperature={temperature})")

    def submit_frame(self, frame: np.ndarray):
        """
        Non-blocking frame submission.
        Old frames are dropped if queue full (always shows latest).
        """
        try:
            # Drain old frames first (always process freshest)
            while not self._frame_q.empty():
                try:
                    self._frame_q.get_nowait()
                except queue.Empty:
                    break
            self._frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def _worker(self):
        """Background inference loop."""
        while not self._stop.is_set():
            try:
                frame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            face = detect_largest_face(frame, self._cascade)
            face_detected = face is not None

            # Notify on face presence change
            if face_detected != self._is_face:
                self._is_face = face_detected
                if self.on_active:
                    self.on_active(face_detected)

            if not face_detected:
                # No face → report neutral with zero confidence
                if self.on_result:
                    self.on_result("neutral", 0.0, None)
                continue

            x, y, w, h = face
            # Safety bounds
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            try:
                label, conf, all_probs = predict_emotion(
                    self._model, face_crop, self.temperature
                )
                smooth_label, smooth_conf = self._smoother.update(label, conf)

                # Confidence threshold gate
                if smooth_conf < MIN_CONFIDENCE:
                    smooth_label = "neutral"

                if self.on_result:
                    self.on_result(smooth_label, smooth_conf, all_probs)

            except Exception as e:
                logger.error(f"Inference error: {e}")

    def shutdown(self):
        self._stop.set()

    @property
    def is_face_detected(self) -> bool:
        return self._is_face


# ─────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────

def run_standalone(model_path: str, src: int,
                   temperature: float, debug: bool):
    print("=" * 55)
    print("  Module 12 — Emotion Detection v2")
    print("=" * 55)

    # State shared between threads
    state = {
        "label":     "neutral",
        "conf":      0.0,
        "all_probs": None,
        "active":    False,
    }

    def on_result(label, conf, probs):
        state["label"]     = label
        state["conf"]      = conf
        state["all_probs"] = probs

    def on_active(active):
        state["active"] = active
        if not active:
            state["label"] = "neutral"

    engine = EmotionEngine(
        model_path  = model_path,
        on_result   = on_result,
        on_active   = on_active,
        temperature = temperature,
    )

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {src}")
        sys.exit(1)

    cascade = build_face_detector()
    frame_idx = 0

    print(f"\n📷 Camera {src} started | debug={'on' if debug else 'off'} | Press 'q' to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Submit every Nth frame for inference
            if frame_idx % INFER_EVERY_N_FRAMES == 0:
                engine.submit_frame(frame)

            # Draw current result
            label  = state["label"]
            conf   = state["conf"]
            active = state["active"]
            color  = COLOR_MAP.get(label, (180, 180, 180))
            h, w   = frame.shape[:2]

            # Draw face box if face detected
            if active:
                face = detect_largest_face(frame, cascade)
                if face is not None:
                    fx, fy, fw, fh = face
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
                    cv2.putText(frame,
                                f"{label} {conf*100:.0f}%",
                                (fx, fy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

                    if debug and state["all_probs"] is not None:
                        top3 = sorted(zip(CLASSES, state["all_probs"]),
                                      key=lambda x: x[1], reverse=True)[:3]
                        for i, (em, p) in enumerate(top3):
                            cv2.putText(frame, f"{em}: {p*100:.0f}%",
                                        (fx, fy + fh + 18 + i*18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                        color if em == label else (160, 160, 160), 1)

            # Status bar
            status = f"{'🔴 ' + label.upper() if active else '⚪ No face'}"
            cv2.rectangle(frame, (0, 0), (360, 35), (0, 0, 0), -1)
            cv2.putText(frame, status, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

            print(f"  {label:12s} conf={conf*100:.0f}%  face={'yes' if active else 'no '}", end="\r")

            cv2.imshow("Emotion Detection v2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.shutdown()
        print("\n\nStopped.")


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class EmotionNodeV2(Node):
    """
    ROS2 emotion detection node.

    Publishes:
      /emotion_label      String — "neutral|happy|sad|fearful|angry|disgusted|surprised"
      /emotion_confidence Float32 — 0.0 to 1.0
      /emotion_active     Bool    — True when face detected in camera frame

    Decision engine subscribes to these to adapt dialogue tone:
      neutral  → standard interaction
      happy    → upbeat, positive responses
      sad      → supportive, gentle responses
      fearful  → calm, reassuring responses
      angry    → de-escalating responses
    """

    def __init__(self, model_path: str, src: int, temperature: float):
        super().__init__("emotion_node_v2")

        # ROS2 parameters
        self.declare_parameter("model_path",   model_path)
        self.declare_parameter("camera_src",   src)
        self.declare_parameter("temperature",  temperature)
        self.declare_parameter("min_confidence", MIN_CONFIDENCE)

        # Publishers
        self.pub_label = self.create_publisher(String,  "/emotion_label",      10)
        self.pub_conf  = self.create_publisher(Float32, "/emotion_confidence", 10)
        self.pub_act   = self.create_publisher(Bool,    "/emotion_active",     10)

        # Engine
        self.engine = EmotionEngine(
            model_path  = self.get_parameter("model_path").value,
            on_result   = self._on_result,
            on_active   = self._on_active,
            temperature = self.get_parameter("temperature").value,
        )

        # Camera thread
        self._src       = self.get_parameter("camera_src").value
        self._running   = True
        self._cam_thread = threading.Thread(target=self._camera_loop,
                                            daemon=True, name="emotion_cam")
        self._cam_thread.start()

        self.get_logger().info(
            f"Emotion node v2 started | camera={src} | "
            f"topics: /emotion_label, /emotion_confidence, /emotion_active"
        )

    def _camera_loop(self):
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            self.get_logger().error(f"Cannot open camera {self._src}")
            return

        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame_idx += 1
            if frame_idx % INFER_EVERY_N_FRAMES == 0:
                self.engine.submit_frame(frame)

        cap.release()

    def _on_result(self, label: str, confidence: float, all_probs):
        """Called from EmotionEngine background thread."""
        min_conf = self.get_parameter("min_confidence").value

        # Apply threshold
        if confidence < min_conf:
            label = "neutral"

        msg_label      = String()
        msg_label.data = label
        self.pub_label.publish(msg_label)

        msg_conf      = Float32()
        msg_conf.data = float(confidence)
        self.pub_conf.publish(msg_conf)

        self.get_logger().debug(f"Emotion: {label} ({confidence*100:.0f}%)")

    def _on_active(self, active: bool):
        """Called when face detection state changes."""
        msg_act      = Bool()
        msg_act.data = active
        self.pub_act.publish(msg_act)
        self.get_logger().info(
            f"Face {'detected' if active else 'lost'} → /emotion_active={'True' if active else 'False'}"
        )

    def destroy_node(self):
        self._running = False
        self.engine.shutdown()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Emotion detection node v2")
    p.add_argument("--ros",         action="store_true",
                   help="Run as ROS2 node")
    p.add_argument("--model",       type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "fer_rebuilt_v2.h5"),
                   help="Path to .h5 FER model file")
    p.add_argument("--src",         type=int,   default=0,
                   help="Camera device index")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Temperature scaling (higher=softer, default=2.0)")
    p.add_argument("--debug",       action="store_true",
                   help="Show top-3 emotion probabilities on screen")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ros:
        if not ROS_AVAILABLE:
            logger.error("rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = EmotionNodeV2(
            model_path  = args.model,
            src         = args.src,
            temperature = args.temperature,
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down emotion node v2")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(
            model_path  = args.model,
            src         = args.src,
            temperature = args.temperature,
            debug       = args.debug,
        )


if __name__ == "__main__":
    main()
