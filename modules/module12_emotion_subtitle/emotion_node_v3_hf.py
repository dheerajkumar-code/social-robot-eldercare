#!/usr/bin/env python3
"""
Module 12 — Emotion Detection Node v3 (HuggingFace PyTorch)
-----------------------------------------------------------
Uses a ViT model from HuggingFace for high-accuracy (~76%) emotion recognition.
Completely fixes the "always disgusted" issue from the FER2013 TensorFlow model.

Standalone:
  python3 emotion_node_v3_hf.py --src 0

ROS2:
  python3 emotion_node_v3_hf.py --ros --src 0

Published Topics:
  /emotion_label      std_msgs/String
  /emotion_confidence std_msgs/Float32
  /emotion_active     std_msgs/Bool
"""

import sys
import time
import threading
import argparse
import logging
import numpy as np

import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emotion_v3_hf")

try:
    import torch
    from transformers import AutoModelForImageClassification
    try:
        from transformers import ViTImageProcessor as ImageProcessorClass
    except ImportError:
        try:
            from transformers.models.vit.image_processing_vit import ViTImageProcessor as ImageProcessorClass
        except ImportError:
            from transformers import AutoImageProcessor as ImageProcessorClass
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32, Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Display colours per emotion
COLOR_MAP = {
    "angry":     (0,   0,   255),
    "disgusted": (0,   128, 128),
    "fearful":   (128, 0,   128),
    "happy":     (0,   200, 0  ),
    "neutral":   (180, 180, 180),
    "sad":       (255, 100, 100),
    "surprised": (0,   200, 200),
}

DEFAULT_MODEL = "trpakov/vit-face-expression"


# ──────────────────────────────────────────────────────────────
# Core inference engine (no ROS dependency)
# ──────────────────────────────────────────────────────────────

class HFEmotionEngine:
    """
    Wraps a HuggingFace ViT model for real-time emotion detection.
    Completely independent of ROS — can be used in standalone or ROS modes.
    """

    def __init__(self, model_name: str, on_result=None, on_active=None):
        if not TORCH_AVAILABLE:
            logger.error("PyTorch/Transformers not installed. Run: pip install torch transformers pillow")
            sys.exit(1)

        self.on_result = on_result
        self.on_active = on_active
        self._is_face = False
        self._running = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading model: {model_name} (cached after first download)...")
        try:
            self.processor = ImageProcessorClass.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("✅ HuggingFace model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            sys.exit(1)

        # Face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("Failed to load Haar Cascade face detector.")
            sys.exit(1)

    def _predict(self, face_bgr):
        """Run HuggingFace inference on a face crop."""
        try:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

            predicted_idx = torch.argmax(probs).item()

            if hasattr(self.model.config, 'id2label'):
                emotion = self.model.config.id2label[predicted_idx].lower()
            else:
                emotion = list(COLOR_MAP.keys())[predicted_idx]

            confidence = probs[predicted_idx].item()
            return emotion, confidence, probs.cpu().numpy()

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return "neutral", 0.0, None

    def get_id2label(self):
        if hasattr(self.model.config, 'id2label'):
            return self.model.config.id2label
        return {}

    def process_frame(self, frame):
        """
        Detect face and predict emotion for one frame.
        Calls on_result(emotion, conf, probs) and on_active(bool) callbacks.
        Returns (frame_annotated, emotion, conf, face_detected).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        face_detected = len(faces) > 0

        # Notify on state change
        if face_detected != self._is_face:
            self._is_face = face_detected
            if self.on_active:
                self.on_active(face_detected)
            logger.info(f"Face {'detected' if face_detected else 'lost'}")

        if not face_detected:
            if self.on_result:
                self.on_result("neutral", 0.0, None)
            return frame, "neutral", 0.0, False

        # Largest face only
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = frame[max(0, y):y+h, max(0, x):x+w]
        if face_roi.size == 0:
            return frame, "neutral", 0.0, False

        emotion, conf, probs = self._predict(face_roi)

        if self.on_result:
            self.on_result(emotion, conf, probs)

        # Annotate frame
        color = COLOR_MAP.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{emotion} ({conf*100:.0f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Draw top-3 emotion probs below box
        if probs is not None:
            id2label = self.get_id2label()
            top3 = np.argsort(probs)[-3:][::-1]
            for i, idx in enumerate(top3):
                em = id2label.get(int(idx), str(idx)).lower()
                p = probs[idx]
                txt_color = color if em == emotion else (160, 160, 160)
                cv2.putText(frame, f"{em}: {p*100:.0f}%",
                            (x, y + h + 18 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_color, 1)

        return frame, emotion, conf, True


# ──────────────────────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────────────────────

def run_standalone(src: int, model_name: str):
    print("=" * 60)
    print("  Module 12 — Emotion Detection V3 (HuggingFace PyTorch)")
    print("=" * 60)

    engine = HFEmotionEngine(model_name=model_name)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {src}")
        sys.exit(1)

    logger.info(f"Camera {src} started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame, emotion, conf, face_found = engine.process_frame(frame)

            status = f"{emotion.upper()}  {conf*100:.0f}%  face={'yes' if face_found else 'no '}"
            print(f"  {status}", end="\r")

            cv2.imshow("Emotion Detection V3 (HuggingFace)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n\nStopped.")


# ──────────────────────────────────────────────────────────────
# ROS2 Node
# ──────────────────────────────────────────────────────────────

class EmotionNodeV3(Node):
    def __init__(self, src: int, model_name: str, headless: bool):
        super().__init__("emotion_node_v3_hf")

        self.declare_parameter("camera_src", src)
        self.declare_parameter("model_name", model_name)
        self.declare_parameter("headless", headless)

        self._src = self.get_parameter("camera_src").value
        self._headless = self.get_parameter("headless").value
        _model = self.get_parameter("model_name").value

        # Publishers
        self.pub_label = self.create_publisher(String,  "/emotion_label",      10)
        self.pub_conf  = self.create_publisher(Float32, "/emotion_confidence", 10)
        self.pub_act   = self.create_publisher(Bool,    "/emotion_active",     10)

        self.engine = HFEmotionEngine(
            model_name=_model,
            on_result=self._on_result,
            on_active=self._on_active,
        )

        self._running = True
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True, name="emotion_v3_cam")
        self._cam_thread.start()

        self.get_logger().info(
            f"✅ EmotionNodeV3 started | camera={src} | model={_model}"
        )

    def _on_result(self, emotion: str, confidence: float, probs):
        msg_label = String()
        msg_label.data = emotion
        self.pub_label.publish(msg_label)

        msg_conf = Float32()
        msg_conf.data = float(confidence)
        self.pub_conf.publish(msg_conf)

        self.get_logger().debug(f"Emotion: {emotion} ({confidence*100:.0f}%)")

    def _on_active(self, active: bool):
        msg_act = Bool()
        msg_act.data = active
        self.pub_act.publish(msg_act)
        self.get_logger().info(f"Face {'detected' if active else 'lost'} → /emotion_active={active}")

    def _camera_loop(self):
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            self.get_logger().error(f"Cannot open camera {self._src}")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame, _, _, _ = self.engine.process_frame(frame)

            if not self._headless:
                cv2.imshow("Emotion Node V3 (HuggingFace)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False

        cap.release()
        cv2.destroyAllWindows()

    def destroy_node(self):
        self._running = False
        super().destroy_node()


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Emotion Detection Node V3 (HuggingFace)")
    parser.add_argument("--ros",      action="store_true", help="Run as ROS2 node")
    parser.add_argument("--src",      type=int, default=0, help="Camera index")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--headless", action="store_true", help="No display window (ROS mode only)")
    args = parser.parse_args()

    if args.ros:
        if not ROS_AVAILABLE:
            logger.error("rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = EmotionNodeV3(src=args.src, model_name=args.model, headless=args.headless)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down emotion node v3...")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(src=args.src, model_name=args.model)


if __name__ == "__main__":
    main()
