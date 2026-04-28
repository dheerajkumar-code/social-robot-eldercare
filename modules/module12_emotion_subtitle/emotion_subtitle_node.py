#!/usr/bin/env python3
"""
Module 12 — Emotion Subtitle Overlay (Camera-based)
---------------------------------------------------
- Loads pretrained FER model (.h5)
- Detects face using OpenCV Haar Cascade
- Predicts emotion class and overlays subtitle with emoji
- Displays real-time feed with color-coded emotion text

Usage:
  python3 emotion_subtitle_node.py --model fer_fixed_model.h5 --src 0
"""

import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------ Emotion Labels ------------------
CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

EMOJI_MAP = {
    "angry": "😡",
    "disgusted": "🤢",
    "fearful": "😨",
    "happy": "🙂",
    "neutral": "😐",
    "sad": "😢",
    "surprised": "😮"
}

COLOR_MAP = {
    "angry": (0, 0, 255),
    "disgusted": (0, 128, 128),
    "fearful": (128, 0, 128),
    "happy": (0, 200, 0),
    "neutral": (180, 180, 180),
    "sad": (255, 100, 100),
    "surprised": (0, 200, 200)
}


# ------------------ Subtitle Renderer ------------------
class EmotionSubtitle:
    def __init__(self):
        self.current_emotion = "neutral"
        self.last_update = time.time()
        self.fade = 1.0

    def update(self, new_emotion):
        if new_emotion != self.current_emotion:
            self.current_emotion = new_emotion
            self.last_update = time.time()
            self.fade = 1.0
        else:
            elapsed = time.time() - self.last_update
            self.fade = max(0.4, 1.0 - elapsed / 3.0)

    def draw(self, frame):
        h, w, _ = frame.shape
        emotion = self.current_emotion
        emoji = EMOJI_MAP.get(emotion, "😐")
        color = COLOR_MAP.get(emotion, (255, 255, 255))
        text = f"{emoji} {emotion.upper()}"

        font_scale = 1.2
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = int((w - tw) / 2)
        y = h - 40

        overlay = frame.copy()
        cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return cv2.addWeighted(overlay, self.fade, frame, 1 - self.fade, 0)


# ------------------ Helper Functions ------------------
def preprocess_face(face_bgr):
    """Resize and preprocess the face for model input"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    face_array = img_to_array(face_resized) / 255.0
    return np.expand_dims(face_array, axis=0)


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def draw_boxes(frame, detections):
    """Draw bounding boxes with emotion labels"""
    for (x, y, w, h, label, conf) in detections:
        color = COLOR_MAP.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} ({conf*100:.0f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


# ------------------ Main Function ------------------
def main():
    parser = argparse.ArgumentParser(description="Emotion subtitle overlay with camera feed")
    parser.add_argument("--model", type=str, required=True, help="Path to FER .h5 model file")
    parser.add_argument("--src", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    # Load FER model
    print("🎯 Loading FER model...")
    try:
        model = tf.keras.models.load_model(args.model)
        print(f"✅ Model loaded successfully from {args.model}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("❌ Could not load Haar Cascade.")
        return

    # Start video capture
    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f"❌ Failed to open camera {args.src}")
        return

    subtitle = EmotionSubtitle()
    print("📷 Camera started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame, face_cascade)
            detections = []

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                inp = preprocess_face(face_roi)
                preds = model.predict(inp, verbose=0)[0]
                score = tf.nn.softmax(preds)
                idx = int(tf.argmax(score))
                emotion = CLASSES[idx]
                conf = float(score[idx])

                detections.append((x, y, w, h, emotion, conf))

            # Update subtitle based on strongest detection
            if detections:
                detections.sort(key=lambda d: d[-1], reverse=True)
                subtitle.update(detections[0][4])
            else:
                subtitle.update("neutral")

            # Draw detections and subtitle
            frame = draw_boxes(frame, detections)
            frame = subtitle.draw(frame)

            cv2.imshow("Emotion Subtitle Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Exiting...")


if __name__ == "__main__":
    main()
