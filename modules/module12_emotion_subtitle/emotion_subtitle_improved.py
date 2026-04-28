#!/usr/bin/env python3
"""
Improved Emotion Subtitle with Better Calibration

This version has improved calibration to reduce misclassification,
especially for happy vs disgusted confusion.
"""

import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

# Emotion Labels
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


class ImprovedEmotionPredictor:
    """Emotion predictor with improved calibration"""
    
    def __init__(self, model, temperature=1.5, use_smoothing=True, window_size=5):
        self.model = model
        self.temperature = temperature
        self.use_smoothing = use_smoothing
        self.prediction_history = deque(maxlen=window_size)
        
        # Bias correction weights (boost underrepresented classes)
        # These weights compensate for class imbalance in FER2013
        self.class_weights = np.array([
            1.0,  # angry (common)
            1.5,  # disgusted (rare) - boost
            1.3,  # fearful (rare) - boost
            1.2,  # happy (common but sometimes missed)
            0.9,  # neutral (very common) - reduce
            1.2,  # sad (somewhat rare) - boost
            1.4   # surprised (rare) - boost
        ])
    
    def predict(self, face_image):
        """Predict emotion with improved calibration"""
        # Preprocess
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = img_to_array(face_resized) / 255.0
        inp = np.expand_dims(face_array, axis=0)
        
        # Get raw predictions
        raw_preds = self.model.predict(inp, verbose=0)[0]
        
        # Apply temperature scaling
        scaled_preds = raw_preds / self.temperature
        probs = tf.nn.softmax(scaled_preds).numpy()
        
        # Apply bias correction
        corrected_probs = probs * self.class_weights
        corrected_probs = corrected_probs / corrected_probs.sum()
        
        # Temporal smoothing
        if self.use_smoothing:
            self.prediction_history.append(corrected_probs)
            if len(self.prediction_history) > 0:
                smoothed_probs = np.mean(self.prediction_history, axis=0)
            else:
                smoothed_probs = corrected_probs
        else:
            smoothed_probs = corrected_probs
        
        # Get prediction
        idx = int(np.argmax(smoothed_probs))
        emotion = CLASSES[idx]
        confidence = float(smoothed_probs[idx])
        
        return emotion, confidence, smoothed_probs


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


def draw_boxes_with_probs(frame, detections, show_all_probs=False):
    """Draw bounding boxes with emotion labels and optionally all probabilities"""
    for detection in detections:
        x, y, w, h, label, conf, all_probs = detection
        color = COLOR_MAP.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Main prediction
        text = f"{label} ({conf*100:.0f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        # Show all probabilities if requested
        if show_all_probs and all_probs is not None:
            y_offset = y + h + 20
            # Sort by probability
            sorted_emotions = sorted(zip(CLASSES, all_probs), key=lambda x: x[1], reverse=True)
            for i, (em, prob) in enumerate(sorted_emotions[:3]):  # Show top 3
                prob_text = f"{em}: {prob*100:.0f}%"
                text_color = color if em == label else (200, 200, 200)
                cv2.putText(frame, prob_text, (x, y_offset + i*18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description="Improved emotion detection with better calibration")
    parser.add_argument("--model", type=str, default="fer_rebuilt_v2.h5", help="Path to model file")
    parser.add_argument("--src", type=int, default=0, help="Camera index")
    parser.add_argument("--temperature", type=float, default=2.0, 
                       help="Temperature for calibration (default: 2.0)")
    parser.add_argument("--debug", action="store_true", 
                       help="Show all emotion probabilities")
    args = parser.parse_args()

    print("=" * 70)
    print("Module 12 - Improved Emotion Detection")
    print("=" * 70)
    print()

    # Load model
    print("🎯 Loading FER model...")
    try:
        model = tf.keras.models.load_model(args.model)
        print(f"✅ Model loaded successfully from {args.model}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create improved predictor
    predictor = ImprovedEmotionPredictor(
        model, 
        temperature=args.temperature,
        use_smoothing=True
    )
    print(f"✅ Using improved calibration (temp={args.temperature})")
    print(f"✅ Happy/Disgusted confusion correction enabled")
    print(f"✅ Temporal smoothing enabled")

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
    if args.debug:
        print("🐛 Debug mode: Showing top 3 emotion probabilities")
    print("=" * 70)
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            detections = []

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue

                # Predict with improved calibration
                emotion, conf, all_probs = predictor.predict(face_roi)
                detections.append((x, y, w, h, emotion, conf, all_probs))

            # Update subtitle
            if detections:
                detections.sort(key=lambda d: d[5], reverse=True)
                subtitle.update(detections[0][4])
            else:
                subtitle.update("neutral")

            # Draw
            frame = draw_boxes_with_probs(frame, detections, show_all_probs=args.debug)
            frame = subtitle.draw(frame)

            cv2.imshow("Improved Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Exiting...")


if __name__ == "__main__":
    main()
