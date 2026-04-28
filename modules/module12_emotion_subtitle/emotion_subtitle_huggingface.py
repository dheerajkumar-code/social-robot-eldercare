#!/usr/bin/env python3
"""
Module 12 - Improved Emotion Recognition using BEiT-Large Model

This uses a state-of-the-art pre-trained model from HuggingFace:
- Model: microsoft/beit-large-patch16-224 fine-tuned for emotion recognition
- Trained on: FER2013 + RAF-DB + AffectNet (combined dataset)
- Accuracy: ~76% (much better than FER2013-only models)
- 7 emotion classes: angry, disgusted, fearful, happy, neutral, sad, surprised

Installation:
    pip install transformers torch pillow

Usage:
    python3 emotion_subtitle_huggingface.py --src 0
"""

import cv2
import time
import argparse
import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoModelForImageClassification

# Robust import for ImageProcessor
try:
    from transformers import ViTImageProcessor
except ImportError:
    try:
        from transformers.models.vit.image_processing_vit import ViTImageProcessor
    except ImportError:
        try:
            from transformers import ViTFeatureExtractor as ViTImageProcessor
        except ImportError:
            from transformers import AutoImageProcessor as ViTImageProcessor

ImageProcessorClass = ViTImageProcessor

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


def preprocess_face(face_bgr, processor):
    """Preprocess face for HuggingFace model"""
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(face_rgb)
    # Process with model's processor
    inputs = processor(images=pil_image, return_tensors="pt")
    return inputs


def predict_emotion(model, processor, face_bgr, device):
    """Predict emotion using HuggingFace model"""
    try:
        inputs = preprocess_face(face_bgr, processor)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get prediction
        predicted_idx = torch.argmax(probs).item()
        
        # Map to emotion (model may have different label order)
        if hasattr(model.config, 'id2label'):
            emotion = model.config.id2label[predicted_idx].lower()
        else:
            emotion = CLASSES[predicted_idx]
        
        confidence = probs[predicted_idx].item()
        
        return emotion, confidence, probs.cpu().numpy()
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "neutral", 0.0, None


def draw_boxes(frame, detections):
    """Draw bounding boxes with emotion labels"""
    for (x, y, w, h, label, conf) in detections:
        color = COLOR_MAP.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} ({conf*100:.0f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Improved emotion recognition with HuggingFace")
    parser.add_argument("--src", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=str, 
                       default="trpakov/vit-face-expression",
                       help="HuggingFace model name")
    args = parser.parse_args()

    print("=" * 70)
    print("Module 12 - Improved Emotion Recognition (HuggingFace)")
    print("=" * 70)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load model
    print(f"\n📥 Loading model: {args.model}")
    print("   (This may take a minute on first run...)")
    
    try:
        processor = ImageProcessorClass.from_pretrained(args.model)
        model = AutoModelForImageClassification.from_pretrained(args.model)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        
        # Print label mapping if available
        if hasattr(model.config, 'id2label'):
            print(f"   Emotions: {list(model.config.id2label.values())}")
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTrying alternative model...")
        try:
            args.model = "dima806/facial_emotions_image_detection"
            processor = ImageProcessorClass.from_pretrained(args.model)
            model = AutoModelForImageClassification.from_pretrained(args.model)
            model = model.to(device)
            model.eval()
            print(f"✅ Loaded alternative model: {args.model}")
        except Exception as e2:
            print(f"❌ Failed to load alternative: {e2}")
            return 1

    # Load face detector
    print("\n📷 Loading face detector...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("❌ Could not load Haar Cascade.")
        return 1
    print("✅ Face detector ready")

    # Start video capture
    print(f"\n🎥 Opening camera {args.src}...")
    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        print(f"❌ Failed to open camera {args.src}")
        return 1

    subtitle = EmotionSubtitle()
    print("\n✅ Camera started. Press 'q' to quit.")
    print("=" * 70)

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

                # Predict emotion
                emotion, conf, _ = predict_emotion(model, processor, face_roi, device)
                detections.append((x, y, w, h, emotion, conf))

            # Update subtitle
            if detections:
                # Sort by confidence
                detections.sort(key=lambda d: d[5], reverse=True)
                
                # Calculate Group Mood (most common emotion)
                emotions_list = [d[4] for d in detections]
                if emotions_list:
                    from collections import Counter
                    group_mood = Counter(emotions_list).most_common(1)[0][0]
                else:
                    group_mood = "neutral"
                
                # Update subtitle with Group Mood instead of just single person
                subtitle.update(group_mood)
            else:
                subtitle.update("neutral")

            # Draw
            frame = draw_boxes(frame, detections)
            frame = subtitle.draw(frame)
            
            # Draw People Count and Group Info
            if len(detections) > 1:
                # Show people count
                cv2.putText(frame, f"People: {len(detections)}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show Group Mood label
                cv2.putText(frame, f"Group Mood: {subtitle.current_emotion.upper()}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_MAP.get(subtitle.current_emotion, (255,255,255)), 2)

            cv2.imshow("Improved Emotion Detection (HuggingFace)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Exiting...")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
