#!/usr/bin/env python3
"""
Test the emotion model with various test images to see what it actually predicts
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Load model
print("Loading model...")
model = tf.keras.models.load_model('fer_rebuilt_v2.h5')
print("✅ Model loaded")

CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def test_with_webcam():
    """Test model with live webcam to see prediction distribution"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n📷 Testing with webcam...")
    print("Make different expressions and see what the model predicts")
    print("Press 'q' to quit\n")
    
    prediction_counts = {emotion: 0 for emotion in CLASSES}
    total_predictions = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Preprocess
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (224, 224))
                face_array = face_resized.astype(np.float32) / 255.0
                inp = np.expand_dims(face_array, axis=0)
                
                # Predict
                preds = model.predict(inp, verbose=0)[0]
                probs = tf.nn.softmax(preds).numpy()
                
                idx = int(np.argmax(probs))
                emotion = CLASSES[idx]
                conf = probs[idx]
                
                prediction_counts[emotion] += 1
                total_predictions += 1
                
                # Draw
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show all probabilities
                y_offset = y + h + 20
                for i, (em, prob) in enumerate(zip(CLASSES, probs)):
                    text = f"{em}: {prob*100:.1f}%"
                    color = (0, 255, 0) if i == idx else (200, 200, 200)
                    cv2.putText(frame, text, (x, y_offset + i*20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show statistics
            if total_predictions > 0:
                y_pos = 30
                cv2.putText(frame, "Prediction Distribution:", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                for emotion in CLASSES:
                    y_pos += 25
                    count = prediction_counts[emotion]
                    pct = (count / total_predictions) * 100
                    text = f"{emotion}: {count} ({pct:.1f}%)"
                    cv2.putText(frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Model Testing", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Prediction Distribution Summary")
    print("=" * 50)
    for emotion in CLASSES:
        count = prediction_counts[emotion]
        pct = (count / total_predictions * 100) if total_predictions > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{emotion:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    print(f"\nTotal predictions: {total_predictions}")

if __name__ == "__main__":
    test_with_webcam()
