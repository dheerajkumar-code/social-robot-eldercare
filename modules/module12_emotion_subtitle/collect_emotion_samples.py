#!/usr/bin/env python3
"""
Emotion Sample Collection Tool for Module 12

This script helps you collect facial expression samples for all 7 emotions.
It will:
1. Capture your face from webcam
2. Detect and crop the face
3. Save samples for each emotion category
4. Create a validation dataset

Usage:
    python3 collect_emotion_samples.py
"""

import cv2
import os
import time
from datetime import datetime
from pathlib import Path

# Emotion categories
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

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

INSTRUCTIONS = {
    "angry": "Make an ANGRY face - furrow your brows, frown",
    "disgusted": "Make a DISGUSTED face - wrinkle your nose, look repulsed",
    "fearful": "Make a FEARFUL face - widen your eyes, open mouth slightly",
    "happy": "Make a HAPPY face - smile genuinely, show teeth",
    "neutral": "Make a NEUTRAL face - relax, no expression",
    "sad": "Make a SAD face - frown, look down slightly",
    "surprised": "Make a SURPRISED face - raise eyebrows, open mouth"
}


class EmotionSampleCollector:
    def __init__(self, output_dir="emotion_samples", samples_per_emotion=10):
        self.output_dir = Path(output_dir)
        self.samples_per_emotion = samples_per_emotion
        self.face_cascade = None
        self.cap = None
        
        # Create output directories
        for emotion in EMOTIONS:
            emotion_dir = self.output_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_camera(self, camera_index=0):
        """Initialize camera and face detector"""
        print("📷 Setting up camera...")
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        if self.face_cascade.empty():
            print("❌ Could not load face detector")
            return False
        
        # Open camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {camera_index}")
            return False
        
        print("✅ Camera ready!")
        return True
    
    def detect_face(self, frame):
        """Detect face in frame and return cropped face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add margin
        margin = int(0.2 * w)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        face_crop = frame[y:y+h, x:x+w]
        return face_crop, (x, y, w, h)
    
    def save_sample(self, face_crop, emotion, sample_num):
        """Save face sample to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{emotion}_{sample_num:02d}_{timestamp}.jpg"
        filepath = self.output_dir / emotion / filename
        
        # Resize to standard size
        face_resized = cv2.resize(face_crop, (224, 224))
        cv2.imwrite(str(filepath), face_resized)
        
        return filepath
    
    def collect_for_emotion(self, emotion):
        """Collect samples for a specific emotion"""
        print("\n" + "=" * 70)
        emoji = EMOJI_MAP[emotion]
        print(f"{emoji} Collecting samples for: {emotion.upper()}")
        print("=" * 70)
        print(f"\n📝 Instructions: {INSTRUCTIONS[emotion]}")
        print(f"\n🎯 Goal: Capture {self.samples_per_emotion} samples")
        print("\nControls:")
        print("  SPACE - Capture sample")
        print("  S     - Skip this emotion")
        print("  Q     - Quit collection")
        print("\nGet ready...")
        
        samples_collected = 0
        countdown = 3
        countdown_start = time.time()
        
        while samples_collected < self.samples_per_emotion:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect face
            face_crop, bbox = self.detect_face(frame)
            
            # Draw on frame
            display_frame = frame.copy()
            
            if bbox is not None:
                x, y, w, h = bbox
                color = COLOR_MAP[emotion]
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(display_frame, "Face Detected", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show countdown
            if countdown > 0:
                elapsed = time.time() - countdown_start
                remaining = countdown - int(elapsed)
                if remaining > 0:
                    cv2.putText(display_frame, f"Starting in {remaining}...", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                else:
                    countdown = 0
            
            # Show instructions
            emoji_text = f"{emoji} {emotion.upper()}"
            cv2.putText(display_frame, emoji_text, (10, display_frame.shape[0] - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_MAP[emotion], 3)
            
            instruction = INSTRUCTIONS[emotion]
            cv2.putText(display_frame, instruction[:50], (10, display_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show progress
            progress = f"Samples: {samples_collected}/{self.samples_per_emotion}"
            cv2.putText(display_frame, progress, (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Emotion Sample Collection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and face_crop is not None and countdown == 0:
                # Capture sample
                filepath = self.save_sample(face_crop, emotion, samples_collected + 1)
                samples_collected += 1
                print(f"✅ Captured sample {samples_collected}/{self.samples_per_emotion}: {filepath.name}")
                
                # Brief pause
                time.sleep(0.3)
            
            elif key == ord('s'):
                print(f"⏭️  Skipping {emotion}")
                return False
            
            elif key == ord('q'):
                print("🛑 Quitting collection")
                return None
        
        print(f"\n🎉 Completed {emotion}! Collected {samples_collected} samples.")
        time.sleep(1)
        return True
    
    def collect_all(self):
        """Collect samples for all emotions"""
        print("\n" + "=" * 70)
        print("Emotion Sample Collection Tool")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"Samples per emotion: {self.samples_per_emotion}")
        print(f"\nEmotions to collect: {', '.join(EMOTIONS)}")
        
        if not self.setup_camera():
            return False
        
        print("\n🎬 Starting collection in 3 seconds...")
        time.sleep(3)
        
        completed = []
        skipped = []
        
        for emotion in EMOTIONS:
            result = self.collect_for_emotion(emotion)
            
            if result is None:
                # User quit
                break
            elif result:
                completed.append(emotion)
            else:
                skipped.append(emotion)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "=" * 70)
        print("Collection Summary")
        print("=" * 70)
        print(f"\n✅ Completed: {len(completed)}/{len(EMOTIONS)} emotions")
        if completed:
            print(f"   {', '.join(completed)}")
        
        if skipped:
            print(f"\n⏭️  Skipped: {', '.join(skipped)}")
        
        print(f"\n📁 Samples saved to: {self.output_dir.absolute()}")
        
        # Count total samples
        total_samples = 0
        for emotion in EMOTIONS:
            emotion_dir = self.output_dir / emotion
            samples = list(emotion_dir.glob("*.jpg"))
            if samples:
                print(f"   {emotion}: {len(samples)} samples")
                total_samples += len(samples)
        
        print(f"\n📊 Total samples collected: {total_samples}")
        print("\n✅ Collection complete!")
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect emotion samples for Module 12")
    parser.add_argument("--output", type=str, default="emotion_samples",
                       help="Output directory for samples")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of samples per emotion")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index")
    
    args = parser.parse_args()
    
    collector = EmotionSampleCollector(
        output_dir=args.output,
        samples_per_emotion=args.samples
    )
    
    try:
        collector.collect_all()
    except KeyboardInterrupt:
        print("\n\n🛑 Collection interrupted by user")
        collector.cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        collector.cleanup()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
