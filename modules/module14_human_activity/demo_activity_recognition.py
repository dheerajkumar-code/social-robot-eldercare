#!/usr/bin/env python3
"""
Demo script for Module 14 - Human Activity Recognition

This script demonstrates the activity recognition system by:
1. Loading a sample pose CSV file
2. Running inference on sliding windows
3. Displaying predictions with confidence scores

Usage:
    python3 demo_activity_recognition.py
    python3 demo_activity_recognition.py --activity standing
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add module to path
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(MODULE_DIR / "tests"))

try:
    import joblib
    from train_activity_node import load_csv_to_frames, extract_features_from_sequence
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install joblib scikit-learn")
    sys.exit(1)


def demo_activity_recognition(activity_name=None, model_path=None):
    """Run demo on a sample activity"""
    
    # Default paths
    if model_path is None:
        model_path = MODULE_DIR / "models" / "pose_activity_model_new.pkl"
        if not model_path.exists():
            model_path = MODULE_DIR / "models" / "pose_activity_model.pkl"
    
    pose_data_dir = MODULE_DIR / "pose_data"
    
    # Load model
    print("=" * 70)
    print("Module 14 - Human Activity Recognition Demo")
    print("=" * 70)
    print(f"\n📥 Loading model: {model_path.name}")
    
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            model = model_data.get("model", model_data)
            classes = model_data.get("classes", None)
        else:
            model = model_data
            classes = getattr(model, 'classes_', None)
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Supported activities: {list(classes)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Select activity to demo
    available_activities = sorted([d.name for d in pose_data_dir.iterdir() if d.is_dir()])
    
    if activity_name is None:
        activity_name = available_activities[0]
        print(f"\n📂 No activity specified, using: {activity_name}")
    elif activity_name not in available_activities:
        print(f"❌ Activity '{activity_name}' not found")
        print(f"Available activities: {available_activities}")
        return 1
    
    # Find CSV file for this activity
    activity_dir = pose_data_dir / activity_name
    csv_files = list(activity_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found for activity: {activity_name}")
        return 1
    
    csv_file = csv_files[0]
    print(f"📄 Using data file: {csv_file.name}")
    
    # Load frames
    print(f"\n🎬 Loading pose data...")
    try:
        frames, timestamps = load_csv_to_frames(str(csv_file))
        print(f"✅ Loaded {len(frames)} frames")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return 1
    
    # Run inference on sliding windows
    window_size = 38  # ~1.5 seconds at 25 fps
    step = 10
    
    print(f"\n🔍 Running inference (window={window_size} frames, step={step} frames)...")
    print("=" * 70)
    
    predictions = []
    
    for i in range(0, max(1, len(frames) - window_size + 1), step):
        end = i + window_size
        if end > len(frames):
            break
        
        window_frames = frames[i:end]
        window_timestamps = timestamps[i:end]
        
        # Extract features
        features = extract_features_from_sequence(window_frames, window_timestamps)
        
        # Predict
        prediction = model.predict(features.reshape(1, -1))[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features.reshape(1, -1))[0]
            pred_idx = list(classes).index(prediction)
            confidence = proba[pred_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(proba)[::-1][:3]
            top_predictions = [(classes[idx], proba[idx]) for idx in top_indices]
        else:
            confidence = 1.0
            top_predictions = [(prediction, 1.0)]
        
        predictions.append({
            'window': i,
            'prediction': prediction,
            'confidence': confidence,
            'top_predictions': top_predictions
        })
        
        # Display
        status = "✅" if prediction == activity_name else "❌"
        print(f"{status} Window {i:3d}-{end:3d}: {prediction:10s} ({confidence*100:5.1f}%)", end="")
        
        # Show top 3
        print(" | Top 3:", end="")
        for act, prob in top_predictions:
            print(f" {act}({prob*100:.0f}%)", end="")
        print()
    
    # Summary
    print("=" * 70)
    print("\n📊 SUMMARY")
    print("=" * 70)
    
    correct = sum(1 for p in predictions if p['prediction'] == activity_name)
    total = len(predictions)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"True activity:     {activity_name}")
    print(f"Total windows:     {total}")
    print(f"Correct:           {correct}")
    print(f"Accuracy:          {accuracy:.1f}%")
    
    # Most common prediction
    from collections import Counter
    pred_counts = Counter([p['prediction'] for p in predictions])
    most_common = pred_counts.most_common(1)[0] if pred_counts else (None, 0)
    
    print(f"Most common pred:  {most_common[0]} ({most_common[1]} times)")
    
    # Average confidence
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    print(f"Avg confidence:    {avg_confidence*100:.1f}%")
    
    print("\n" + "=" * 70)
    
    if accuracy >= 80:
        print("🎉 EXCELLENT performance!")
    elif accuracy >= 60:
        print("👍 GOOD performance")
    else:
        print("⚠️ Model may need more training data")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Demo for Module 14 Activity Recognition")
    parser.add_argument("--activity", type=str, help="Activity to demo (standing, sitting, walking, etc.)")
    parser.add_argument("--model", type=str, help="Path to model file")
    
    args = parser.parse_args()
    
    model_path = Path(args.model) if args.model else None
    
    return demo_activity_recognition(args.activity, model_path)


if __name__ == "__main__":
    sys.exit(main())
