#!/usr/bin/env python3
"""
Validation script for Module 14 - Human Activity Recognition

This script validates the trained model by:
1. Loading the model
2. Testing predictions on sample data
3. Generating a detailed report
4. Creating visualizations (if matplotlib available)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add module to path
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

try:
    import joblib
except ImportError:
    print("❌ joblib not found. Install with: pip install joblib")
    sys.exit(1)

# Import training functions directly
sys.path.insert(0, str(MODULE_DIR / "tests"))
try:
    from train_activity_node import (
        load_csv_to_frames,
        extract_features_from_sequence
    )
except ImportError as e:
    print(f"❌ Could not import training functions: {e}")
    sys.exit(1)


def load_model(model_path):
    """Load the trained model"""
    print(f"📥 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None
    
    try:
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            model = model_data.get("model", model_data)
            classes = model_data.get("classes", None)
        else:
            model = model_data
            classes = getattr(model, 'classes_', None)
        
        print(f"✅ Model loaded: {type(model).__name__}")
        if classes is not None:
            print(f"✅ Classes: {list(classes)}")
        
        return model, classes
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None


def validate_on_pose_data(model, classes, pose_data_dir, window_size=30):
    """Validate model on all pose data"""
    print(f"\n📊 Validating on pose data from: {pose_data_dir}")
    
    results = defaultdict(lambda: {"correct": 0, "total": 0, "predictions": []})
    
    activities = sorted([d for d in os.listdir(pose_data_dir) 
                        if os.path.isdir(os.path.join(pose_data_dir, d))])
    
    for activity in activities:
        activity_dir = os.path.join(pose_data_dir, activity)
        csv_files = list(Path(activity_dir).glob("*.csv"))
        
        print(f"\n  Testing activity: {activity} ({len(csv_files)} files)")
        
        for csv_file in csv_files:
            try:
                # Load frames from CSV
                frames, timestamps = load_csv_to_frames(str(csv_file))
                
                if len(frames) < window_size:
                    print(f"    ⚠️ Skipping {csv_file.name} (only {len(frames)} frames)")
                    continue
                
                # Test on multiple windows from this file
                num_windows = max(1, (len(frames) - window_size) // 10 + 1)
                
                for i in range(num_windows):
                    start = min(i * 10, len(frames) - window_size)
                    end = start + window_size
                    
                    window_frames = frames[start:end]
                    window_timestamps = timestamps[start:end]
                    
                    # Extract features
                    features = extract_features_from_sequence(window_frames, window_timestamps)
                    
                    # Predict
                    prediction = model.predict(features.reshape(1, -1))[0]
                    
                    # Get probability if available
                    confidence = 1.0
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features.reshape(1, -1))[0]
                        if classes is not None:
                            pred_idx = list(classes).index(prediction)
                            confidence = proba[pred_idx]
                        else:
                            confidence = max(proba)
                    
                    # Record result
                    results[activity]["total"] += 1
                    results[activity]["predictions"].append({
                        "file": csv_file.name,
                        "window": i,
                        "prediction": prediction,
                        "confidence": confidence,
                        "correct": prediction == activity
                    })
                    
                    if prediction == activity:
                        results[activity]["correct"] += 1
                
            except Exception as e:
                print(f"    ❌ Error processing {csv_file.name}: {e}")
    
    return results


def print_validation_report(results):
    """Print detailed validation report"""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    
    total_correct = 0
    total_samples = 0
    
    for activity in sorted(results.keys()):
        correct = results[activity]["correct"]
        total = results[activity]["total"]
        accuracy = (correct / total * 100) if total > 0 else 0
        
        total_correct += correct
        total_samples += total
        
        print(f"\n{activity.upper()}")
        print(f"  Samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.1f}%")
        
        # Show some example predictions
        predictions = results[activity]["predictions"]
        if predictions:
            print(f"  Sample predictions:")
            for pred in predictions[:3]:  # Show first 3
                status = "✓" if pred["correct"] else "✗"
                print(f"    {status} {pred['file'][:30]:30s} → {pred['prediction']:10s} ({pred['confidence']*100:.0f}%)")
    
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"OVERALL ACCURACY: {overall_accuracy:.1f}% ({total_correct}/{total_samples})")
    print("=" * 70)
    
    return overall_accuracy


def create_confusion_matrix(results, classes):
    """Create confusion matrix from results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Build confusion matrix
        n_classes = len(classes)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for true_activity in results:
            if true_activity not in class_to_idx:
                continue
            true_idx = class_to_idx[true_activity]
            
            for pred in results[true_activity]["predictions"]:
                pred_activity = pred["prediction"]
                if pred_activity not in class_to_idx:
                    continue
                pred_idx = class_to_idx[pred_activity]
                confusion[true_idx, pred_idx] += 1
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Activity Recognition')
        
        output_path = MODULE_DIR / "validation_confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Confusion matrix saved to: {output_path}")
        plt.close()
        
    except ImportError:
        print("\n⚠️ matplotlib/seaborn not available, skipping confusion matrix visualization")
    except Exception as e:
        print(f"\n⚠️ Error creating confusion matrix: {e}")


def main():
    """Main validation function"""
    print("=" * 70)
    print("Module 14 - Human Activity Recognition Validation")
    print("=" * 70)
    
    # Paths
    model_path = MODULE_DIR / "models" / "pose_activity_model_new.pkl"
    pose_data_dir = MODULE_DIR / "pose_data"
    
    # Check if new model exists, otherwise use old one
    if not model_path.exists():
        model_path = MODULE_DIR / "models" / "pose_activity_model.pkl"
        print(f"⚠️ Using original model (new model not found)")
    
    # Load model
    model, classes = load_model(model_path)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return 1
    
    # Validate on pose data
    results = validate_on_pose_data(model, classes, pose_data_dir)
    
    if not results:
        print("❌ No validation results. Check pose data directory.")
        return 1
    
    # Print report
    accuracy = print_validation_report(results)
    
    # Create confusion matrix if possible
    if classes is not None:
        create_confusion_matrix(results, classes)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✅ Model: {model_path.name}")
    print(f"✅ Overall Accuracy: {accuracy:.1f}%")
    print(f"✅ Activities tested: {len(results)}")
    
    if accuracy >= 80:
        print("\n🎉 Model performance is GOOD (≥80%)")
        return 0
    elif accuracy >= 60:
        print("\n⚠️ Model performance is MODERATE (60-80%)")
        print("   Consider collecting more training data")
        return 0
    else:
        print("\n❌ Model performance is POOR (<60%)")
        print("   Need more training data or feature engineering")
        return 1


if __name__ == "__main__":
    sys.exit(main())
