#!/usr/bin/env python3
"""
Module 12 - Emotion Detection Launcher
Automatically selects the best available model and runs emotion detection
"""

import os
import sys
import subprocess
import argparse

def check_pytorch():
    """Check if PyTorch is available and compatible with transformers"""
    # Use subprocess to avoid import caching issues
    test_code = """
import sys
try:
    import torch
    from transformers import AutoModelForImageClassification
    print("SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, timeout=10, text=True)
    
    # Check if transformers disabled PyTorch due to version mismatch
    if "Disabling PyTorch" in result.stdout or "Disabling PyTorch" in result.stderr:
        return False
    
    return result.returncode == 0 and "SUCCESS" in result.stdout

def check_tensorflow():
    """Check if TensorFlow is available"""
    test_code = """
import sys
try:
    import tensorflow as tf
    sys.exit(0)
except Exception:
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, timeout=10)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run emotion detection with best available model")
    parser.add_argument("--src", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    parser.add_argument("--force-pytorch", action="store_true", help="Force PyTorch model (will fail if not installed)")
    parser.add_argument("--force-tensorflow", action="store_true", help="Force TensorFlow model")
    args = parser.parse_args()

    print("=" * 70)
    print("Module 12 - Emotion Detection System")
    print("=" * 70)
    print()

    # Check available frameworks
    has_pytorch = check_pytorch()
    has_tensorflow = check_tensorflow()

    print(f"🔍 Checking available frameworks:")
    print(f"   PyTorch: {'✅ Available' if has_pytorch else '❌ Not installed'}")
    print(f"   TensorFlow: {'✅ Available' if has_tensorflow else '❌ Not installed'}")
    print()

    # Determine which model to use
    use_pytorch = False
    use_tensorflow = False

    if args.force_pytorch:
        if not has_pytorch:
            print("❌ Error: PyTorch not installed but --force-pytorch specified")
            print("   Install with: pip install torch transformers")
            return 1
        use_pytorch = True
    elif args.force_tensorflow:
        if not has_tensorflow:
            print("❌ Error: TensorFlow not installed but --force-tensorflow specified")
            return 1
        use_tensorflow = True
    else:
        # Auto-select: prefer PyTorch (more accurate) if available
        if has_pytorch:
            use_pytorch = True
            print("🎯 Auto-selected: PyTorch (HuggingFace ViT model - higher accuracy)")
        elif has_tensorflow:
            use_tensorflow = True
            print("🎯 Auto-selected: TensorFlow (FER2013 model)")
        else:
            print("❌ Error: Neither PyTorch nor TensorFlow is installed!")
            print("   Install one with:")
            print("   - PyTorch: pip install torch transformers")
            print("   - TensorFlow: pip install tensorflow")
            return 1

    print()

    # Run the appropriate script
    if use_pytorch:
        print("🚀 Launching PyTorch-based emotion detection...")
        print("   Model: HuggingFace ViT (trpakov/vit-face-expression)")
        print("   Expected accuracy: ~76%")
        print()
        cmd = ["python3", "emotion_subtitle_huggingface.py", "--src", str(args.src)]
        
    elif use_tensorflow:
        print("🚀 Launching TensorFlow-based emotion detection...")
        print("   Model: FER2013 EfficientNetB7")
        print("   Expected accuracy: ~65%")
        print()
        
        # Check if model file exists
        model_file = "fer_rebuilt_v2.h5"
        if not os.path.exists(model_file):
            print(f"⚠️  Model file '{model_file}' not found. Rebuilding...")
            rebuild_result = subprocess.run(["python3", "rebuild_improved.py"])
            if rebuild_result.returncode != 0:
                print("❌ Failed to rebuild model")
                return 1
        
        # Use the original enhanced version
        cmd = ["python3", "emotion_subtitle_enhanced.py", 
               "--model", model_file, "--src", str(args.src)]
        
        if args.debug:
            cmd.append("--debug")

    print("=" * 70)
    print("Press 'q' in the video window to quit")
    print("=" * 70)
    print()

    # Execute the command
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
