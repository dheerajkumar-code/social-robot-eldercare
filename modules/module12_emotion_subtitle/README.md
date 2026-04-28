# Module 12 - Emotion Subtitle System

## Overview
This module implements real-time facial emotion recognition and displays the detected emotion as a subtitle overlay with emojis. It supports two different models:

1.  **HuggingFace ViT (Recommended)**: Uses a Vision Transformer trained on a massive combined dataset (FER2013 + RAF-DB + AffectNet) for high accuracy (~76%).
2.  **FER2013 Model (Legacy)**: Uses EfficientNetB7 trained on FER2013 (~65% accuracy).

## Features
- **7 Emotions**: Angry 😡, Disgusted 🤢, Fearful 😨, Happy 🙂, Neutral 😐, Sad 😢, Surprised 😮
- **Multi-Person Support**: Detects emotions for multiple people, shows people count and group mood.
- **Real-time Performance**: Optimized for CPU inference (5-20 FPS).
- **Subtitle Overlay**: Color-coded subtitles with emojis.

## Installation

```bash
pip install -r requirements.txt
pip install transformers torch pillow
```

## Usage

### Quick Start (Interactive Menu)
```bash
./run_demo.sh
```

### Manual Usage

**1. Auto-detect Best Model (Recommended)**
```bash
python3 run_emotion_detection.py --src 0
```
This will automatically detect whether you have PyTorch or TensorFlow installed and use the best available model.

**2. Force TensorFlow Model**
```bash
python3 run_emotion_detection.py --src 0 --force-tensorflow --debug
```

**3. Force PyTorch Model (requires PyTorch installation)**
```bash
python3 run_emotion_detection.py --src 0 --force-pytorch
```

**4. Demo Mode (Cycle through all emotions - no camera needed)**
```bash
python3 demo_all_emotions.py
```

## Advanced Usage

**Direct TensorFlow Script**
```bash
# First rebuild the model if needed
python3 rebuild_improved.py

# Run detection
python3 emotion_subtitle_enhanced.py --model fer_rebuilt_v2.h5 --src 0
```

**Direct PyTorch Script (requires PyTorch)**
```bash
python3 emotion_subtitle_huggingface.py --src 0
```

## Files
- `emotion_subtitle_huggingface.py`: Main script using HuggingFace ViT model.
- `emotion_subtitle_enhanced.py`: Enhanced script for FER2013 model with calibration.
- `emotion_subtitle_node.py`: Original script (legacy).
- `collect_emotion_samples.py`: Tool to collect your own training images.
- `demo_all_emotions.py`: Demonstration script showing all emotions.
- `rebuild_improved.py`: Utility to fix/rebuild the FER2013 model weights.

## Troubleshooting
- **ImportError: ViTImageProcessor**: Run `pip install --upgrade pillow transformers`.
- **Camera not found**: Check `--src` argument (default is 0).
- **Low accuracy**: Ensure good lighting and face the camera directly. The HuggingFace model is significantly more accurate than the FER2013 model.
