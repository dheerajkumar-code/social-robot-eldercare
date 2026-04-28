# Module 14 — Human Activity Recognition

[![Status](https://img.shields.io/badge/Status-Verified-success)](VERIFICATION_SUMMARY.md)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.5%25-brightgreen)](VERIFICATION_SUMMARY.md)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](requirements.txt)

Real-time human activity recognition system using MediaPipe Pose and RandomForest classification.

## 🎯 Features

- **6 Activity Classes**: standing, sitting, walking, waving, falling, laying
- **High Accuracy**: 91.5% overall accuracy
- **Real-time**: 0.26 ms feature extraction, 25+ fps capable
- **Lightweight**: 75 KB model size
- **Robust**: Normalized features handle different body sizes
- **Temporal Awareness**: Velocity features capture motion patterns

## 📁 Project Structure

```
module14_human_activity/
├── activity_node.py              # Real-time inference engine
├── train_activity_model.py       # Simple training script
├── collect_pose_data.py          # Data collection utility
├── test_module14.py              # Comprehensive test suite
├── validate_model.py             # Model validation script
├── demo_activity_recognition.py  # Interactive demo
├── requirements.txt              # Python dependencies
├── models/
│   ├── pose_activity_model.pkl       # Original model
│   └── pose_activity_model_new.pkl   # Improved model (91.5% accuracy)
├── pose_data/                    # Training data
│   ├── standing/*.csv
│   ├── sitting/*.csv
│   ├── walking/*.csv
│   ├── waving/*.csv
│   ├── falling/*.csv
│   └── laying/*.csv
└── tests/
    └── train_activity_node.py   # Advanced training with sliding windows
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 test_module14.py
```

### Running Demo

```bash
# Demo with waving activity
python3 demo_activity_recognition.py --activity waving

# Demo with walking activity
python3 demo_activity_recognition.py --activity walking

# Demo with custom model
python3 demo_activity_recognition.py --activity standing --model models/my_model.pkl
```

### Live Inference (requires webcam)

```bash
# Real-time activity detection
python3 activity_node.py \
    --cam 0 \
    --model models/pose_activity_model_new.pkl \
    --fps 25.0 \
    --window 1.5 \
    --step 0.4
```

Press `q` to quit.

## 📊 Performance

### Accuracy by Activity

| Activity | Accuracy | Samples | Status |
|----------|----------|---------|--------|
| Standing | 100.0% | 20 | ✨ Perfect |
| Waving | 95.0% | 20 | ⭐ Excellent |
| Sitting | 90.0% | 20 | ⭐ Excellent |
| Walking | 90.0% | 20 | ⭐ Excellent |
| Laying | 89.5% | 19 | ⭐ Excellent |
| Falling | 83.3% | 18 | ✓ Good |
| **Overall** | **91.5%** | **117** | **✅ Verified** |

### Performance Metrics

- **Feature Extraction**: 0.26 ms per call
- **Real-time Capable**: Yes (25+ fps)
- **Model Size**: 75 KB
- **Memory Efficient**: Yes

## 🔧 Training Your Own Model

### Collect Data

```bash
# Collect pose data for an activity
python3 collect_pose_data.py --activity standing --duration 10
```

### Train Model

```bash
# Using advanced sliding window approach (recommended)
python3 tests/train_activity_node.py \
    --data pose_data \
    --out models/my_model.pkl \
    --window 1.5 \
    --fps 25.0 \
    --step 0.4
```

### Validate Model

```bash
# Validate on all pose data
python3 validate_model.py
```

## 🧪 Testing

### Run All Tests

```bash
# Comprehensive test suite
python3 test_module14.py
```

**Expected Output**:
```
✅ test_landmarks_to_xy              - PASSED
✅ test_normalize_landmarks_xy       - PASSED
✅ test_angle_calculation            - PASSED
✅ test_joint_angles                 - PASSED
✅ test_extract_features_from_sequence - PASSED
✅ test_pose_data_exists             - PASSED
✅ test_model_file_exists            - PASSED
✅ test_model_loaded                 - PASSED
✅ test_model_prediction             - PASSED
✅ test_feature_extraction_speed     - PASSED

Tests run: 10
Successes: 10
✅ All tests passed!
```

## 🏗️ Architecture

### Feature Engineering

The system extracts **70 features** per window:

1. **Normalized Landmarks** (66 features)
   - 33 MediaPipe pose landmarks
   - Normalized to body size (centered on hip, scaled by torso length)
   - 2D coordinates (x, y)

2. **Joint Angles** (2 features)
   - Left shoulder-elbow-wrist angle
   - Right shoulder-elbow-wrist angle

3. **Velocity Statistics** (2 features)
   - Mean velocity magnitude across keypoints
   - Standard deviation of velocity

### Model

- **Algorithm**: RandomForest Classifier
- **Estimators**: 200 trees
- **Training**: Sliding window approach (1.5s windows, 0.4s step)
- **Features**: 70-dimensional feature vector

### Inference Pipeline

1. **Capture Frame** → MediaPipe Pose extraction
2. **Buffer Frames** → Sliding window (1.5 seconds)
3. **Extract Features** → Normalize + angles + velocity
4. **Predict** → RandomForest classification
5. **Display** → Activity label + confidence

## 📝 Dependencies

```
opencv-python >= 4.5
mediapipe >= 0.10
numpy >= 1.21
scipy >= 1.7
scikit-learn >= 1.3
joblib >= 1.3
```

## 🔍 Troubleshooting

### Issue: Model not found

```bash
# Train a new model
python3 tests/train_activity_node.py --data pose_data --out models/pose_activity_model_new.pkl
```

### Issue: Low accuracy

- Collect more training data (10-20 samples per activity)
- Ensure good lighting conditions
- Ensure full body is visible in frame
- Check pose data quality

### Issue: Slow inference

- Reduce window size: `--window 1.0`
- Increase step size: `--step 0.5`
- Reduce camera resolution

## 📚 Documentation

- [VERIFICATION_SUMMARY.md](VERIFICATION_SUMMARY.md) - Quick verification summary
- [Detailed Walkthrough](../../.gemini/antigravity/brain/2c4031cf-87ca-4b08-a90b-7b40199477a3/walkthrough.md) - Complete verification report

## 🎯 Use Cases

- **Elderly Care**: Monitor daily activities, detect falls
- **Healthcare**: Track patient mobility and activity levels
- **Fitness**: Count exercises, monitor form
- **Security**: Detect unusual activities
- **Smart Home**: Activity-based automation

## 🚀 Future Enhancements

1. **More Activities**: Add more activity classes (eating, sleeping, etc.)
2. **LSTM Model**: Better temporal modeling
3. **3D Features**: Use depth information from landmarks
4. **Multi-person**: Track multiple people simultaneously
5. **ROS 2 Integration**: Robot control based on activities
6. **Edge Deployment**: Optimize for Raspberry Pi, Jetson Nano
7. **Alert System**: Real-time notifications for falls

## 📄 License

Part of the Elderly Robot Head project.

## 👥 Contributing

To add new activities:

1. Collect pose data: `python3 collect_pose_data.py --activity new_activity`
2. Train model: `python3 tests/train_activity_node.py`
3. Validate: `python3 validate_model.py`
4. Test: `python3 demo_activity_recognition.py --activity new_activity`

## ✅ Verification Status

**Last Verified**: 2025-11-25  
**Status**: ✅ PASSED  
**Accuracy**: 91.5%  
**Tests**: 10/10 passed

See [VERIFICATION_SUMMARY.md](VERIFICATION_SUMMARY.md) for details.

---

**Module 14 is production-ready and verified correct!** 🎉
