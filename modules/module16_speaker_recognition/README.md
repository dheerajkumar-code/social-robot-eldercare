# Module 16 — Speaker Recognition with Auto-Enrollment

Lightweight, fully offline speaker recognition for the DIAT Social Robot.

## Architecture

```
Microphone → VAD → [2s Sliding Window]
                          ↓
                   MFCC (78 features)
                          ↓
                    SVM Classifier
                          ↓
                  Confidence Score
                    /           \
               ≥ 0.65          < 0.65
              KNOWN           UNKNOWN
                ↓                ↓
         Publish Name      Auto-Enrollment
         /speaker_id       Flow Triggered
```

## Files

| File | Purpose |
|---|---|
| `feature_extractor.py`       | MFCC extraction (shared by training + inference) |
| `train_speaker_model.py`     | Trains SVM from voice samples |
| `register_voice_auto.py`     | Auto-enrollment flow |
| `speaker_recognition_node.py`| Main ROS2 node + standalone runner |
| `requirements.txt`           | Clean dependencies (no torch, no librosa) |

## Quick Start

### 1. Install dependencies
```bash
sudo apt-get install libportaudio2 libsndfile1 espeak
pip install -r requirements.txt
```

### 2. Enroll your first speaker
```bash
python3 speaker_recognition_node.py --enroll
```
Or manually place WAV files:
```
known_voices/
  YourName/
    sample_1.wav
    sample_2.wav
    sample_3.wav  (minimum 2, recommended 5)
```

### 3. Train the model
```bash
python3 train_speaker_model.py
```

### 4. Run standalone (no ROS2)
```bash
python3 speaker_recognition_node.py --standalone
```

### 5. Run as ROS2 node
```bash
source /opt/ros/humble/setup.bash
python3 speaker_recognition_node.py --ros
```

## Auto-Enrollment Flow

When an unknown speaker is detected:
1. Robot says: *"I don't recognize you. May I know your name please?"*
2. User types their name
3. Robot records 5 × 2s voice samples
4. Model retrains automatically
5. Robot says: *"Nice to meet you, [Name]!"*

## ROS2 Topics

| Topic | Type | Description |
|---|---|---|
| `/speaker_id` | `std_msgs/String` | Speaker name or "unknown" |
| `/speaker_confidence` | `std_msgs/Float32` | Confidence 0.0–1.0 |

## Configuration

Edit constants in `speaker_recognition_node.py`:
```python
CONFIDENCE_THRESHOLD = 0.65   # tune if too many false unknowns
UNKNOWN_COOLDOWN_S   = 30.0   # seconds between enrollment triggers
WINDOW_SEC           = 2.0    # audio window per prediction
```

## Feature Stack

- **MFCC** (13 coefficients) + **delta** + **delta-delta** = 39/frame
- **Mean + Std** over all frames = **78 features** per sample
- **SVM (RBF kernel)** with `probability=True` for confidence scores
- **No GPU** needed — <1ms inference on Pi5
