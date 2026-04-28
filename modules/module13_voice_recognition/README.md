# Module 13 — Eagle Speaker Recognition (Official SDK)

This module implements speaker enrollment and recognition using the **Picovoice Eagle SDK**.

## Features
- ✅ Official Eagle API (no MFCC/SVM hacks)
- ✅ Cross-platform (Linux, macOS, Windows, Raspberry Pi)
- ✅ High-quality voiceprints
- ✅ Real-time speaker identification

## Setup

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Set Access Key
Export your Picovoice access key:
```bash
export PVE_ACCESS_KEY="05+tHbSVWZXlTKVuq5uay9S0SeZj75P9YxHxn86BEAlQsIh8U2lBXw=="
```

## Usage

### Enroll a Speaker
```bash
python3 register_voice.py --name Dheeraj
```

Follow the prompts to record audio samples. The system will guide you through the enrollment process until 100% completion.

### Run Recognition
```bash
python3 voice_recognition_node.py --threshold 0.7
```

The system will:
- Load all enrolled speaker profiles from `known_profiles/`
- Listen to the microphone in real-time
- Identify speakers with scores above the threshold
- Print "Unknown" for unrecognized voices

### Optional Arguments
- `--threshold`: Recognition confidence threshold (default: 0.7)
- `--device`: Audio input device index (default: system default)

## Directory Structure
```
module13_voice_recognition/
├── known_profiles/       # Enrolled speaker profiles (.eagle files)
├── register_voice.py     # Enrollment script
├── voice_recognition_node.py  # Real-time recognition
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Technical Details
- **Sample Rate**: 16 kHz
- **Audio Format**: 16-bit PCM mono
- **Profile Format**: `.eagle` (Picovoice proprietary)
- **Recognition**: Streaming audio chunks (100ms blocks)

## Troubleshooting

**No profiles found**: Run `register_voice.py` to enroll at least one speaker first.

**Access key error**: Make sure `PVE_ACCESS_KEY` environment variable is set.

**Audio device issues**: List available devices with `python3 -m sounddevice` and use `--device` flag.
