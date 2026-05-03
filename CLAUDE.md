# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular ROS2-based social robotic head for elderly healthcare assistance, targeting Raspberry Pi 5. The system combines audio, vision, speech, navigation, and caregiver-facing components across 16 modules. Modules run as standalone Python scripts or as ROS2 nodes.

## Environment Setup

ROS2 packages (`rclpy`, `std_msgs`, etc.) come from the system ROS2 install — **never pip-install them**:
```bash
source /opt/ros/humble/setup.bash
```

All Python dependencies (M1–M14) share a single venv:
```bash
pip install -r requirements.txt
```

Key constraints from `requirements.txt`:
- `numpy` is pinned to `==1.26.4` (required by M12)
- Do **not** add `opencv-python` — `mediapipe` installs `opencv-contrib-python`; both conflict
- `torch`/`speechbrain`/`faiss-cpu` are excluded (too heavy for Pi5); M13 uses MFCC+SVM instead

## Build (ROS2 modules)

Module 15 (DWA Navigation) and Module 10 (Integration) are colcon packages:
```bash
cd modules/module15_dwa_navigation   # or module10_integration
colcon build --symlink-install
source install/setup.bash
```

## Running Nodes

Each module can run standalone (no ROS2 required) or as a ROS2 node.

**Standalone:**
```bash
python3 modules/module1_doa/doa_node_v2.py --device 2 --channels 6
python3 modules/module3_dialog_manager/dialog_node_v2.py --interactive
```

**ROS2 node:**
```bash
python3 modules/module1_doa/doa_node_v2.py --ros --device 2 --channels 6
```

**Module 15 (DWA Navigation with Gazebo):**
```bash
source /opt/ros/humble/setup.bash
source modules/module15_dwa_navigation/install/setup.bash
ros2 launch social_nav social_nav.launch.py
```

**Flask backend (M7):**
```bash
cd modules/module7_backend && python3 app.py
```

## Running Tests

Tests use pytest. Run a single module's tests:
```bash
python3 -m pytest modules/module1_doa/tests/test_doa.py -q
python3 -m pytest modules/module12_emotion_subtitle/tests/ -q
python3 -m pytest modules/module14_human_activity/test_module14.py -q
```

Interactive demos (no hardware needed):
```bash
python3 modules/module3_dialog_manager/demo_dialog.py
python3 modules/module12_emotion_subtitle/run_emotion_detection.py --src 0
python3 modules/module14_human_activity/demo_activity_recognition.py
```

## Architecture

### Signal Flow (core pipeline)

```
Microphone → M1 (DoA) ─────────────────────────────► /doa_angle, /doa_active
           → M2 (ASR/Whisper) ──────────────────────► /asr_text
                                                          │
M11 (Face Recognition) ──────────────────────────────► /person_name
M12 (Emotion Detection) ─────────────────────────────► /emotion_label
M13/M16 (Speaker Recognition) ───────────────────────► /speaker_id
M14 (Activity Recognition) ──────────────────────────► /activity_label
                                                          │
                                            M3 (Dialog Manager) ──► /tts_speak, /tts_speak_urgent
                                                                         │
                                                            M4 (TTS/pyttsx3) → Bluetooth speaker
                                                            M5 (Head Controller) ← /doa_angle
```

**M6** (Reminder System) runs independently: SQLite-backed scheduler publishes `/reminder_alert` → M4, logs acknowledgements via `/reminder_ack`.

**M7** (Flask REST API) + **M8** (React dashboard) form the caregiver-facing layer: the backend bridges ROS2 state (reminders, alerts, logs) to the browser dashboard.

**M9** generates daily summaries from M7's SQLite database.

### Module 15 — Social DWA Navigation (Gazebo simulation only)

Three-node pipeline inside the colcon package `social_nav`:
1. `human_tracker_node.py` — subscribes `/gazebo/model_states`, runs Kalman filter per actor, publishes `/human_positions` (JSON) and `/human_markers`
2. `social_costmap_node.py` — reads `/human_positions`, writes asymmetric Gaussian cost fields into a 2D grid, publishes `/social_costmap_raw`
3. `social_override_node.py` — sits between DWA output and robot: `/cmd_vel_raw` → proxemics-zone speed scaling + lookahead cost check + lateral bias → `/cmd_vel`

Nav2 is configured for TurtleBot3 Burger with DWB local planner at social speeds (see `config/nav2_params.yaml`).

### Dual-mode design pattern

Every v2 ROS2 node follows the same pattern: the node class works with or without ROS2. Pass `--ros` flag to activate the ROS2 executor; omit it for interactive/testing use. This means hardware-free development is always possible.

### Dialog Manager (M3) inputs

`dialog_node_v2.py` fuses all perception into a single response. It subscribes to `/asr_text`, `/emotion_label`, `/activity_label`, `/speaker_id`, and `/doa_active`. Responses go to `/tts_speak` (normal) or `/tts_speak_urgent` (emergency, interrupts). Intent/action tags are published to `/dialog_intent` and `/dialog_action`.

### Emotion detection (M12) — model selection

The module auto-detects the best available backend:
- **HuggingFace ViT** (preferred, ~76% accuracy) — requires `transformers` + `torch`
- **TensorFlow FER model** (legacy, ~65%) — uses `fer_rebuilt_v2.h5`

On Pi5, use the TensorFlow path or `tflite-runtime` to avoid the ~4 GB torch overhead.

### Speaker recognition — two modules

- **M13** (`voice_recognition_node.py`): Picovoice Eagle SDK, requires `PVE_ACCESS_KEY` env var
- **M16** (`speaker_recognition_node.py`): Fully offline MFCC+SVM, no cloud dependency — preferred for Pi5

Both publish to `/speaker_id` and `/speaker_confidence`.
