#!/usr/bin/env python3
"""
Module 14 — Pose Data Collector (Human Activity)
------------------------------------------------
Captures pose landmarks for 6 activities:
standing, walking, waving, falling, laying, sitting.

Each recording lasts ~20 seconds per activity.
No training happens yet — data saved to pose_data/<activity>/.
"""

import os
import csv
import time
import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(__file__)
POSE_DIR = os.path.join(BASE_DIR, "pose_data")
ACTIVITIES = ["standing", "walking", "waving", "falling", "laying", "sitting"]
DURATION = 20  # seconds
FPS = 15       # capture rate
POSE_LANDMARKS = 33

# ---------------- Helpers ----------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def flatten_landmarks(landmarks):
    flat = []
    for lm in landmarks:
        flat.extend([lm.x, lm.y, lm.z, lm.visibility])
    return flat

def record_activity(activity, duration=DURATION, fps=FPS, src=0):
    print(f"\n🎥 Recording activity: {activity.upper()} ({duration}s)")

    out_dir = os.path.join(POSE_DIR, activity)
    ensure_dir(out_dir)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(out_dir, f"{activity}_{ts}.csv")
    print(f"💾 Saving to: {file_path}")

    fieldnames = ["timestamp", "frame_idx"] + [
        f"lm{i}_{coord}" for i in range(POSE_LANDMARKS) for coord in ("x", "y", "z", "v")
    ]

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    writer = open(file_path, "w", newline="")
    csv_writer = csv.writer(writer)
    csv_writer.writerow(fieldnames)

    start_time = time.time()
    frame_idx = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera read error.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            flat = flatten_landmarks(lm)
            row = [datetime.utcnow().isoformat(), frame_idx] + flat
            csv_writer.writerow(row)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(frame, f"Activity: {activity}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {int(time.time() - start_time)}s/{duration}s",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Pose Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ User interrupted.")
            break

        frame_idx += 1
        time.sleep(1 / fps)

    cap.release()
    writer.close()
    pose.close()
    cv2.destroyAllWindows()
    print(f"✅ Saved: {file_path}")


def main():
    print("=== Human Activity Data Collector ===")
    ensure_dir(POSE_DIR)
    print(f"Data will be saved under: {POSE_DIR}\n")
    print("Available activities:")
    for idx, act in enumerate(ACTIVITIES, 1):
        print(f"  {idx}. {act}")

    choice = input("\n➡️ Choose an activity number (1-6) or 'all' to record all sequentially: ").strip()

    if choice.lower() == "all":
        for act in ACTIVITIES:
            input(f"\nPress ENTER to start recording: {act.upper()}...")
            record_activity(act)
            time.sleep(2)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ACTIVITIES):
                act = ACTIVITIES[idx]
                input(f"\nPress ENTER to start recording: {act.upper()}...")
                record_activity(act)
            else:
                print("❌ Invalid choice.")
        except ValueError:
            print("❌ Invalid input.")


if __name__ == "__main__":
    main()
