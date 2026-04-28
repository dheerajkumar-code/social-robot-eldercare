#!/usr/bin/env python3
"""
Module 13 — Eagle Speaker Enrollment (Official SDK Clean Version)

Usage:
    export PVE_ACCESS_KEY="YOUR_KEY"
    python3 register_voice.py --name Dheeraj
"""

import os
import argparse
import sounddevice as sd
import numpy as np
import pveagle

BASE_DIR = os.path.join(os.path.dirname(__file__), "known_profiles")
os.makedirs(BASE_DIR, exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def record_audio(num_samples):
    """Record exactly num_samples frames from microphone."""
    audio = sd.rec(num_samples, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE)
    sd.wait()
    return audio.tobytes()


def enroll_speaker(access_key, name):
    # Create profiler
    profiler = pveagle.create_profiler(access_key)
    print(f"Created Eagle profiler for: {name}")

    percentage = 0.0
    min_samples = profiler.min_enroll_samples

    while percentage < 100.0:
        print(f"\n🎙 Speak when ready (chunk = {min_samples} samples).")
        input("Press ENTER to record...")

        audio_chunk = record_audio(min_samples)
        percentage, feedback = profiler.enroll(audio_chunk)

        print(f"Progress: {percentage:.1f}% | Feedback: {feedback.name}")

    # Export profile
    profile_obj = profiler.export()
    profiler.delete()

    # Convert to raw bytes using official API
    profile_bytes = profile_obj.to_bytes()

    # Save to file
    out_path = os.path.join(BASE_DIR, f"{name}.eagle")
    with open(out_path, "wb") as f:
        f.write(profile_bytes)

    print(f"\n✅ Enrollment complete. Saved profile → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=False)
    args = parser.parse_args()

    access_key = os.environ.get("PVE_ACCESS_KEY")
    if not access_key:
        raise SystemExit("❌ ERROR: Set PVE_ACCESS_KEY environment variable.")

    name = args.name or input("Enter Speaker Name: ").strip()
    if not name:
        raise SystemExit("Invalid name.")

    enroll_speaker(access_key, name)


if __name__ == "__main__":
    main()
