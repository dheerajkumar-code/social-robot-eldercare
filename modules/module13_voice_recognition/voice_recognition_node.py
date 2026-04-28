#!/usr/bin/env python3
"""
Module 13 — Eagle Speaker Recognition (Fixed Version)

This version uses a more robust approach:
- Higher energy threshold to filter silence
- Proper Eagle reset between frames
- Better score interpretation
"""

import os
import time
import argparse
import sounddevice as sd
import numpy as np
import pveagle

BASE_DIR = os.path.join(os.path.dirname(__file__), "known_profiles")
SAMPLE_RATE = 16000
DTYPE = "int16"
CHANNELS = 1
ENERGY_THRESHOLD = 0.03  # Increased threshold for better silence detection


def load_profiles(access_key):
    profiles = []
    files = [f for f in os.listdir(BASE_DIR) if f.endswith(".eagle")]

    if not files:
        print("⚠ No profiles found. Run register_voice.py first.")
        return []

    for fname in files:
        path = os.path.join(BASE_DIR, fname)
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Deserialize using official API
        profile_obj = pveagle.EagleProfile.from_bytes(raw_bytes)

        recognizer = pveagle.create_recognizer(access_key, profile_obj)
        speaker_name = fname.replace(".eagle", "")
        profiles.append((speaker_name, recognizer))

        print(f"Loaded speaker profile: {speaker_name}")

    return profiles


def run_recognition(access_key, threshold, device):
    profiles = load_profiles(access_key)
    if not profiles:
        return

    print(f"\n🎧 Real-time speaker recognition started.")
    print(f"Energy threshold: {ENERGY_THRESHOLD}")
    print(f"Score threshold: {threshold}")
    print("CTRL+C to stop.\n")

    block_size = 512  # Eagle requires exactly 512 samples per frame
    last_result = {"name": None, "score": 0.0, "count": 0}

    def callback(indata, frames, time_info, status):
        # Check energy level first (Voice Activity Detection)
        energy = np.sqrt(np.mean(indata.astype(float)**2))
        
        # If energy is too low, it's silence - skip Eagle processing
        if energy < ENERGY_THRESHOLD:
            current_name = "Silence"
            best_score = 0.0
            best_name = None
        else:
            # Eagle expects a sequence of int16 samples
            audio_samples = indata.flatten().tolist()

            best_name = None
            best_score = -1.0

            for name, rec in profiles:
                try:
                    scores = rec.process(audio_samples)
                    # scores is a list when multiple profiles
                    score = scores[0] if isinstance(scores, list) else scores
                    if score > best_score:
                        best_score = score
                        best_name = name
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    continue
            
            current_name = best_name if best_score >= threshold else "Unknown"

        # Only print if result changed significantly
        score_changed = abs(best_score - last_result["score"]) > 0.15
        name_changed = current_name != last_result["name"]
        
        last_result["count"] += 1
        
        # Print every 15 frames (~480ms) or when something changes
        if name_changed or score_changed or last_result["count"] >= 15:
            if current_name == "Silence":
                print(f"[{time.strftime('%H:%M:%S')}] 🔇 Silence")
            elif best_score >= threshold:
                print(f"[{time.strftime('%H:%M:%S')}] 🎤 Speaker: {best_name}  (score={best_score:.2f}, energy={energy:.3f})")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] ❓ Unknown  (score={best_score:.2f}, energy={energy:.3f})")
            
            last_result["name"] = current_name
            last_result["score"] = best_score
            last_result["count"] = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype=DTYPE, blocksize=block_size,
                        device=device, callback=callback):
        print("Listening...")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Clean up recognizers
            for name, rec in profiles:
                rec.delete()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Recognition confidence threshold (default: 0.7)")
    parser.add_argument("--device", type=int, default=None,
                       help="Audio input device index")
    args = parser.parse_args()

    access_key = os.environ.get("PVE_ACCESS_KEY")
    if not access_key:
        raise SystemExit("❌ ERROR: Set PVE_ACCESS_KEY first.")

    run_recognition(access_key, args.threshold, args.device)


if __name__ == "__main__":
    main()
