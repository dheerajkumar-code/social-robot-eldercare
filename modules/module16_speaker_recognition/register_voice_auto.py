#!/usr/bin/env python3
"""
Module 16 — register_voice_auto.py
=====================================
Auto-enrollment module for unknown speakers.

Flow:
  1. Robot detects unknown speaker (low confidence)
  2. Robot says: "I don't recognize you. May I know your name?"
  3. Captures name via keyboard (+ optional TTS response)
  4. Records N voice samples (~2s each)
  5. Saves to known_voices/<name>/sample_X.wav
  6. Triggers model retraining
  7. Confirms: "Nice to meet you, <name>!"

Design:
  - Completely standalone (no ROS2 dependency)
  - Called by speaker_recognition_node when UNKNOWN detected
  - Returns enrolled speaker name on success, None on failure/cancel
"""

import os
import sys
import time
import logging
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import SAMPLE_RATE, is_speech

logger = logging.getLogger("AutoEnroll")

# ── Configuration ──
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR       = os.path.join(BASE_DIR, "known_voices")
N_SAMPLES        = 5        # voice samples to record per new speaker
SAMPLE_DURATION  = 2.0      # seconds per sample
SILENCE_TIMEOUT  = 3.0      # max seconds to wait for speech onset
MIN_SPEECH_RATIO = 0.3      # fraction of frames that must be speech


# ─────────────────────────────────────────────────────────────
# TTS helper (optional pyttsx3)
# ─────────────────────────────────────────────────────────────

def _speak(text: str):
    """Speak text if pyttsx3 available, else print."""
    print(f"\n🤖 Robot: {text}")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 145)
        engine.say(text)
        engine.runAndWait()
        del engine
    except Exception:
        pass  # TTS optional


# ─────────────────────────────────────────────────────────────
# Audio recording
# ─────────────────────────────────────────────────────────────

def record_sample(duration: float = SAMPLE_DURATION,
                  device=None) -> np.ndarray:
    """
    Record one audio sample synchronously.
    Returns float32 numpy array.
    """
    n_samples = int(duration * SAMPLE_RATE)
    audio     = sd.rec(
        n_samples,
        samplerate = SAMPLE_RATE,
        channels   = 1,
        dtype      = "float32",
        device     = device,
    )
    sd.wait()
    return audio.flatten()


def record_sample_with_vad(duration: float = SAMPLE_DURATION,
                            device=None,
                            energy_threshold: float = 0.008) -> np.ndarray:
    """
    Record a sample but only accept it if speech was detected.
    Returns audio array or None if no speech detected.
    """
    audio = record_sample(duration, device)

    # Check speech ratio
    frame_size = int(0.02 * SAMPLE_RATE)   # 20ms frames
    frames     = [audio[i:i+frame_size]
                  for i in range(0, len(audio)-frame_size, frame_size)]
    speech_frames = sum(1 for f in frames if is_speech(f, energy_threshold))
    ratio = speech_frames / max(len(frames), 1)

    if ratio < MIN_SPEECH_RATIO:
        return None   # mostly silence
    return audio


# ─────────────────────────────────────────────────────────────
# Name capture
# ─────────────────────────────────────────────────────────────

def capture_name_keyboard() -> str:
    """
    Capture speaker name via keyboard input.
    Returns sanitised name string.
    """
    try:
        name = input("  Your name: ").strip()
        # Sanitise: letters + numbers + underscore only
        name = "".join(c for c in name if c.isalnum() or c == "_")
        return name.capitalize() if name else ""
    except (EOFError, KeyboardInterrupt):
        return ""


# ─────────────────────────────────────────────────────────────
# Main enrollment function
# ─────────────────────────────────────────────────────────────

def enroll_new_speaker(device=None,
                        n_samples: int    = N_SAMPLES,
                        retrain:   bool   = True) -> str:
    """
    Complete auto-enrollment flow for an unknown speaker.

    Args:
        device   : sounddevice input device index (None=default)
        n_samples: number of voice samples to record
        retrain  : automatically retrain model after enrollment

    Returns:
        Speaker name (str) on success
        Empty string on failure or cancellation
    """
    print("\n" + "─"*50)
    print("  🔔  UNKNOWN SPEAKER DETECTED")
    print("─"*50)

    # Step 1: Ask for name
    _speak("I don't recognize you. May I know your name please?")
    time.sleep(0.5)

    # Step 2: Capture name
    print("  (Type your name and press Enter, or press Enter to cancel)")
    name = capture_name_keyboard()

    if not name:
        _speak("No problem. You can register later.")
        logger.info("Enrollment cancelled — no name provided")
        return ""

    if len(name) < 2:
        _speak("Sorry, I couldn't understand the name.")
        return ""

    logger.info(f"Enrolling new speaker: {name}")

    # Step 3: Create speaker directory
    speaker_dir = os.path.join(VOICES_DIR, name)
    os.makedirs(speaker_dir, exist_ok=True)

    # Find next available sample number
    existing = [f for f in os.listdir(speaker_dir) if f.endswith(".wav")]
    start_idx = len(existing) + 1

    # Step 4: Record samples
    _speak(f"Nice to meet you, {name}! "
           f"Please say a few sentences so I can learn your voice. "
           f"I will record {n_samples} samples.")
    time.sleep(0.8)

    recorded_paths = []

    for i in range(n_samples):
        _speak(f"Sample {i+1} of {n_samples}. Please speak now.")
        time.sleep(0.3)

        print(f"\n  🎙️  Recording sample {i+1}/{n_samples} "
              f"({SAMPLE_DURATION}s)...", end="", flush=True)

        # Try up to 3 times to get a valid speech sample
        audio = None
        for attempt in range(3):
            audio = record_sample_with_vad(SAMPLE_DURATION, device)
            if audio is not None:
                break
            print(f"\n  ⚠️  No speech detected, please try again...", end="", flush=True)

        if audio is None:
            print(f"\n  ⚠️  Skipping sample {i+1} (no speech)")
            continue

        # Save WAV file
        wav_path = os.path.join(speaker_dir, f"sample_{start_idx+i}.wav")
        sf.write(wav_path, audio, SAMPLE_RATE)
        recorded_paths.append(wav_path)
        print(f"  ✅ Saved")

        # Brief pause between samples
        if i < n_samples - 1:
            time.sleep(0.5)

    if len(recorded_paths) < 2:
        _speak(f"I could not record enough samples. Please try again later.")
        logger.warning(f"Only {len(recorded_paths)} samples recorded for {name}")
        # Clean up partial files
        for p in recorded_paths:
            try: os.remove(p)
            except: pass
        return ""

    logger.info(f"Recorded {len(recorded_paths)} samples for {name}")

    # Step 5: Retrain model
    if retrain:
        print(f"\n  🧠 Retraining model with {name} added...")
        success = retrain_model()
        if success:
            _speak(f"I have learned your voice, {name}. "
                   f"Nice to meet you! I will recognize you next time.")
            print(f"\n  ✅ {name} successfully enrolled and model updated!")
        else:
            _speak(f"I have saved your voice samples, {name}. "
                   f"The model will be updated shortly.")
            logger.warning("Retraining failed — samples saved for manual retraining")
    else:
        _speak(f"Nice to meet you, {name}! I have saved your voice samples.")

    print("─"*50)
    return name


# ─────────────────────────────────────────────────────────────
# Retrain trigger
# ─────────────────────────────────────────────────────────────

def retrain_model() -> bool:
    """
    Trigger model retraining after new speaker enrollment.
    Returns True on success.
    """
    try:
        from train_speaker_model import load_dataset, train, save_model
        X, y = load_dataset()
        if len(set(y)) < 1:
            logger.error("No speakers found for retraining")
            return False
        pipeline = train(X, y, evaluate=False)
        save_model(pipeline, sorted(set(y)))
        logger.info(f"Model retrained with speakers: {sorted(set(y))}")
        return True
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== Auto Enrollment Standalone Test ===")
    name = enroll_new_speaker(n_samples=3, retrain=True)
    if name:
        print(f"\n✅ Enrolled: {name}")
    else:
        print("\n❌ Enrollment cancelled or failed")
