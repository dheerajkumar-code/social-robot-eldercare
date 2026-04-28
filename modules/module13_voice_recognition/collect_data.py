#!/usr/bin/env python3
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import time

SAMPLE_RATE = 16000
DURATION = 3  # Seconds per sample
CHANNELS = 1

def record_audio(duration, samplerate):
    print(f"🎙 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=CHANNELS)
    sd.wait()
    print("✅ Done.")
    return audio

def main():
    print("=======================================")
    print("   Voice Data Collector")
    print("=======================================")
    
    name = input("Enter your name: ").strip().lower()
    if not name:
        print("Invalid name.")
        return

    save_dir = os.path.join("dataset", name)
    os.makedirs(save_dir, exist_ok=True)
    
    existing_files = len(os.listdir(save_dir))
    print(f"Found {existing_files} existing samples for '{name}'.")
    
    num_samples = 5
    print(f"\nWe will record {num_samples} samples.")
    print("Please speak a different sentence each time (e.g., 'Hello robot', 'What is the time', 'Play music').")
    
    for i in range(num_samples):
        input(f"\nPress ENTER to record sample {i+1}/{num_samples}...")
        audio = record_audio(DURATION, SAMPLE_RATE)
        
        filename = os.path.join(save_dir, f"sample_{existing_files + i + 1}.wav")
        sf.write(filename, audio, SAMPLE_RATE)
        print(f"Saved: {filename}")
        
    print(f"\n🎉 Data collection complete for '{name}'!")
    print(f"Total samples: {len(os.listdir(save_dir))}")

if __name__ == "__main__":
    main()
