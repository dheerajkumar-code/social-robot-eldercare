#!/usr/bin/env python3
"""Quick test to check ambient noise levels"""
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
DURATION = 0.5  # 500ms chunks

print("Recording ambient noise for 5 seconds...")
print("Please stay silent and don't speak.\n")

energies = []
for i in range(10):
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    energy = np.sqrt(np.mean(audio**2))
    energies.append(energy)
    print(f"Sample {i+1}: energy = {energy:.4f}")
    time.sleep(0.1)

avg_energy = np.mean(energies)
max_energy = np.max(energies)

print(f"\nAverage ambient energy: {avg_energy:.4f}")
print(f"Max ambient energy: {max_energy:.4f}")
print(f"\nRecommended threshold: {max_energy * 1.5:.4f}")
