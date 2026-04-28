#!/usr/bin/env python3
import pveagle
import os

key = "05+tHbSVWZXlTKVuq5uay9S0SeZj75P9YxHxn86BEAlQsIh8U2lBXw=="

# Load the profile
with open("known_profiles/Dheeraj.eagle", "rb") as f:
    profile_bytes = f.read()

profile = pveagle.EagleProfile.from_bytes(profile_bytes)
recognizer = pveagle.create_recognizer(key, profile)

print("Frame length:", recognizer.frame_length)
print("Sample rate:", recognizer.sample_rate)
print("Expected bytes:", recognizer.frame_length * 2)  # int16 = 2 bytes per sample

recognizer.delete()
