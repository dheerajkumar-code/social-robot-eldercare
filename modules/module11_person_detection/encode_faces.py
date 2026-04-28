#!/usr/bin/env python3
"""
encode_faces.py — Module 11 Preprocessing
-----------------------------------------
Encodes all known faces into a single encodings.pkl file.

Structure expected:
face_recognition_model/
 ├─ known_faces/
 │   ├─ Dheeraj/
 │   │   ├─ d1.jpg
 │   │   ├─ d2.jpg
 │   │   └─ ...
 │   ├─ Caregiver/
 │   │   ├─ c1.jpg
 │   │   └─ ...
 └─ encodings.pkl
"""

import os
import pickle
import face_recognition
import cv2

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "face_recognition_model", "known_faces")
ENCODINGS_PATH = os.path.join(BASE_DIR, "face_recognition_model", "encodings.pkl")

print(f"🔍 Scanning known faces in: {KNOWN_FACES_DIR}")

known_encodings = []
known_names = []

# Walk through each person's folder
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"🧠 Encoding faces for: {person_name}")
    for filename in os.listdir(person_path):
        path = os.path.join(person_path, filename)

        # Skip non-image files
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"⚠️ Skipping non-image file: {filename}")
            continue

        # Try loading the image safely
        try:
            img = face_recognition.load_image_file(path)
        except Exception as e:
            print(f"❌ Could not load {filename}: {e}")
            continue

        # Try encoding the face
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 0:
            print(f"😕 No face detected in {filename}")
            continue

        encoding = encodings[0]
        known_encodings.append(encoding)
        known_names.append(person_name)
        print(f"✅ Added face from {filename}")

print(f"\n💾 Saving {len(known_encodings)} encodings → {ENCODINGS_PATH}")
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("✅ Encoding complete!")
