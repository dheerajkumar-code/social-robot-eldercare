# diagnostics_activity.py
import joblib, os, glob, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

MODEL_PATH = "models/pose_activity_model.pkl"
POSE_DIR = "pose_data"

def print_counts():
    counts = {}
    for cls in os.listdir(POSE_DIR):
        p = os.path.join(POSE_DIR, cls)
        if os.path.isdir(p):
            counts[cls] = len(glob.glob(os.path.join(p, "*.csv")))
    print("Sample counts per class:", counts)

def show_model_info():
    m = joblib.load(MODEL_PATH)
    if isinstance(m, dict):
        clf = m.get("model")
        classes = m.get("classes")
    else:
        clf = m
        classes = getattr(clf, "classes_", None)
    print("Loaded model type:", type(clf))
    print("Classes in model:", classes)

if __name__ == "__main__":
    print_counts()
    show_model_info()
