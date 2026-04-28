#!/usr/bin/env python3
import os
import glob
import numpy as np
import librosa
import joblib
from sklearn.mixture import GaussianMixture

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        # We need to transpose to shape (n_samples, n_features) for GMM
        return mfcc.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def train_model():
    dataset_dir = "dataset"
    model_file = "speaker_model.pkl"
    
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory not found. Run collect_data.py first.")
        return

    speakers = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if not speakers:
        print("❌ No speaker data found.")
        return

    models = {}
    
    print(f"Found speakers: {speakers}")
    
    for speaker in speakers:
        print(f"\nTraining model for: {speaker}")
        speaker_dir = os.path.join(dataset_dir, speaker)
        files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        features = []
        for f in files:
            feat = extract_features(f)
            if feat is not None:
                features.append(feat)
        
        if not features:
            print(f"⚠️ No valid audio for {speaker}, skipping.")
            continue
            
        # Stack all features for this speaker
        X = np.vstack(features)
        
        # Train GMM with more components for better discrimination
        # Increased from 4 to 8 components for better speaker separation
        gmm = GaussianMixture(n_components=8, covariance_type='diag', n_init=5, max_iter=200)
        gmm.fit(X)
        
        models[speaker] = gmm
        print(f"✅ Model trained for {speaker}")

    if models:
        joblib.dump(models, model_file)
        print(f"\n🎉 All models saved to {model_file}")
    else:
        print("\n❌ No models were trained.")

if __name__ == "__main__":
    train_model()
