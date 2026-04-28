#!/usr/bin/env python3
import os
import joblib
import numpy as np
import librosa

class SpeakerIdentity:
    def __init__(self, model_path="speaker_model.pkl"):
        self.models = None
        self.load_model(model_path)

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.models = joblib.load(path)
                print(f"✅ Loaded speaker models: {list(self.models.keys())}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print("⚠️ No speaker model found. Please train first.")

    def identify(self, audio_data, sample_rate=16000):
        """
        Identify speaker from raw audio data (numpy array).
        """
        if not self.models:
            return "Unknown"

        try:
            # Extract MFCC
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Normalize if needed (librosa expects -1 to 1)
            if np.max(np.abs(audio_data)) > 1.0:
                 audio_data = audio_data / np.max(np.abs(audio_data))

            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
            X = mfcc.T
            
            best_score = -float('inf')
            best_speaker = "Unknown"
            
            # Score against each model
            scores_dict = {}
            for speaker, gmm in self.models.items():
                score = gmm.score(X)  # This returns log-likelihood per sample
                avg_score = np.mean(score)
                scores_dict[speaker] = avg_score
                print(f"DEBUG: Score for {speaker}: {avg_score:.2f}") 
            
            # Find best speaker
            best_speaker = max(scores_dict, key=scores_dict.get)
            best_score = scores_dict[best_speaker]
            
            # Calculate score difference for confidence
            scores_list = sorted(scores_dict.values(), reverse=True)
            if len(scores_list) > 1:
                score_diff = scores_list[0] - scores_list[1]
                print(f"DEBUG: Score difference: {score_diff:.2f}")
                
                # If scores are too close, return Unknown
                if score_diff < 5.0:  # Require at least 5 point difference
                    print(f"DEBUG: Scores too close, returning Unknown")
                    return "Unknown"
            
            # Threshold check - relaxed threshold
            if best_score < -120: 
                print(f"DEBUG: Best score {best_score:.2f} is below threshold -120")
                return "Unknown"
                
            return best_speaker

        except Exception as e:
            print(f"Error in identification: {e}")
            return "Error"
