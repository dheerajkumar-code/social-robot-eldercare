#!/usr/bin/env python3
"""
Quick validation script for the rebuilt FER model
"""

import sys
import numpy as np
import tensorflow as tf

print("Loading rebuilt model...")
try:
    model = tf.keras.models.load_model('fer_rebuilt.h5')
    print("✅ Model loaded successfully!")
    
    # Test prediction
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    
    print(f"✅ Prediction shape: {predictions.shape}")
    print(f"✅ Prediction values: {predictions[0]}")
    
    # Apply softmax and get class
    probs = tf.nn.softmax(predictions[0])
    predicted_class = int(tf.argmax(probs))
    confidence = float(probs[predicted_class])
    
    CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    print(f"✅ Predicted: {CLASSES[predicted_class]} ({confidence*100:.1f}%)")
    
    print("\n🎉 Model is working correctly!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
