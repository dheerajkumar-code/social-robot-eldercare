#!/usr/bin/env python3
"""
Improved rebuild script that properly loads ALL weights from the original model
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras import regularizers
import sys

def build_model(num_classes=7):
    """Rebuild EfficientNetB7 FER architecture"""
    print("🔧 Building model architecture...")
    
    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights=None,  # Don't load ImageNet weights
        input_shape=(224, 224, 3),
        pooling="max"
    )
    
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.016),
            activity_regularizer=regularizers.l1(0.006),
            bias_regularizer=regularizers.l1(0.006)
        ),
        Dropout(rate=0.45, seed=123),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("=" * 70)
    print("Improved Model Rebuild - Loading ALL Weights")
    print("=" * 70)
    
    # Build model
    model = build_model()
    
    # Try to load weights from original model
    print("\n📥 Loading weights from model.h5...")
    try:
        # Load the entire original model first
        print("Loading original model structure...")
        original_model = tf.keras.models.load_model('model.h5', compile=False)
        print("✅ Original model loaded")
        
        # Get weights
        print("Extracting weights...")
        original_weights = original_model.get_weights()
        print(f"✅ Found {len(original_weights)} weight arrays")
        
        # Set weights to new model
        print("Setting weights to new model...")
        model.set_weights(original_weights)
        print("✅ Weights transferred successfully!")
        
    except Exception as e:
        print(f"⚠️ Could not load original model directly: {e}")
        print("Trying alternative method...")
        
        try:
            # Try loading weights by name
            model.load_weights('model.h5', by_name=True, skip_mismatch=False)
            print("✅ Weights loaded by name")
        except Exception as e2:
            print(f"❌ Failed to load weights: {e2}")
            print("Model will have random weights - predictions won't be accurate!")
            return 1
    
    # Compile model
    print("\n🔧 Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Test prediction
    print("\n🧪 Testing model prediction...")
    import numpy as np
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    probs = tf.nn.softmax(predictions[0])
    
    CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    print(f"Prediction probabilities: {probs.numpy()}")
    predicted_class = int(tf.argmax(probs))
    print(f"Predicted: {CLASSES[predicted_class]} ({probs[predicted_class]*100:.1f}%)")
    
    # Check if predictions are reasonable (not all the same)
    prob_std = float(tf.math.reduce_std(probs))
    if prob_std < 0.01:
        print("⚠️ WARNING: All predictions are similar - weights may not be loaded correctly!")
    else:
        print(f"✅ Prediction variance looks good (std={prob_std:.4f})")
    
    # Save model
    print("\n💾 Saving model as fer_rebuilt_v2.h5...")
    model.save('fer_rebuilt_v2.h5')
    print("✅ Model saved!")
    
    print("\n" + "=" * 70)
    print("✅ Done! Use: python3 emotion_subtitle_node.py --model fer_rebuilt_v2.h5 --src 0")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
