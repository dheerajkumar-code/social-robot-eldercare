#!/usr/bin/env python3
"""
Rebuild and Save FER EfficientNetB7 Model as .h5
------------------------------------------------
- Rebuilds the same architecture as in your Kaggle notebook
- Loads pretrained weights (model.h5 or checkpoint)
- Saves the fully serialized model as fer_fixed_model.h5
"""

import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras import regularizers


def build_model(num_classes=7):
    """Rebuild EfficientNetB7 FER architecture (same as trained model)."""
    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
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

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Rebuild FER model and save as .h5")
    parser.add_argument("--weights", type=str, default="model.h5", help="Path to pretrained weights (.h5 file)")
    parser.add_argument("--output", type=str, default="fer_fixed_model.h5", help="Output filename (.h5)")
    args = parser.parse_args()

    print("🔧 Rebuilding FER EfficientNetB7 architecture...")
    model = build_model(num_classes=7)

    if not os.path.exists(args.weights):
        print(f"❌ Weights file not found: {args.weights}")
        return

    print(f"🔁 Loading pretrained weights from {args.weights} (by_name=True, skip_mismatch=True)...")
    try:
        model.load_weights(args.weights, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning while loading weights: {e}")

    print(f"💾 Saving fully serialized model as {args.output}...")
    model.save(args.output)
    print(f"✅ Done! Model saved to: {args.output}")
    print("You can now load it with:")
    print(f"  model = tf.keras.models.load_model('{args.output}')")


if __name__ == "__main__":
    main()
