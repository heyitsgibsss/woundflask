"""
Quick diagnostic to check if the model file is loadable.
Run this BEFORE starting the Flask app.

Usage:
    venv\Scripts\python.exe check_model.py
"""

import sys
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB1

MODEL_PATH = "model/model_efficientnetB1augmented.weights.h5"
NUM_CLASSES = 10  # Update this to match your actual number of classes

def build_model_architecture(num_classes=10):
    """Rebuild the model architecture exactly as trained."""
    input_layer = layers.Input(shape=(224, 224, 3))
    # Don't load imagenet weights here - we'll load our trained weights after
    base_model = EfficientNetB1(weights=None, include_top=False, input_tensor=input_layer)
    base_model.trainable = False
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(224, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

print("=" * 60)
print("MODEL DIAGNOSTIC CHECK")
print("=" * 60)
print(f"\nAttempting to load: {MODEL_PATH}")
print(f"TensorFlow version: {tf.__version__}\n")

try:
    if MODEL_PATH.endswith('.weights.h5'):
        # For weights-only file, rebuild architecture and load weights
        print("Rebuilding model architecture...")
        model = build_model_architecture(num_classes=NUM_CLASSES)
        print("Loading weights...")
        model.load_weights(MODEL_PATH)
        print("✓ Weights loaded successfully!\n")
    else:
        # For full model file
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✓ Model loaded successfully!\n")
    print("✓ Model loaded successfully!\n")
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Total params: {model.count_params():,}")
    
    # Check if input is RGB (3 channels)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    
    channels = input_shape[-1]
    height = input_shape[1]
    width = input_shape[2]
    
    print(f"\n  Expected input: (batch, {height}, {width}, {channels})")
    
    if channels == 3:
        print("\n✓ Model expects 3 channels (RGB) - CORRECT!")
        print("\nYou can now run the Flask app:")
        print("  venv\\Scripts\\python.exe app.py")
    else:
        print(f"\n✗ Model expects {channels} channel(s) - INCORRECT!")
        print("\n" + "!" * 60)
        print("ERROR: Model was saved with wrong input shape!")
        print("!" * 60)
        print("\nEfficientNetB1 requires 3 channels (RGB).")
        print(f"Your model expects {channels} channel(s).\n")
        print("FIX REQUIRED:")
        print("  1. Open ABC_EffecienNetB1withRMSprop.ipynb")
        print("  2. Find the cell that saves the model:")
        print("     model_efficientnet.save('model_efficientnetB1.keras')")
        print("  3. Before that cell, verify model.input_shape shows (None, 224, 224, 3)")
        print("  4. If not, rebuild the model with:")
        print("     input_layer = layers.Input(shape=(224, 224, 3))")
        print("  5. Re-run training and save")
        print("  6. Copy the new .keras file here")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Failed to load model!\n")
    print(f"Error: {type(e).__name__}: {e}\n")
    print("=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    error_str = str(e)
    
    if "received input with shape" in error_str and ", 1)" in error_str:
        print("\nThe model file is CORRUPTED.")
        print("It was saved with a grayscale (1-channel) input layer.")
        print("EfficientNetB1 requires RGB (3-channel) input.\n")
        print("ROOT CAUSE:")
        print("  The model architecture in the .keras file specifies")
        print("  a 1-channel input, but EfficientNet's stem_conv layer")
        print("  expects 3 channels.\n")
    
    print("FIX REQUIRED:")
    print("  You MUST re-export the model from the notebook.\n")
    print("  1. Open: ABC_EffecienNetB1withRMSprop.ipynb")
    print("  2. Find the model building cell (should contain):")
    print("     input_layer = layers.Input(shape=(224, 224, 3))")
    print("     base_model = EfficientNetB1(..., input_tensor=input_layer)")
    print("\n  3. Verify the input is correct:")
    print("     print(model_efficientnet.input_shape)")
    print("     # Should show: (None, 224, 224, 3)")
    print("\n  4. Re-save the model:")
    print("     model_efficientnet.Colab Keras version: 3.10.0save('model_efficientnetB1.keras')   ")
    print("\n  5. Immediately verify the save:")
    print("     test = tf.keras.models.load_model('model_efficientnetB1.keras')")
    print("     print(f'Saved input: {test.input_shape}')")
    print("     # Must show: (None, 224, 224, 3)")
    print("\n  6. Copy the new file to this directory:")
    print("     model/model_efficientnetB1.keras")
    print("\n" + "=" * 60)
    sys.exit(1)

print("\n" + "=" * 60)
print("READY TO RUN!")
print("=" * 60)
