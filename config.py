import os

# ---- Paths ----
# Your trained model path (weights only to avoid Keras 3 serialization issues)
MODEL_PATH = os.getenv("MODEL_PATH", "model/model_efficientnetB1augmented.weights.h5")

# Folders for uploads & outputs
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "static/outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---- Flask ----
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")

# ---- Image / Model ----
# EfficientNetB1 notebook uses 224x224 and 'top_conv' for Grad-CAM
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = os.getenv("LAST_CONV_LAYER", "top_conv")

# Class list from your notebook
CLASS_NAMES = [
    "Abrasions", "Bruises", "Burns", "Cut", "Diabetic Wounds",
    "Laseration", "Normal", "Pressure Wounds", "Surgical Wounds", "Venous Wounds"
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ---- Gemini ----
# Set GEMINI_API_KEY in your environment before running the app
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyA41s8Xu9VXiCpZhIWiRDOvUMWYXu9TqxE")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# Control whether the model should be preloaded at app startup. Default to False
# so that lightweight LLM triage can run before loading the heavy model.
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").lower() in ("1", "true", "yes")

# Enforce TensorFlow version per your spec
REQUIRED_TF = "2.19.0"