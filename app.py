import os
from flask import Flask, render_template, request, redirect, flash

import tensorflow as tf
import config
from utils import allowed_file, handle_upload_and_process, get_model

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

# expose config in templates (footer info)
app.jinja_env.globals.update(config=config)

# Version note
if tf.__version__ != config.REQUIRED_TF:
    print(f"[WARN] TensorFlow {config.REQUIRED_TF} required; found {tf.__version__}.")

# ⚡ PRELOAD MODEL AT STARTUP - This makes first prediction MUCH faster!
print("=" * 60)
print("🔥 Preloading model at startup...")
print("=" * 60)
try:
    model = get_model()
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total params: {model.count_params():,}")
    print("=" * 60)
except Exception as e:
    print(f"❌ Failed to preload model: {e}")
    print("=" * 60)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["image"]
        if not file or file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload PNG/JPG/JPEG/WEBP.")
            return redirect(request.url)

        result = handle_upload_and_process(file)
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
