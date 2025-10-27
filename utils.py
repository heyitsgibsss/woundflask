import os
import uuid
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.applications.efficientnet import preprocess_input
import google.generativeai as genai
import config


# ---------------- helpers ----------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS

def _unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}.{ext}"

def ensure_dirs():
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

def format_gemini_markdown(text: str) -> str:
    """Convert Gemini markdown-style text to HTML"""
    import re
    
    # Replace bold (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace italic (*text*)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Replace headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Replace numbered lists (keep line breaks for CSS to handle)
    text = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap consecutive <li> in <ol>
    text = re.sub(r'(<li>.*?</li>\s*)+', lambda m: f'<ol>{m.group(0)}</ol>', text, flags=re.DOTALL)
    
    # Replace double newlines with paragraph breaks
    text = re.sub(r'\n\n+', '</p><p>', text)
    
    # Wrap in paragraph if not already wrapped
    if not text.startswith('<'):
        text = f'<p>{text}</p>'
    
    return text


# --------------- model & classes ---------------
_model_cache = None
_input_size_cache: Tuple[int, int, int] = (config.IMG_SIZE[0], config.IMG_SIZE[1], 3)

def build_model_architecture(num_classes: int = 10):
    """
    Rebuild the model architecture exactly as it was trained.
    This avoids Keras 3 serialization issues.
    """
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.efficientnet import EfficientNetB1
    
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

def get_model():
    """
    Load the model. If MODEL_PATH ends with .weights.h5, rebuild architecture and load weights.
    Otherwise try to load the full model.
    """
    global _model_cache, _input_size_cache
    if _model_cache is not None:
        return _model_cache

    try:
        if config.MODEL_PATH.endswith('.weights.h5'):
            # Rebuild architecture and load weights
            print(f"Rebuilding model architecture and loading weights from {config.MODEL_PATH}")
            num_classes = len(config.CLASS_NAMES)
            m = build_model_architecture(num_classes=num_classes)
            m.load_weights(config.MODEL_PATH)
            print("✓ Model weights loaded successfully!")
        else:
            # Try to load full model (legacy path)
            m = load_model(config.MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model at '{config.MODEL_PATH}'. "
            f"Likely the saved model has an input/channel mismatch (e.g. 1-channel). "
            f"Please re-export the model with input=(224,224,3). "
            f"Original error: {e}"
        )

    # Derive expected input size/channels from the loaded model
    ishape = m.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    # ishape: (None, H, W, C)
    H, W, C = ishape[1], ishape[2], ishape[3]
    _input_size_cache = (H or config.IMG_SIZE[0], W or config.IMG_SIZE[1], C or 3)

    # If the model itself reports C!=3, we still can run by *feeding* RGB and letting Keras handle it,
    # but for EfficientNet backbones it should be 3. Warn loudly:
    if C != 3:
        print(f"[WARN] Loaded model expects {C} input channels; EfficientNetB1 expects 3. "
              f"If predict fails, re-save the model with RGB input.")

    # ⚡ OPTIMIZATION: Run a dummy prediction to warm up the model
    # This compiles TensorFlow graph and makes subsequent predictions faster
    print("🔥 Warming up model with dummy prediction...")
    dummy_input = np.random.rand(1, H, W, C).astype(np.float32)
    _ = m.predict(dummy_input, verbose=0)
    print("✓ Model warmed up!")

    _model_cache = m
    return _model_cache

def load_class_names() -> List[str]:
    return list(config.CLASS_NAMES)


# --------------- preprocessing & predict ---------------
def load_image_for_model(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (preprocessed_batch, original_resized_rgb)
    - Reads as RGB (3-channel) to satisfy EfficientNet's stem.
    - Resizes to the height/width reported by model.input_shape (or config default).
    """
    _, _, C = _input_size_cache
    H, W, _ = _input_size_cache

    # Always convert to RGB to ensure 3 channels for EfficientNet
    img = Image.open(img_path).convert("RGB")
    original_resized = np.array(img.resize((W, H), Image.BILINEAR))

    # EfficientNet preprocessing (expects RGB in [0..255] float32)
    x = preprocess_input(original_resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    return x, original_resized


def predict_image(model, img_batch: np.ndarray):
    """
    Returns (pred_idx, confidence (0-1), probs np.ndarray)
    """
    preds = model.predict(img_batch)
    if preds.ndim == 2 and preds.shape[1] > 1:
        probs = tf.nn.softmax(preds, axis=1).numpy().squeeze()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf, probs
    elif preds.ndim == 2 and preds.shape[1] == 1:
        p1 = float(tf.sigmoid(preds).numpy().squeeze())
        idx = int(p1 >= 0.5)
        probs = np.array([1.0 - p1, p1])
        return idx, float(probs[idx]), probs
    else:
        probs = tf.nn.softmax(preds.reshape(1, -1), axis=1).numpy().squeeze()
        idx = int(np.argmax(probs)); conf = float(probs[idx])
        return idx, conf, probs


# --------------- grad-cam ---------------
def make_gradcam_heatmap(model, img_batch: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
    last_conv_layer_name = config.LAST_CONV_LAYER
    # Validate the conv layer exists; if not, try to guess one
    try:
        _ = model.get_layer(last_conv_layer_name)
    except ValueError:
        # fallback: scan for last Conv2D
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0, 1))
    conv = conv_out[0]
    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def save_heatmap_and_overlay(original_path: str, heatmap: np.ndarray, alpha: float = 0.35):
    orig = Image.open(original_path).convert("RGB")
    w, h = orig.size
    heat_rgb = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb).resize((w, h), Image.BILINEAR)

    heat_name = _unique_name("heatmap", "png")
    overlay_name = _unique_name("overlay", "png")

    heat_path = os.path.join(config.OUTPUT_FOLDER, heat_name)
    heat_img.save(heat_path, "PNG")

    overlay = Image.blend(orig, heat_img, alpha=alpha)
    overlay_path = os.path.join(config.OUTPUT_FOLDER, overlay_name)
    overlay.save(overlay_path, "PNG")

    return heat_path.replace("\\", "/"), overlay_path.replace("\\", "/")


# --------------- Gemini ---------------
_gemini_model = None

def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        if not config.GEMINI_API_KEY:
            return None
        genai.configure(api_key=config.GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
    return _gemini_model

def gemini_check_is_wound(image_path: str):
    model = _get_gemini()
    if model is None:
        return True, "Gemini API key not configured; skipping wound check."

    prompt = "Answer ONLY with 'WOUND' or 'NOT_WOUND' for this image."
    try:
        resp = model.generate_content([prompt, genai.upload_file(image_path)])
        txt = (resp.text or "").strip().upper()
        # Normalize and check whole-word patterns. Model replies sometimes include
        # variants like 'NOT WOUND', 'NOT_WOUND', or 'NOT-WOUND', and naive
        # substring checks (e.g. 'NOT_WOUND' contains 'WOUND') cause logic errors.
        import re
        # Check negative first: variants of NOT WOUND
        if re.search(r"\bNOT[_\-\s]?WOUND\b", txt):
            print(f"[gemini_check_is_wound] Gemini response (interpreted as NOT_WOUND): {txt}")
            return False, "that is not a wound image"
        # Then check positive 'WOUND' (but won't match 'NOT_WOUND' now)
        if re.search(r"\bWOUND\b", txt):
            print(f"[gemini_check_is_wound] Gemini response (interpreted as WOUND): {txt}")
            return True, "wound image detected"

        # If the model reply is ambiguous or doesn't contain the expected tokens,
        # fall back to permissive behavior (allow processing) but include the
        # raw Gemini text in the message for debugging.
        print(f"[gemini_check_is_wound] Gemini response ambiguous: {txt}")
        return True, f"Gemini unclear ('{txt}'); proceeding as wound."
    except Exception as e:
        return True, f"Gemini check failed ({e}); proceeding as wound."

def gemini_penanganan(pred_label: str, original_path: str, heatmap_path: str, overlay_path: str) -> str:
    model = _get_gemini()
    if model is None:
        return ("Gemini tidak terkonfigurasi (GEMINI_API_KEY kosong). "
                "Lewati analisis LLM pada mode ini.")

    prompt = f"""
Analisis gambar luka yang disediakan beserta heatmap Grad-CAM dan overlay-nya.

Berdasarkan hasil klasifikasi model ('{pred_label}'), karakteristik visual luka pada gambar asli, dan area yang disorot oleh heatmap Grad-CAM:

1. **Jelaskan Tipe Luka:** Berikan penjelasan jelas mengenai tipe luka berdasarkan prediksi model ('{pred_label}') dan analisis visual Anda. Jelaskan ciri-ciri utamanya yang terlihat pada gambar.
2. **Sediakan Saran Penanganan:** Tawarkan saran penanganan yang terstruktur, detail, dan spesifik untuk mengelola tipe luka ini. **Saran-saran ini harus didasarkan pada metode yang terbukti secara klinis, merujuk pada pengetahuan dari jurnal medis.** Sajikan langkah-langkah penanganan dalam daftar bernomor atau berpoin yang jelas.
3. **Pertimbangkan Keparahan:** Nyatakan secara eksplisit bahwa untuk luka yang parah (misalnya, dalam, besar, terinfeksi, tidak kunjung sembuh), evaluasi medis profesional dan penanganan oleh dokter atau spesialis perawatan luka sangat direkomendasikan dan mungkin diperlukan sebagai pengganti perawatan di rumah. Sesuaikan rekomendasi ini berdasarkan tingkat keparahan yang terlihat pada gambar.
4. **Interpretasi Heatmap:** Jelaskan secara singkat area mana yang disorot oleh heatmap Grad-CAM dan bagaimana area tersebut kemungkinan berkontribusi pada klasifikasi model ('{pred_label}'). Jika heatmap tidak informatif (misalnya, berwarna solid), sebutkan keterbatasan ini.
5. **Sertakan referensi dari jurnal atau sumber mana penanganan yang disediakan (Haruslah sumber berkreditasi), semua sumbernya dibuat section sendiri saja di bagian bawah**

Fokuslah pada saran yang praktis dan dapat ditindaklanjuti sambil tetap patuh pada persyaratan berbasis klinis dan menekankan perawatan medis profesional untuk kasus-kasus serius. Pastikan nadanya informatif dan hati-hati.
""".strip()

    try:
        resp = model.generate_content([
            prompt,
            genai.upload_file(original_path),
            genai.upload_file(heatmap_path),
            genai.upload_file(overlay_path),
        ])
        return resp.text or "(Tidak ada teks dari Gemini.)"
    except Exception as e:
        return f"Analisis Gemini gagal: {e}"


# --------------- main pipeline ---------------
def handle_upload_and_process(file_storage) -> dict:
    """
    - Save upload
    - Gemini triage wound/not-wound
    - Predict
    - Grad-CAM + overlay
    - Gemini penanganan
    """
    ensure_dirs()

    filename = secure_filename(file_storage.filename)
    ext = filename.rsplit(".", 1)[1].lower()
    saved_name = _unique_name("input", ext)
    saved_path = os.path.join(config.UPLOAD_FOLDER, saved_name)
    file_storage.save(saved_path)
    saved_path = saved_path.replace("\\", "/")

    # LLM triage BEFORE loading the heavy model (fast-fail for non-wound images)
    is_wound, msg = gemini_check_is_wound(saved_path)
    if not is_wound:
        # Return quickly; no model load required
        return {"status": "not_wound", "message": msg, "input_image": saved_path}

    # Load model (will raise if the saved file is malformed)
    try:
        model = get_model()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

    # Preprocess & predict
    xbatch, original_resized = load_image_for_model(saved_path)
    pred_idx, conf, probs = predict_image(model, xbatch)
    classes = load_class_names()
    pred_label = classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}"
    confidence_pct = round(conf * 100.0, 2)

    # Grad-CAM
    heatmap = make_gradcam_heatmap(model, xbatch, class_idx=pred_idx)
    heatmap_path, overlay_path = save_heatmap_and_overlay(saved_path, heatmap, alpha=0.35)

    # Gemini penanganan
    gemini_text = gemini_penanganan(pred_label, saved_path, heatmap_path, overlay_path)
    gemini_html = format_gemini_markdown(gemini_text)

    return {
        "status": "ok",
        "input_image": saved_path,
        "heatmap_image": heatmap_path,
        "overlay_image": overlay_path,
        "pred_label": pred_label,
        "confidence_pct": confidence_pct,
        "probs": probs.tolist(),
        "classes": classes,
        "gemini_text": gemini_text,
        "gemini_html": gemini_html,
    }
