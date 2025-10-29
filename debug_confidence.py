"""
Debug script: compute and inspect model confidence scores.

Usage examples:
  python debug_confidence.py --images_dir static/uploads --labels labels.csv
  python debug_confidence.py --image sample.jpg

labels.csv format (no header) or TSV: filename,label
  sample1.jpg,Abrasions
  sample2.jpg,Normal

The script will:
 - Load the model using your existing `config.py` and `utils.get_model()` (compatible
   with weights-only .weights.h5 files because utils rebuilds the architecture).
 - For each image: show raw model output, whether outputs look like logits or probs,
   softmaxed probabilities, predicted label, confidence (0..1) and the true label (if available).
 - Compute dataset metrics: accuracy, Brier score (multiclass), reliability curve, ECE.
 - Optional: fit a temperature scaling scalar and show improved calibration metrics.

This single-file tool avoids changing other project files; it imports your `config` and `utils`.
"""

import os
import sys
import argparse
import csv
import math
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

import config
import utils
from flask import Flask, request, jsonify, render_template_string

# Local model cache for multiple models so we don't interfere with utils' cache
_local_model_cache = {}

# Define available model options and expected filenames under the `model/` folder.
# For placeholders, the filename may not exist yet; in that case we fall back to the default model
# to avoid breaking the workflow.
MODEL_OPTIONS = [
    # EfficientNetB1 (default + variants)
    ("efficientnetb1_default", "EfficientNetB1 - default", "model_efficientnetB1.weights.h5"),
    ("efficientnetb1_adamax", "EfficientNetB1 - Adamax (augmented)", "model_efficientnetB1adamaxaugmented.weights.h5"),
    ("efficientnetb1_rmsprop", "EfficientNetB1 - RMSProp (augmented)", "model_efficientnetB1RMSpropaugmented.weights.h5"),
    ("efficientnetb1_sgd", "EfficientNetB1 - SGD (augmented)", "model_efficientnetB1SGDaugmented.weights.h5"),

    # EfficientNetB0 variants
    ("efficientnetb0_adamax", "EfficientNetB0 - Adamax (augmented)", "model_efficientnetB0Adamaxaugmented.weights.h5"),
    ("efficientnetb0_asgd", "EfficientNetB0 - ASGD (augmented)", "model_efficientnetB0ASGDaugmented.weights.h5"),
    ("efficientnetb0_rmsprop", "EfficientNetB0 - RMSProp (augmented)", "model_efficientnetB0RMSpropaugmented.weights.h5"),

    # InceptionV3 variants
    ("inceptionv3_adamax", "InceptionV3 - Adamax (augmented)", "model_InceptionV3_ADAMAXaugmented.weights.h5"),
    ("inceptionv3_rmsprop", "InceptionV3 - RMSProp (augmented)", "model_InceptionV3_RMSPROPaugmented.weights.h5"),
    ("inceptionv3_sgd", "InceptionV3 - SGD (augmented)", "model_InceptionV3_SGDaugmented.weights.h5"),

    # ResNet-50 variants (note filenames include 'ResNET-50')
    ("resnet50_adamax", "ResNet-50 - Radamax (augmented)", "model_ResNET-50Radamaxaugmented.weights.h5"),
    ("resnet50_rmsprop", "ResNet-50 - RMSProp (augmented)", "model_ResNET-50RMSPROPaugmented.weights.h5"),
    ("resnet50_sgd", "ResNet-50 - SGD (augmented)", "model_ResNET-50SGDaugmented.weights.h5"),
]





def build_resnet50_architecture(num_classes: int = 10):
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50

    input_layer = layers.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_layer)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(224, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


def build_efficientnetb0_architecture(num_classes: int = 10):
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0

    input_layer = layers.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=input_layer)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(224, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


def build_inceptionv3_architecture(num_classes: int = 10):
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import InceptionV3

    input_layer = layers.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    base_model = InceptionV3(weights=None, include_top=False, input_tensor=input_layer)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(224, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

def get_model_for_key(key: Optional[str]):
    """Return a loaded Keras model for the selected key.
    If the requested model file doesn't exist or loading fails, fall back to utils.get_model().
    This keeps the rest of the system working without interruption.
    """
    # If no key specified, return default
    if not key:
        return utils.get_model(), None

    if key in _local_model_cache:
        return _local_model_cache[key], None

    # Find mapping
    map_entry = next((e for e in MODEL_OPTIONS if e[0] == key), None)
    if map_entry is None:
        return utils.get_model(), f"unknown model key '{key}', using default"

    filename = map_entry[2]
    candidate = os.path.join(os.path.dirname(__file__), 'model', filename)
    print(f"[get_model_for_key] Attempting to load model: {candidate}")
    
    if not os.path.exists(candidate):
        # For now, if the requested model weights/file is not present,
        # return a sentinel so the caller can respond with a clear message.
        print(f"[get_model_for_key] Model file not found: {candidate}")
        return None, f"Model file not found: {filename}"

    # Try to load weights into a matching architecture or full model
    try:
        kl = key.lower()
        print(f"[get_model_for_key] Loading model type: {kl}")
        
        if 'efficientnetb1' in kl:
            num_classes = len(config.CLASS_NAMES)
            print(f"[get_model_for_key] Building EfficientNetB1 architecture with {num_classes} classes")
            m = utils.build_model_architecture(num_classes=num_classes)
            m.load_weights(candidate)
        elif 'resnet50' in kl:
            num_classes = len(config.CLASS_NAMES)
            print(f"[get_model_for_key] Building ResNet50 architecture with {num_classes} classes")
            m = build_resnet50_architecture(num_classes=num_classes)
            m.load_weights(candidate)
        elif 'efficientnetb0' in kl:
            num_classes = len(config.CLASS_NAMES)
            print(f"[get_model_for_key] Building EfficientNetB0 architecture with {num_classes} classes")
            m = build_efficientnetb0_architecture(num_classes=num_classes)
            m.load_weights(candidate)
        elif 'inceptionv3' in kl:
            num_classes = len(config.CLASS_NAMES)
            print(f"[get_model_for_key] Building InceptionV3 architecture with {num_classes} classes")
            m = build_inceptionv3_architecture(num_classes=num_classes)
            m.load_weights(candidate)
        else:
            # Try loading full model first (in case user saves complete model later)
            print(f"[get_model_for_key] Attempting to load full model")
            try:
                m = tf.keras.models.load_model(candidate, compile=False)
            except Exception as load_err:
                print(f"[get_model_for_key] Failed to load full model: {load_err}")
                return None, f"Failed to load model file {filename}: {load_err}"

        # warm up
        print(f"[get_model_for_key] Warming up model...")
        ishape = m.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        H, W, C = ishape[1], ishape[2], ishape[3]
        _ = m.predict(np.random.rand(1, H, W, C).astype(np.float32), verbose=0)
        _local_model_cache[key] = m
        print(f"[get_model_for_key] Successfully loaded and cached model: {key}")
        return m, None
    except Exception as e:
        print(f"[get_model_for_key] Error loading model {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Failed to load model at '{candidate}'. Error: {e}"


def read_label_csv(path: str) -> List[Tuple[str, str]]:
    """Return list of (filename, label)
    Accepts CSV with two columns (filename,label) without header."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            if len(r) < 2:
                continue
            rows.append((r[0].strip(), r[1].strip()))
    return rows


def is_probability_array(arr: np.ndarray, atol_sum: float = 1e-3) -> bool:
    """Heuristic: check if rows sum to ~1 and all entries in [0,1]."""
    if arr.ndim != 2:
        return False
    row_sums = arr.sum(axis=1)
    if not np.all(np.isfinite(arr)):
        return False
    if np.any(arr < -1e-6) or np.any(arr > 1.0 + 1e-6):
        return False
    if not np.allclose(row_sums, 1.0, atol=atol_sum):
        return False
    return True


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def compute_brier_score(probs: np.ndarray, true_idx: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between one-hot true and predicted probs.
    probs: (N, K), true_idx: (N,)
    """
    N, K = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), true_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 10):
    """Return per-bin average confidence, accuracy and counts. ECE computed with absolute gap weighting by count."""
    # Take predicted confidence for predicted class
    pred_idx = np.argmax(probs, axis=1)
    pred_conf = probs[np.arange(len(probs)), pred_idx]
    correct = (pred_idx == true_idx)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(pred_conf, bins) - 1
    bin_stats = []
    ece = 0.0
    N = len(probs)
    for b in range(n_bins):
        mask = bin_ids == b
        count = np.sum(mask)
        if count == 0:
            avg_conf = 0.0
            acc = 0.0
        else:
            avg_conf = float(np.mean(pred_conf[mask]))
            acc = float(np.mean(correct[mask]))
        bin_stats.append((b, bins[b], bins[b+1], count, avg_conf, acc))
        ece += (count / N) * abs(avg_conf - acc)
    return bin_stats, float(ece)


def fit_temperature(logits: np.ndarray, true_idx: np.ndarray, max_iters: int = 200, lr: float = 0.01) -> float:
    """Fit a scalar temperature T that minimizes NLL on logits/T.
    logits: (N, K) numpy array. Returns learned T (>0).
    Uses TensorFlow optimization for stability.
    """
    # Convert to tensors
    logits_tf = tf.constant(logits, dtype=tf.float32)
    y_tf = tf.constant(true_idx, dtype=tf.int32)

    # Initialize log_T so T = exp(log_T) > 0
    log_T = tf.Variable(0.0, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    prev_loss = None
    for i in range(max_iters):
        with tf.GradientTape() as tape:
            T = tf.exp(log_T)
            scaled = logits_tf / T
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_tf, scaled, from_logits=True))
        grads = tape.gradient(loss, [log_T])
        opt.apply_gradients(zip(grads, [log_T]))
        cur_loss = float(loss.numpy())
        if prev_loss is not None and abs(prev_loss - cur_loss) < 1e-6:
            break
        prev_loss = cur_loss
    T_final = float(tf.exp(log_T).numpy())
    return T_final


def analyze_dataset(images: List[str], labels: Optional[List[str]], args):
    # Build model (uses project's config.MODEL_PATH by default)
    if args.model:
        print(f"Overriding config.MODEL_PATH with: {args.model}")
        config.MODEL_PATH = args.model

    model = utils.get_model()
    class_names = utils.load_class_names()
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    all_probs = []
    all_preds = []
    all_true_idx = []
    raw_outputs = []

    for i, img_path in enumerate(images):
        if not os.path.isabs(img_path):
            # try relative to provided images_dir
            if args.images_dir:
                img_path = os.path.join(args.images_dir, img_path)
        img_path = os.path.abspath(img_path)
        if not os.path.exists(img_path):
            print(f"[WARN] image not found: {img_path}")
            continue

        xb, orig = utils.load_image_for_model(img_path)
        # Raw model output
        preds = model.predict(xb)
        raw_outputs.append(preds.reshape(-1))

        # Decide whether preds are probs or logits
        if preds.ndim == 2 and preds.shape[1] > 1:
            preds2 = preds  # shape (1,K)
            probs_candidate = preds2
            probs_candidate = np.asarray(probs_candidate)
            if is_probability_array(probs_candidate):
                probs = probs_candidate.squeeze()
                used_from = 'probs_as_output'
            else:
                probs = softmax_np(probs_candidate)
                probs = probs.squeeze()
                used_from = 'softmaxed_logits'
        elif preds.ndim == 2 and preds.shape[1] == 1:
            # binary logits
            p1 = float(tf.sigmoid(preds).numpy().squeeze())
            probs = np.array([1.0 - p1, p1])
            used_from = 'sigmoid'
        else:
            # fallback: flatten and softmax
            flat = preds.reshape(1, -1)
            probs = softmax_np(flat).squeeze()
            used_from = 'flatten_softmax'

        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

        true_i = None
        if labels is not None:
            lbl = labels[i]
            if lbl in name_to_idx:
                true_i = name_to_idx[lbl]
            else:
                print(f"[WARN] label '{lbl}' not found in CLASS_NAMES; skipping true label for {img_path}")
        all_probs.append(probs)
        all_preds.append(pred_idx)
        all_true_idx.append(true_i if true_i is not None else -1)

        if args.verbose or i < args.head:
            print("---------------------------------------------------------")
            print(f"Image: {img_path}")
            print(f"Raw model output (shape {preds.shape}): {np.array2string(preds.reshape(-1), precision=5, suppress_small=True)}")
            print(f"Interpreted as: {used_from}")
            print(f"Probs (top5):")
            topk = np.argsort(probs)[::-1][:min(5, len(probs))]
            for k in topk:
                print(f"  {class_names[k]:20s} : {probs[k]:.4f}")
            print(f"Predicted: {class_names[pred_idx]} (idx={pred_idx}) confidence={conf:.4f}")
            if true_i is not None:
                print(f"True label: {labels[i]} (idx={true_i}) => {'CORRECT' if true_i==pred_idx else 'WRONG'}")
            print("")

    if len(all_probs) == 0:
        print("No predictions were made. Check image paths.")
        return

    probs_arr = np.vstack(all_probs)
    true_idx_arr = np.array([x for x in all_true_idx if x >= 0], dtype=np.int32)

    # If some labels were missing, align arrays
    if labels is not None and len(true_idx_arr) != probs_arr.shape[0]:
        # Keep only entries with valid true labels
        mask = np.array([x >= 0 for x in all_true_idx])
        probs_arr = probs_arr[mask]
        true_idx_arr = np.array([x for x in all_true_idx if x >= 0], dtype=np.int32)

    pred_idx_arr = np.argmax(probs_arr, axis=1)
    acc = float(np.mean(pred_idx_arr == true_idx_arr)) if len(true_idx_arr) > 0 else float('nan')
    overall_conf = float(np.mean(np.max(probs_arr, axis=1)))
    print("\n================ Summary ================")
    if len(true_idx_arr) > 0:
        print(f"Samples with labels: {len(true_idx_arr)}")
        print(f"Accuracy: {acc*100:.2f}%")
    else:
        print("No true labels provided; per-sample outputs were shown only.")
    print(f"Mean predicted confidence (top class): {overall_conf*100:.2f}%")

    # Brier score
    if len(true_idx_arr) > 0:
        brier = compute_brier_score(probs_arr, true_idx_arr)
        print(f"Brier score (multiclass): {brier:.6f}")

        # Reliability / ECE
        bin_stats, ece = reliability_curve(probs_arr, true_idx_arr, n_bins=args.bins)
        print(f"ECE ({args.bins} bins): {ece:.6f}")
        print("Bin | range      | count | avg_conf | acc")
        for b, lo, hi, cnt, avgc, accb in bin_stats:
            print(f"{b:3d} | [{lo:.2f},{hi:.2f}) | {cnt:5d} | {avgc:.3f}   | {accb:.3f}")

        # Show average confidence for correct/incorrect
        correct_mask = pred_idx_arr == true_idx_arr
        if np.sum(correct_mask) > 0:
            print(f"Avg conf (correct): {np.mean(np.max(probs_arr[correct_mask], axis=1))*100:.2f}%")
        if np.sum(~correct_mask) > 0:
            print(f"Avg conf (incorrect): {np.mean(np.max(probs_arr[~correct_mask], axis=1))*100:.2f}%")

        # Temperature scaling
        if args.temperature:
            # Need logits to fit temperature. Use raw_outputs we collected.
            logits_all = np.vstack([r for r in raw_outputs])
            # If raw outputs look like probabilities, try to invert via log(probs)
            if is_probability_array(logits_all):
                print("Raw model outputs look like probabilities; converting to logits via log(probs+1e-12)")
                logits_for_fit = np.log(np.clip(logits_all, 1e-12, 1.0))
            else:
                logits_for_fit = logits_all
            T = fit_temperature(logits_for_fit, true_idx_arr, max_iters=args.temp_iters, lr=args.temp_lr)
            print(f"Learned temperature T = {T:.4f}")
            # Apply temperature & compute new probs
            scaled = logits_for_fit / T
            scaled_probs = softmax_np(scaled)
            brier_after = compute_brier_score(scaled_probs, true_idx_arr)
            bin_stats2, ece2 = reliability_curve(scaled_probs, true_idx_arr, n_bins=args.bins)
            print(f"After temp-scaling: Brier={brier_after:.6f}, ECE={ece2:.6f}")

    print("\nDone.")


def collect_images_from_dir(images_dir: str) -> List[str]:
    imgs = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.split('.')[-1].lower() in config.ALLOWED_EXTENSIONS:
                imgs.append(os.path.join(root, f))
    imgs.sort()
    return imgs


def main(argv=None):
    p = argparse.ArgumentParser(description="Debug model confidence and calibration")
    p.add_argument('--images_dir', help='Directory containing images (will recurse).')
    p.add_argument('--labels', help='CSV: filename,label (no header). If filenames are relative they are resolved against images_dir.')
    p.add_argument('--image', help='Single image path to inspect')
    p.add_argument('--model', help='Override model path (sets config.MODEL_PATH)')
    p.add_argument('--head', type=int, default=5, help='Show details for first N images')
    p.add_argument('--verbose', action='store_true', help='Verbose per-image output')
    p.add_argument('--bins', type=int, default=10, help='Number of bins for reliability curve')
    p.add_argument('--temperature', action='store_true', help='Fit temperature scaling and report post-calibration metrics')
    p.add_argument('--temp-iters', type=int, default=200, help='Iterations for temp scaling optimizer')
    p.add_argument('--temp-lr', type=float, default=0.01, help='Learning rate for temp scaling')
    p.add_argument('--serve', action='store_true', help='Start a minimal local web server to upload images and get JSON predictions')
    args = p.parse_args(argv)

    if args.image and (args.images_dir or args.labels):
        print("When --image is used, other dataset args are ignored.")

    images = []
    labels = None
    if args.image:
        images = [args.image]
        labels = None
    else:
        if args.images_dir:
            images = collect_images_from_dir(args.images_dir)
            if len(images) == 0:
                print(f"No images found in {args.images_dir}")
                return
        if args.labels:
            rows = read_label_csv(args.labels)
            # If images_dir provided, filenames may be relative; keep order of CSV
            images = [r[0] for r in rows]
            labels = [r[1] for r in rows]
        if not images:
            print("No images specified. Use --image or --images_dir/--labels")
            return

    analyze_dataset(images, labels, args)


if __name__ == '__main__':
    # If --serve is passed, start a tiny Flask app for simple local interaction
    # We parse known args first to detect --serve without changing CLI behavior
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--serve', action='store_true')
    ns, _ = pre.parse_known_args()
    if ns.serve:
        app = Flask('debug_confidence_server')

        @app.route('/', methods=['GET'])
        def index():
            # Improved UI: AJAX upload, console.log JSON, and render sorted class percentages
            return render_template_string('''
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8" />
              <title>Debug Confidence - Upload</title>
              <style>
                body { font-family: Arial, sans-serif; max-width: 720px; margin: 24px; }
                .card { border: 1px solid #ddd; padding: 12px; border-radius: 6px; }
                .result { margin-top: 16px; }
                .list { list-style: none; padding: 0; }
                .list li { padding: 6px 4px; border-bottom: 1px solid #f0f0f0; }
                .label { font-weight: 600; }
                .pct { float: right; color: #333; }
              </style>
            </head>
            <body>
              <h2>Debug Confidence — Upload Image</h2>
                            <div class="card">
                                <input type="file" id="imageFile" accept="image/*" />
                                <select id="modelSelect">
                                    <option value="efficientnetb1_default">EfficientNetB1 - default</option>
                                    <option value="efficientnetb1_adamax">EfficientNetB1 - Adamax (augmented)</option>
                                    <option value="efficientnetb1_rmsprop">EfficientNetB1 - RMSProp (augmented)</option>
                                    <option value="efficientnetb1_sgd">EfficientNetB1 - SGD (augmented)</option>

                                    <option value="efficientnetb0_adamax">EfficientNetB0 - Adamax (augmented)</option>
                                    <option value="efficientnetb0_asgd">EfficientNetB0 - ASGD (augmented)</option>
                                    <option value="efficientnetb0_rmsprop">EfficientNetB0 - RMSProp (augmented)</option>

                                    <option value="inceptionv3_adamax">InceptionV3 - Adamax (augmented)</option>
                                    <option value="inceptionv3_rmsprop">InceptionV3 - RMSProp (augmented)</option>
                                    <option value="inceptionv3_sgd">InceptionV3 - SGD (augmented)</option>

                                    <option value="resnet50_adamax">ResNet-50 - Radamax (augmented)</option>
                                    <option value="resnet50_rmsprop">ResNet-50 - RMSProp (augmented)</option>
                                    <option value="resnet50_sgd">ResNet-50 - SGD (augmented)</option>
                                </select>
                                <button id="uploadBtn">Upload & Predict</button>
                            </div>

              <div id="status" class="result"></div>

              <div id="out" class="card" style="display:none;">
                <h3>Prediction</h3>
                <div id="predSummary"></div>
                <h4>All classes</h4>
                <ul id="classList" class="list"></ul>
              </div>

              <script>
                const btn = document.getElementById('uploadBtn');
                const fileInput = document.getElementById('imageFile');
                const status = document.getElementById('status');
                const out = document.getElementById('out');
                const predSummary = document.getElementById('predSummary');
                const classList = document.getElementById('classList');

                btn.addEventListener('click', async () => {
                  const f = fileInput.files[0];
                  if (!f) { alert('Choose an image first'); return; }
                  status.textContent = 'Uploading...';
                  const fd = new FormData();
                  const sel = document.getElementById('modelSelect');
                  const modelKey = sel ? sel.value : '';
                  fd.append('model', modelKey);
                  fd.append('image', f);
                  try {
                    const resp = await fetch('/predict', { method: 'POST', body: fd });
                    const js = await resp.json();
                    console.log('Prediction JSON:', js);
                    if (js.status !== 'ok') {
                      status.textContent = 'Error: ' + (js.message || JSON.stringify(js));
                      out.style.display = 'none';
                      return;
                    }
                    status.textContent = 'Done.';
                                        const r = js.result;
                                        // server returned probs array and full classes array
                                        const probs = r.probs; // array of floats
                                        const classNames = r.classes || null;
                                        predSummary.innerHTML = `<strong>${r.pred_label}</strong> — ${ (r.confidence*100).toFixed(2) }%`;
                                        // Build array of {label, prob}
                                        let classes = [];
                                        if (classNames && classNames.length === probs.length) {
                                            for (let i=0;i<probs.length;i++) {
                                                classes.push({label: classNames[i], prob: probs[i]});
                                            }
                                        } else {
                                            // Fallback: use topk + generic names
                                            const topk = r.topk || [];
                                            const topLabels = topk.map(x => x.label);
                                            const usedLabels = new Set(topLabels);
                                            for (let i=0;i<topk.length;i++) {
                                                classes.push({label: topk[i].label, prob: topk[i].prob});
                                            }
                                            for (let i=0;i<probs.length;i++){
                                                const p = probs[i];
                                                if (classes.find(c=>Math.abs(c.prob-p)<1e-12)) continue;
                                                const name = `Class_${i}`;
                                                if (!usedLabels.has(name)) classes.push({label: name, prob: p});
                                            }
                                        }
                    // Sort descending
                    classes.sort((a,b)=>b.prob - a.prob);
                    // Render
                    classList.innerHTML = '';
                    for (const c of classes) {
                      const li = document.createElement('li');
                      li.innerHTML = `<span class="label">${c.label}</span><span class="pct">${(c.prob*100).toFixed(2)}%</span>`;
                      classList.appendChild(li);
                    }
                    out.style.display = 'block';
                  } catch (e) {
                    console.error(e);
                    status.textContent = 'Upload failed: ' + e;
                    out.style.display = 'none';
                  }
                });
              </script>
            </body>
            </html>
            ''')

        def predict_single_image_file(img_path: str, model_key: Optional[str] = None):
            """Return a dict with prediction info for a single image path.
            model_key: optional key from MODEL_OPTIONS to select which model to load.
            """
            print(f"[predict_single_image_file] Predicting with model_key: {model_key}")
            model, note = get_model_for_key(model_key)
            # If the loader reported the model is not available, return an explicit message
            if model is None:
                error_msg = note or 'model not here yet'
                print(f"[predict_single_image_file] Model loading failed: {error_msg}")
                return {
                    'status': 'error',
                    'message': error_msg,
                    'requested_model': model_key,
                }

            class_names = utils.load_class_names()

            # predict
            xb, orig = utils.load_image_for_model(img_path)
            preds = model.predict(xb)

            # Interpret outputs same as analyze_dataset
            if preds.ndim == 2 and preds.shape[1] > 1:
                probs_candidate = np.asarray(preds)
                if is_probability_array(probs_candidate):
                    probs = probs_candidate.squeeze()
                else:
                    probs = softmax_np(probs_candidate).squeeze()
            elif preds.ndim == 2 and preds.shape[1] == 1:
                p1 = float(tf.sigmoid(preds).numpy().squeeze())
                probs = np.array([1.0 - p1, p1])
            else:
                flat = preds.reshape(1, -1)
                probs = softmax_np(flat).squeeze()

            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
            topk = np.argsort(probs)[::-1][:5]
            topk_list = [{"label": class_names[int(k)], "prob": float(probs[int(k)])} for k in topk]
            out = {
                "raw_output": preds.reshape(-1).tolist(),
                "probs": probs.tolist(),
                "pred_idx": pred_idx,
                "pred_label": class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}",
                "confidence": conf,
                "topk": topk_list,
                "classes": class_names,
            }
            if note:
                out['note'] = note
            return out

        @app.route('/predict', methods=['POST'])
        def predict_route():
            if 'image' not in request.files:
                return jsonify({'error': 'no image file provided'}), 400
            f = request.files['image']
            if f.filename == '':
                return jsonify({'error': 'no selected file'}), 400
            # save to upload folder
            utils.ensure_dirs()
            filename = f.filename
            save_path = os.path.join(str(config.UPLOAD_FOLDER), filename)
            f.save(save_path)
            save_path = os.path.abspath(save_path)
            try:
                model_key = request.form.get('model') or request.values.get('model')
                res = predict_single_image_file(save_path, model_key=model_key)
                # Console log on server: concise sorted top classes
                try:
                    topk = res.get('topk', [])
                    top_str = ', '.join([f"{t['label']}:{t['prob']*100:.2f}%" for t in topk])
                except Exception:
                    top_str = ''
                mk = model_key or 'default'
                conf_pct = float(res.get('confidence', 0.0)) * 100.0
                print(f"[predict] {filename} -> {res.get('pred_label')} ({conf_pct:.2f}%) model={mk} {top_str}")
                return jsonify({'status': 'ok', 'model': mk, 'result': res})
            except Exception as e:
                print(f"[predict][error] {filename} -> {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        print('Starting debug server at http://127.0.0.1:5000')
        app.run(host='127.0.0.1', port=5000, debug=False)
    else:
        main()
