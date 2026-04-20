"""
Apple Leaf Disease Classifier — Flask Web App
"""

import os, io, base64, json, time
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "apple_leaf_model.h5")
IDX_PATH   = os.path.join(MODEL_DIR, "class_indices.json")

IMG_SIZE = 224

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)

with open(IDX_PATH) as f:
    class_indices = json.load(f)

# Invert: index → class name
idx_to_class = {v: k for k, v in class_indices.items()}
print(f"Model loaded. Classes: {idx_to_class}")

# ── Care tips per class ───────────────────────────────────────────────────────
TIPS = {
    "Apple black_spot": {
        "label":    "Tache Noire (Black Spot)",
        "severity": "moderate",
        "emoji":    "🟠",
        "summary":  "Your apple leaf shows signs of Black Spot disease, caused by the fungus Alternaria mali. Act quickly to prevent spread.",
        "tips": [
            {
                "icon":  "🗑️",
                "title": "Remove infected leaves immediately",
                "body":  "Pick up and dispose of all affected leaves — never compost them, as spores survive and spread. Use a sealed bag."
            },
            {
                "icon":  "💧",
                "title": "Water at the base, not the leaves",
                "body":  "Wet foliage encourages fungal growth. Use drip irrigation or water early morning so leaves dry quickly."
            },
            {
                "icon":  "🌬️",
                "title": "Improve air circulation",
                "body":  "Prune overcrowded branches to let air flow through the canopy. Good airflow dries the leaves faster and limits fungal spread."
            },
            {
                "icon":  "🧪",
                "title": "Apply a copper-based fungicide",
                "body":  "Spray with a copper-based or neem oil fungicide every 7–14 days during wet weather. Always follow label instructions."
            },
            {
                "icon":  "🧹",
                "title": "Clean tools after use",
                "body":  "Disinfect pruning shears with rubbing alcohol between cuts to avoid spreading spores to healthy parts."
            },
        ],
    },

    "Apple Brown_spot": {
        "label":    "Tache Brune (Brown Spot)",
        "severity": "moderate",
        "emoji":    "🟡",
        "summary":  "Your apple leaf shows Brown Spot lesions, often linked to Marssonina coronaria or nutrient stress. Early treatment works well.",
        "tips": [
            {
                "icon":  "✂️",
                "title": "Prune affected branches",
                "body":  "Cut back to healthy wood — at least 10 cm below any visible brown lesions. Dispose of cuttings, do not compost."
            },
            {
                "icon":  "🌿",
                "title": "Check soil nutrition",
                "body":  "Brown spots can signal potassium or calcium deficiency. A simple soil test kit (garden center) will confirm — fertilize accordingly."
            },
            {
                "icon":  "💊",
                "title": "Use a preventive fungicide",
                "body":  "Apply a mancozeb or captan-based fungicide when new leaves emerge and after rain. Prevention is much easier than cure."
            },
            {
                "icon":  "🍂",
                "title": "Clean up fallen leaves in autumn",
                "body":  "The fungus overwinters in fallen leaves. Rake and destroy them every autumn to break the infection cycle for next year."
            },
            {
                "icon":  "🌞",
                "title": "Ensure adequate sunlight",
                "body":  "Apple trees need at least 6 hours of direct sunlight. Shade stress weakens the plant and makes it more susceptible to disease."
            },
        ],
    },

    "Apple Normal": {
        "label":    "Feuille Saine (Healthy Leaf)",
        "severity": "healthy",
        "emoji":    "🟢",
        "summary":  "Great news! Your apple leaf looks perfectly healthy. Keep up with these good practices to maintain this condition.",
        "tips": [
            {
                "icon":  "💧",
                "title": "Water deeply but infrequently",
                "body":  "Water 2–3 times a week, letting the soil dry slightly between sessions. Deep watering encourages strong root growth."
            },
            {
                "icon":  "🌱",
                "title": "Fertilize in spring",
                "body":  "Apply a balanced slow-release fertilizer (10-10-10) in early spring as buds begin to swell. Avoid over-fertilizing — it attracts pests."
            },
            {
                "icon":  "✂️",
                "title": "Prune annually in late winter",
                "body":  "Remove dead, crossing, or inward-growing branches while the tree is dormant. This keeps the canopy open and productive."
            },
            {
                "icon":  "🔍",
                "title": "Inspect leaves weekly",
                "body":  "Check both sides of leaves regularly for spots, discoloration, or insects. Early detection makes treatment easy."
            },
            {
                "icon":  "🪲",
                "title": "Monitor for pests",
                "body":  "Look out for aphids, codling moths, and mites. Sticky traps are a great non-toxic early warning system."
            },
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(img: Image.Image) -> dict:
    start = time.time()
    tensor = preprocess_image(img)
    preds  = model.predict(tensor, verbose=0)[0]
    elapsed = round((time.time() - start) * 1000)

    pred_idx   = int(np.argmax(preds))
    class_name = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx])

    all_scores = {
        idx_to_class[i]: round(float(preds[i]) * 100, 1)
        for i in range(len(preds))
    }

    tips_data = TIPS.get(class_name, TIPS["Apple Normal"])

    return {
        "class_name": class_name,
        "label":      tips_data["label"],
        "severity":   tips_data["severity"],
        "emoji":      tips_data["emoji"],
        "confidence": round(confidence * 100, 1),
        "all_scores": all_scores,
        "summary":    tips_data["summary"],
        "tips":       tips_data["tips"],
        "elapsed_ms": elapsed,
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data and "image" in data:
            # Base64 image from webcam
            header, encoded = data["image"].split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))
        elif "file" in request.files:
            # File upload
            file = request.files["file"]
            img = Image.open(file.stream)
        else:
            return jsonify({"error": "No image provided"}), 400

        result = predict_image(img)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
