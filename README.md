# 🍎 AppleLeaf — AI Leaf Health Analyzer

A beginner-friendly web app that uses deep learning to classify apple leaf diseases and give personalized care tips.

## 🌿 What It Does

| Class | Meaning |
|-------|---------|
| **Apple Normal** | Leaf is healthy — keep up good practices |
| **Apple Black Spot** | Fungal infection (*Alternaria mali*) — treat with fungicide |
| **Apple Brown Spot** | Bacterial/fungal lesions — prune and improve drainage |

Users can **upload a photo** or use their **live webcam** (with auto-analyze mode) to get instant results.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~10-20 min depending on hardware)
```bash
python train_model.py
```
This will create a `model/` folder with:
- `apple_leaf_model.keras` — the trained model
- `class_indices.json` — class mapping

### 3. Run the web app
```bash
python app.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

```
entr/
├── leafs/                      # Dataset (Apple train/test)
│   └── Apple/
│       ├── train/
│       │   ├── Apple black_spot/   (469 images)
│       │   ├── Apple Brown_spot/   (1060 images)
│       │   └── Apple Normal/       (824 images)
│       └── test/
├── model/                      # Generated after training
│   ├── apple_leaf_model.keras
│   ├── class_indices.json
│   └── history_*.csv
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── train_model.py              # Training script
├── app.py                      # Flask web server
└── requirements.txt
```

---

## 🧠 Model Architecture

- **Base**: MobileNetV2 pre-trained on ImageNet (lightweight, fast)
- **Training strategy**: Two phases
  1. Head-only training (frozen base, 10 epochs)
  2. Fine-tuning top 30 layers (20 epochs)
- **Input size**: 224×224 RGB
- **Data augmentation**: rotation, flip, zoom, brightness, shift

---

## 💡 Tips for Best Results

- Use a clear, well-lit photo of a single leaf
- Fill the frame with the leaf
- Avoid blurry or very dark photos
- For webcam: hold the leaf inside the green frame

# Apple-leafs-classification
