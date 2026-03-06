import tensorflow as tf
from PIL import Image
import numpy as np
import json, io, os

# Load once at import time — not on every request
_model = None
_idx_to_class = None


def _load():
    global _model, _idx_to_class
    if _model is None:
        model_path = os.path.join(os.path.dirname(__file__), "model_final.h5")
        _model = tf.keras.models.load_model(model_path)

        json_path = os.path.join(os.path.dirname(__file__), "class_names.json")
        class_idx = json.load(open(json_path))
        _idx_to_class = {v: k for k, v in class_idx.items()}
        print("✅ Model loaded")


GUIDANCE = {
    "cardboard": {
        "icon": "📦",
        "bin": "Blue Recycling Bin",
        "instructions": [
            "Flatten all boxes before disposal",
            "Remove tape, staples, and plastic windows",
            "Keep dry — wet cardboard is not recyclable",
            "Pizza boxes with grease go in compost, not recycling",
        ],
        "facility_query": "recycling centre",
    },
    "glass": {
        "icon": "🍾",
        "bin": "Green Glass Bin",
        "instructions": [
            "Rinse clean before disposal",
            "Remove metal lids — recycle separately",
            "Do NOT break glass — deposit whole",
            "Window glass and mirrors need special facilities",
        ],
        "facility_query": "glass recycling",
    },
    "metal": {
        "icon": "🥫",
        "bin": "Yellow Recycling Bin",
        "instructions": [
            "Rinse food residue from cans",
            "Crush cans to save space",
            "Aluminium foil: ball up to golf-ball size first",
            "Aerosol cans: must be completely empty",
        ],
        "facility_query": "scrap metal recycling",
    },
    "paper": {
        "icon": "📄",
        "bin": "Blue Recycling Bin",
        "instructions": [
            "Keep dry and clean",
            "Shred sensitive documents before recycling",
            "Tissue and wax-coated paper go in general waste",
            "Remove plastic film from envelopes",
        ],
        "facility_query": "paper recycling",
    },
    "plastic": {
        "icon": "🧴",
        "bin": "Yellow Recycling Bin",
        "instructions": [
            "Check resin code (1–7) on bottom — codes 1 & 2 most recyclable",
            "Rinse containers clean",
            "Do not bag loose plastics inside plastic bags",
            "Black plastic trays often cannot be recycled",
        ],
        "facility_query": "plastic recycling",
    },
    "trash": {
        "icon": "🗑️",
        "bin": "Black General Waste Bin",
        "instructions": [
            "This item cannot be recycled through standard streams",
            "Check if local authority has special collection days",
            "Consider if item can be repaired or donated first",
            "Hazardous items like batteries need special disposal",
        ],
        "facility_query": "waste disposal",
    },
}


def predict_image(image_bytes: bytes) -> dict:
    _load()  # loads model only on first call

    # Preprocess image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((380, 380))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 380, 380, 3)

    # Predict — model outputs 6 softmax probabilities
    preds = _model.predict(arr, verbose=0)[0]  # shape: (6,)
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    # Low confidence — tell user to try a clearer photo
    if confidence < 60:
        return {
            "class": "uncertain",
            "confidence": round(confidence, 1),
            "icon": "❓",
            "bin": "Unable to determine",
            "instructions": [
                "Please try a clearer, well-lit photo of a single item"
            ],
            "facility_query": "recycling",
            "inference_ms": 0,
            "all_scores": {
                _idx_to_class[i]: round(float(p) * 100, 1)
                for i, p in enumerate(preds)
            },
        }

    label = _idx_to_class[top_idx]
    guidance = GUIDANCE[label]

    return {
        "class": label,
        "confidence": round(confidence, 1),
        "icon": guidance["icon"],
        "bin": guidance["bin"],
        "instructions": guidance["instructions"],
        "facility_query": guidance["facility_query"],
        "all_scores": {
            _idx_to_class[i]: round(float(p) * 100, 1)
            for i, p in enumerate(preds)
        },
    }
