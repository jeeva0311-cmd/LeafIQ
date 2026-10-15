"""
LeafIQ - Phase 5: Single-Image Hybrid Prediction
==================================================
Loads the trained CNN feature extractor + XGBoost classifier and predicts
the disease class for one or more images.

Usage:
    python src/predict_phase5.py path/to/image.jpg
    python src/predict_phase5.py img1.jpg img2.jpg img3.jpg
    python src/predict_phase5.py path/to/image.jpg --json     # JSON output
    python src/predict_phase5.py path/to/image.jpg --topk 4  # show all probs
"""

import os, sys, json, argparse, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]

CNN_CKPT    = MODELS_DIR / "custom_cnn_best.keras"
XGB_PKL     = MODELS_DIR / "xgboost_best.pkl"
CNN_INPUT_H = 64


def load_models():
    """Load CNN extractor and XGBoost model once, return both."""
    import tensorflow as tf
    import pickle

    tf.get_logger().setLevel("ERROR")

    # Build CNN extractor (same logic as extract_features.py)
    full_model = tf.keras.models.load_model(str(CNN_CKPT))
    embedding_tensor = None
    for i, layer in enumerate(full_model.layers):
        if layer.__class__.__name__ == "Activation":
            prev  = full_model.layers[i - 1] if i > 0 else None
            prev2 = full_model.layers[i - 2] if i > 1 else None
            if (prev and prev.__class__.__name__ == "BatchNormalization" and
                    prev2 and prev2.__class__.__name__ == "Dense" and
                    prev2.get_config().get("units") == 256):
                embedding_tensor = layer.output
                break
    if embedding_tensor is None:
        # fallback: layer before predictions
        for i, layer in enumerate(full_model.layers):
            if layer.name == "predictions":
                embedding_tensor = full_model.layers[i - 1].output
                break
    if embedding_tensor is None:
        raise RuntimeError("Cannot locate embedding layer in CNN.")

    extractor = tf.keras.Model(inputs=full_model.input,
                               outputs=embedding_tensor,
                               name="CNN_Extractor")

    with open(XGB_PKL, "rb") as f:
        xgb_model = __import__("pickle").load(f)

    return extractor, xgb_model


def load_image(path: str) -> np.ndarray:
    """Load a JPEG/PNG image, resize to 64×64, normalise to [0,1]."""
    import tensorflow as tf
    raw  = tf.io.read_file(path)
    img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img  = tf.image.resize(img, [CNN_INPUT_H, CNN_INPUT_H])
    img  = tf.cast(img, tf.float32) / 255.0
    return img.numpy()  # (64, 64, 3)


def predict_single(extractor, xgb_model, image_path: str) -> dict:
    """Return prediction dict for one image."""
    import xgboost as xgb

    img     = load_image(image_path)              # (64, 64, 3)
    batch   = img[np.newaxis, ...]                # (1, 64, 64, 3)
    embed   = extractor(batch, training=False).numpy()  # (1, 256)

    dmat    = xgb.DMatrix(embed)
    probs   = xgb_model.predict(dmat).reshape(-1, len(CLASS_NAMES))[0]
    pred_i  = int(np.argmax(probs))

    return {
        "image":      image_path,
        "prediction": CLASS_NAMES[pred_i],
        "confidence": round(float(probs[pred_i]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 4)
            for i in range(len(CLASS_NAMES))
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="LeafIQ Phase 5 — Hybrid CNN+XGBoost Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict_phase5.py leaf.jpg
  python src/predict_phase5.py img1.jpg img2.jpg --json
  python src/predict_phase5.py leaf.jpg --topk 4
""")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--json",  action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--topk", type=int, default=1,
                        help="Show top-k class probabilities (default: 1)")
    args = parser.parse_args()

    # Validate paths
    missing = [p for p in args.images if not Path(p).exists()]
    if missing:
        print(f"ERROR: file(s) not found: {missing}", file=sys.stderr)
        sys.exit(1)

    print("Loading models ...", file=sys.stderr)
    extractor, xgb_model = load_models()
    print(f"Models loaded. Predicting {len(args.images)} image(s) ...\n",
          file=sys.stderr)

    results = []
    for img_path in args.images:
        try:
            result = predict_single(extractor, xgb_model, img_path)
            results.append(result)
        except Exception as e:
            results.append({"image": img_path, "error": str(e)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"ERROR  {r['image']}: {r['error']}")
                continue
            print(f"Image      : {r['image']}")
            print(f"Prediction : {r['prediction']}")
            print(f"Confidence : {r['confidence']:.2f}%")
            if args.topk > 1:
                print("Probabilities:")
                sorted_probs = sorted(r["probabilities"].items(),
                                      key=lambda x: -x[1])
                for cls, pct in sorted_probs[:args.topk]:
                    print(f"  {cls:<28} {pct:.2f}%")
            print()


if __name__ == "__main__":
    main()
