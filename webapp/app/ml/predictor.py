"""
LeafIQ ML Prediction Engine — Phase 8 enhanced.
Singleton model loading, file validation, top-3 output, uncertainty flag.
"""
import os, time, json, io, logging, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ── Singleton globals ─────────────────────────────────────────────────────
_cnn_model     = None
_cnn_extractor = None
_xgb_model     = None

CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]

# Confidence thresholds for uncertainty flagging
CONF_HIGH   = 0.80
CONF_MEDIUM = 0.60
CONF_LOW    = 0.40   # below this → uncertain warning


# ── Validation ────────────────────────────────────────────────────────────

def validate_image_file(file_path: str) -> tuple[bool, str]:
    """
    Verify the file is a valid, decodable image.
    Returns (ok: bool, error_message: str).
    Checks: file exists, readable, valid image format, not corrupt.
    """
    p = Path(file_path)
    if not p.exists():
        return False, "File not found."
    if p.stat().st_size == 0:
        return False, "File is empty."
    if p.stat().st_size > 16 * 1024 * 1024:
        return False, "File exceeds 16 MB limit."
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()     # detects truncated / corrupt files
        # verify() closes the file; reopen to check size
        with Image.open(file_path) as img:
            w, h = img.size
            if w < 8 or h < 8:
                return False, f"Image too small ({w}×{h}). Minimum 8×8 pixels."
            if w > 10000 or h > 10000:
                return False, f"Image too large ({w}×{h}). Maximum 10000×10000."
            # Confirm it has colour channels
            if img.mode not in ("RGB", "RGBA", "L", "P"):
                return False, f"Unsupported colour mode: {img.mode}."
        return True, ""
    except Exception as e:
        return False, f"Invalid or corrupt image: {e}"


# ── Model loading ─────────────────────────────────────────────────────────

def _load_cnn(ckpt_path: str):
    global _cnn_model, _cnn_extractor
    if _cnn_model is not None:
        return
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {ckpt_path}")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    log.info("Loading CNN checkpoint: %s", ckpt_path)
    _cnn_model = tf.keras.models.load_model(ckpt_path)
    log.info("CNN loaded: input=%s params=%d",
             _cnn_model.input_shape, _cnn_model.count_params())
    emb_tensor = None
    for i, layer in enumerate(_cnn_model.layers):
        if layer.__class__.__name__ == "Activation":
            prev  = _cnn_model.layers[i-1] if i > 0 else None
            prev2 = _cnn_model.layers[i-2] if i > 1 else None
            if (prev  and prev.__class__.__name__  == "BatchNormalization" and
                prev2 and prev2.__class__.__name__ == "Dense" and
                    prev2.get_config().get("units") == 256):
                emb_tensor = layer.output
                break
    if emb_tensor is None:
        for i, lyr in enumerate(_cnn_model.layers):
            if lyr.name == "predictions":
                emb_tensor = _cnn_model.layers[i-1].output
                break
    if emb_tensor is not None:
        _cnn_extractor = tf.keras.Model(
            inputs=_cnn_model.input, outputs=emb_tensor, name="CNN_Extractor"
        )


def _load_xgb(pkl_path: str):
    global _xgb_model
    if _xgb_model is not None:
        return
    if not Path(pkl_path).exists():
        raise FileNotFoundError(f"XGBoost model not found: {pkl_path}")
    import pickle
    with open(pkl_path, "rb") as f:
        _xgb_model = pickle.load(f)
    log.info("XGBoost loaded from %s", pkl_path)


def is_cnn_loaded() -> bool:
    return _cnn_model is not None


def is_xgb_loaded() -> bool:
    return _xgb_model is not None


# ── Preprocessing ─────────────────────────────────────────────────────────

def _preprocess(image_path: str, hw: int) -> np.ndarray:
    """Load → resize to hw×hw → normalise [0,1] → add batch dim."""
    import tensorflow as tf
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, [hw, hw])
    return img.numpy()[np.newaxis, ...]


def preprocess_array(image_path: str, hw: int) -> np.ndarray:
    """Public wrapper used by evaluator.py."""
    return _preprocess(image_path, hw)


# ── Top-N helper ──────────────────────────────────────────────────────────

def _top_n(probs: np.ndarray, n: int = 3) -> list[dict]:
    """Return top-n classes sorted by probability descending."""
    order = np.argsort(probs)[::-1][:n]
    return [
        {
            "class_name": CLASS_NAMES[i],
            "probability": float(probs[i]),
            "rank": int(rank) + 1,
        }
        for rank, i in enumerate(order)
    ]


def _uncertainty_flag(confidence: float, predicted_class: str) -> dict:
    """
    Return a structured uncertainty flag.
    - is_uncertain: True when confidence < CONF_LOW
    - level: "high" | "medium" | "low" | "uncertain"
    - warning: human-readable message when uncertain
    """
    if confidence >= CONF_HIGH:
        level = "high"
        warning = ""
    elif confidence >= CONF_MEDIUM:
        level = "medium"
        warning = (
            "Moderate confidence. Result may be less reliable on borderline images. "
            "Consider submitting a clearer, better-lit image."
        )
    elif confidence >= CONF_LOW:
        level = "low"
        warning = (
            "Low confidence. The model is not certain about this prediction. "
            "The image may be blurry, poorly lit, or not a tomato leaf."
        )
    else:
        level = "uncertain"
        warning = (
            "Very low confidence — result is unreliable. "
            "Please submit a clearer image. "
            "Do not use this result for treatment decisions."
        )
    return {
        "is_uncertain": confidence < CONF_MEDIUM,
        "level":        level,
        "warning":      warning,
        "confidence_pct": round(confidence * 100, 1),
    }


# ── CNN inference ─────────────────────────────────────────────────────────

def predict_cnn(image_path: str, cnn_ckpt: str, input_hw: int) -> dict:
    _load_cnn(cnn_ckpt)
    import tensorflow as tf
    img = _preprocess(image_path, input_hw)
    _cnn_model(img, training=False)       # warmup
    t0    = time.perf_counter()
    probs = _cnn_model(img, training=False).numpy()[0]
    ms    = (time.perf_counter() - t0) * 1000
    pred_i = int(np.argmax(probs))
    conf   = float(probs[pred_i])
    return {
        "predicted_class": CLASS_NAMES[pred_i],
        "confidence":      conf,
        "probabilities":   {CLASS_NAMES[i]: float(probs[i]) for i in range(4)},
        "top3":            _top_n(probs, 3),
        "uncertainty":     _uncertainty_flag(conf, CLASS_NAMES[pred_i]),
        "inference_ms":    round(ms, 2),
        "model_used":      "custom_cnn",
    }


# ── XGBoost hybrid ────────────────────────────────────────────────────────

def predict_hybrid(image_path: str, cnn_ckpt: str, xgb_pkl: str,
                   input_hw: int) -> dict:
    _load_cnn(cnn_ckpt)
    _load_xgb(xgb_pkl)
    import xgboost as xgb
    img   = _preprocess(image_path, input_hw)
    t0    = time.perf_counter()
    emb   = _cnn_extractor(img, training=False).numpy()
    dmat  = xgb.DMatrix(emb)
    probs = _xgb_model.predict(dmat).reshape(-1, 4)[0]
    ms    = (time.perf_counter() - t0) * 1000
    pred_i = int(np.argmax(probs))
    conf   = float(probs[pred_i])
    return {
        "predicted_class": CLASS_NAMES[pred_i],
        "confidence":      conf,
        "probabilities":   {CLASS_NAMES[i]: float(probs[i]) for i in range(4)},
        "top3":            _top_n(probs, 3),
        "uncertainty":     _uncertainty_flag(conf, CLASS_NAMES[pred_i]),
        "inference_ms":    round(ms, 2),
        "model_used":      "cnn_xgboost",
    }


# ── Grad-CAM ──────────────────────────────────────────────────────────────

def generate_gradcam(image_path: str, cnn_ckpt: str, input_hw: int,
                     save_path: str) -> bool:
    try:
        _load_cnn(cnn_ckpt)
        import tensorflow as tf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2

        img_batch = _preprocess(image_path, input_hw)
        last_conv = None
        for layer in reversed(_cnn_model.layers):
            if "conv2d" in layer.name.lower():
                last_conv = layer.name
                break
        if last_conv is None:
            return False

        grad_model = tf.keras.Model(
            inputs=_cnn_model.input,
            outputs=[_cnn_model.get_layer(last_conv).output, _cnn_model.output]
        )
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_batch, tf.float32)
            conv_out, preds = grad_model(inputs)
            pred_class = tf.argmax(preds[0])
            class_prob = preds[:, pred_class]
        grads  = tape.gradient(class_prob, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = conv_out[0] @ pooled[..., tf.newaxis]
        cam    = tf.squeeze(cam).numpy()
        cam    = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_r  = cv2.resize(cam, (input_hw, input_hw))
        orig   = img_batch[0]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle("Grad-CAM Visualization", fontsize=13,
                     fontweight="bold", color="white")
        fig.patch.set_facecolor("#0d1f0d")
        for ax in axes:
            ax.set_facecolor("#0d1f0d")
        axes[0].imshow(orig);          axes[0].set_title("Original", color="white"); axes[0].axis("off")
        axes[1].imshow(cam_r, cmap="jet"); axes[1].set_title("Grad-CAM", color="white"); axes[1].axis("off")
        heatmap_rgb = plt.cm.jet(cam_r)[:, :, :3]
        overlay     = np.clip(0.6 * orig + 0.4 * heatmap_rgb, 0, 1)
        axes[2].imshow(overlay);       axes[2].set_title("Overlay", color="white"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0d1f0d")
        plt.close(fig)
        return True
    except Exception as e:
        log.error("Grad-CAM failed: %s", e, exc_info=True)
        return False


# ── Unified entry point ───────────────────────────────────────────────────

def predict(image_path: str, model_choice: str,
            cnn_ckpt: str, xgb_pkl: str, input_hw: int,
            gradcam_save_path: str | None = None,
            skip_gradcam: bool = False) -> dict:
    """
    Unified prediction entry point. Returns full result dict including
    top3, uncertainty flag, and gradcam_generated boolean.
    """
    # Validate file first
    ok, err = validate_image_file(image_path)
    if not ok:
        raise ValueError(f"Invalid image: {err}")

    if model_choice == "cnn_xgboost":
        result = predict_hybrid(image_path, cnn_ckpt, xgb_pkl, input_hw)
    else:
        result = predict_cnn(image_path, cnn_ckpt, input_hw)

    if gradcam_save_path and not skip_gradcam and model_choice in ("custom_cnn", "cnn_xgboost"):
        ok = generate_gradcam(image_path, cnn_ckpt, input_hw, gradcam_save_path)
        result["gradcam_generated"] = ok
    else:
        result["gradcam_generated"] = False

    return result
