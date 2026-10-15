"""
LeafIQ Model Evaluator — Phase 8.
Runs CNN and/or CNN+XGBoost on the cached test set (outputs/cache/).
Results cached to outputs/eval_cache.json so the evaluation page loads
instantly on repeat visits.

Usage (from project root, venv311 active):
    python webapp/app/ml/evaluator.py          # evaluate CNN only
    python webapp/app/ml/evaluator.py --force  # force re-evaluation
    python webapp/app/ml/evaluator.py --both   # CNN + XGBoost

Never modifies model files or training data.
"""
import os, sys, json, time, logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
import warnings; warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
EVAL_CACHE   = PROJECT_ROOT / "outputs" / "eval_cache.json"
MODELS_DIR   = PROJECT_ROOT / "models"

log = logging.getLogger(__name__)

CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]


def _load_test_data():
    imgs  = np.load(CACHE_DIR / "test_images.npy",  mmap_mode="r")
    labs  = np.load(CACHE_DIR / "test_labels.npy")
    return imgs, labs


def _compute_metrics(y_true, y_pred, y_probs=None):
    """Return accuracy, per-class and macro precision/recall/F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
    n = len(y_true)
    acc   = float(accuracy_score(y_true, y_pred))
    prec  = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec   = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1m   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1w   = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm    = confusion_matrix(y_true, y_pred).tolist()
    errors = int((np.array(y_true) != np.array(y_pred)).sum())

    # Per-class
    per_class = []
    for c in range(len(CLASS_NAMES)):
        mask = np.array(y_true) == c
        tp   = int(((np.array(y_pred) == c) & mask).sum())
        fn   = int(mask.sum()) - tp
        fp   = int(((np.array(y_pred) == c) & ~mask).sum())
        tn   = n - tp - fn - fp
        p_c  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_c  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_c  = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        support = int(mask.sum())
        # most confused with
        wrong_preds = [y_pred[i] for i in range(n)
                       if y_true[i] == c and y_pred[i] != c]
        if wrong_preds:
            from collections import Counter
            confused_with = Counter(wrong_preds).most_common(1)[0][0]
            confused_class = CLASS_NAMES[confused_with]
            confused_count = Counter(wrong_preds).most_common(1)[0][1]
        else:
            confused_class = None
            confused_count = 0
        per_class.append({
            "class_idx":      c,
            "class_name":     CLASS_NAMES[c],
            "support":        support,
            "precision":      round(p_c, 4),
            "recall":         round(r_c, 4),
            "f1":             round(f_c, 4),
            "errors":         int(fn + fp) // 2,  # misclassified as this class or from this class
            "error_count":    int((np.array(y_pred)[np.array(y_true)==c] != c).sum()),
            "confused_with":  confused_class,
            "confused_count": confused_count,
        })

    return {
        "accuracy":      round(acc, 6),
        "precision_macro": round(prec, 6),
        "recall_macro":    round(rec, 6),
        "f1_macro":        round(f1m, 6),
        "f1_weighted":     round(f1w, 6),
        "errors":          errors,
        "total":           n,
        "confusion_matrix": cm,
        "per_class":        per_class,
    }


def evaluate_cnn(ckpt_path: str, input_hw: int = 64,
                 batch_size: int = 64) -> dict:
    """Run CNN on test set, return metrics dict."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    if not Path(ckpt_path).exists():
        return {"error": f"Checkpoint not found: {ckpt_path}"}

    log.info("Evaluating CNN on test set...")
    model = tf.keras.models.load_model(ckpt_path)
    imgs, labs = _load_test_data()
    n = len(imgs)

    # Build dataset
    ds = (
        tf.data.Dataset.from_tensor_slices(imgs)
        .map(lambda x: tf.image.resize(x, [input_hw, input_hw]),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size).prefetch(2)
    )

    t0 = time.perf_counter()
    all_probs = []
    for batch in ds:
        all_probs.append(model(batch, training=False).numpy())
    elapsed = time.perf_counter() - t0

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1).tolist()
    labs_list = labs.tolist()

    metrics = _compute_metrics(labs_list, preds, probs)
    metrics["inference_ms_total"]   = round(elapsed * 1000, 1)
    metrics["inference_ms_per_img"] = round(elapsed * 1000 / n, 2)
    metrics["model_key"]            = "custom_cnn"
    metrics["model_params"]         = int(model.count_params())
    metrics["evaluated_at"]         = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    del model
    return metrics


def evaluate_xgboost(cnn_ckpt: str, xgb_pkl: str,
                     input_hw: int = 64,
                     batch_size: int = 64) -> dict:
    """Run CNN+XGBoost on test set, return metrics dict."""
    if not Path(cnn_ckpt).exists():
        return {"error": f"CNN checkpoint not found: {cnn_ckpt}"}
    if not Path(xgb_pkl).exists():
        return {"error": f"XGBoost model not found: {xgb_pkl}"}

    import tensorflow as tf, pickle, xgboost as xgb
    tf.get_logger().setLevel("ERROR")

    log.info("Evaluating CNN+XGBoost on test set...")
    model = tf.keras.models.load_model(cnn_ckpt)

    # Build extractor
    emb_tensor = None
    for i, layer in enumerate(model.layers):
        if layer.__class__.__name__ == "Activation":
            prev  = model.layers[i-1] if i > 0 else None
            prev2 = model.layers[i-2] if i > 1 else None
            if (prev and prev.__class__.__name__ == "BatchNormalization" and
                    prev2 and prev2.__class__.__name__ == "Dense" and
                    prev2.get_config().get("units") == 256):
                emb_tensor = layer.output
                break
    if emb_tensor is None:
        return {"error": "Cannot locate embedding layer in CNN."}

    extractor = tf.keras.Model(inputs=model.input,
                               outputs=emb_tensor, name="Extractor")
    with open(xgb_pkl, "rb") as f:
        xgb_model = pickle.load(f)

    imgs, labs = _load_test_data()
    n = len(imgs)
    ds = (
        tf.data.Dataset.from_tensor_slices(imgs)
        .map(lambda x: tf.image.resize(x, [input_hw, input_hw]),
             num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size).prefetch(2)
    )

    t0 = time.perf_counter()
    feats = []
    for batch in ds:
        feats.append(extractor(batch, training=False).numpy())
    feats_all = np.concatenate(feats, axis=0)
    dtest  = xgb.DMatrix(feats_all)
    probs  = xgb_model.predict(dtest).reshape(-1, 4)
    elapsed = time.perf_counter() - t0

    preds     = probs.argmax(axis=1).tolist()
    labs_list = labs.tolist()

    metrics = _compute_metrics(labs_list, preds, probs)
    metrics["inference_ms_total"]   = round(elapsed * 1000, 1)
    metrics["inference_ms_per_img"] = round(elapsed * 1000 / n, 2)
    metrics["model_key"]            = "cnn_xgboost"
    metrics["model_params"]         = "N/A (tree model)"
    metrics["evaluated_at"]         = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    del model, extractor
    return metrics


def run_evaluation(cnn_ckpt: str, xgb_pkl: str,
                   input_hw: int = 64,
                   include_xgb: bool = False,
                   force: bool = False) -> dict:
    """
    Run evaluation and cache results to eval_cache.json.
    Returns the cache dict.
    """
    cache = {}

    # Load existing cache
    if EVAL_CACHE.exists() and not force:
        try:
            cache = json.loads(EVAL_CACHE.read_text(encoding="utf-8"))
            log.info("Loaded eval cache from %s", EVAL_CACHE)
        except Exception:
            cache = {}

    changed = False

    if force or "custom_cnn" not in cache:
        log.info("Running CNN evaluation...")
        cache["custom_cnn"] = evaluate_cnn(cnn_ckpt, input_hw)
        changed = True

    if include_xgb and (force or "cnn_xgboost" not in cache):
        log.info("Running CNN+XGBoost evaluation...")
        cache["cnn_xgboost"] = evaluate_xgboost(cnn_ckpt, xgb_pkl, input_hw)
        changed = True

    if changed:
        EVAL_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        log.info("Eval cache written to %s", EVAL_CACHE)

    return cache


def load_cached_eval() -> dict:
    """Return cached evaluation results, or empty dict if none."""
    if EVAL_CACHE.exists():
        try:
            return json.loads(EVAL_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Force re-evaluation")
    ap.add_argument("--both",  action="store_true", help="Include CNN+XGBoost")
    args = ap.parse_args()

    cnn_ckpt = str(MODELS_DIR / "custom_cnn_best.keras")
    xgb_pkl  = str(MODELS_DIR / "xgboost_best.pkl")
    cache    = run_evaluation(cnn_ckpt, xgb_pkl,
                              include_xgb=args.both,
                              force=args.force)
    for model_key, m in cache.items():
        if "error" in m:
            print(f"  {model_key}: ERROR — {m['error']}")
        else:
            print(f"  {model_key}: acc={m['accuracy']*100:.2f}% "
                  f"F1={m['f1_macro']*100:.2f}%  errors={m['errors']}/{m['total']}")
