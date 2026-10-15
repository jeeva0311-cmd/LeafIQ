"""
LeafIQ - Evaluation Script  (Phase 3)
=======================================
Evaluates saved model checkpoints on the held-out TEST set.
Never uses training or validation data during evaluation.

For each available/valid model:
  - Accuracy, Precision, Recall, Macro F1
  - Per-class classification report (text)
  - Confusion matrix plot (raw counts + normalised)
  - Training/validation accuracy and loss curves
  - Inference time per image (ms)

Combined:
  - outputs/evaluation_results.csv
  - outputs/plots/model_comparison.png

Usage
-----
    python src/evaluate.py                   # evaluate all available checkpoints
    python src/evaluate.py --model custom_cnn
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import sys
import csv
import time
import logging
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "outputs" / "cache"
MODELS_DIR   = PROJECT_ROOT / "models"
PLOTS_DIR    = PROJECT_ROOT / "outputs" / "plots"
LOG_DIR      = PROJECT_ROOT / "outputs" / "logs"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES   = ["Tomato_Healthy", "Tomato_Early_Blight",
                 "Tomato_Late_Blight", "Tomato_Leaf_Mold"]
DISPLAY_NAMES = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]
CLASS_COLORS  = ["#27ae60", "#e67e22", "#c0392b", "#8e44ad"]

# ── Checkpoints: model_key -> (checkpoint_file, min_val_acc_to_be_valid)
# MobileNetV2 Phase A and FT both failed to train (val_acc <= 35%)
# They are included but flagged invalid so we don't fabricate results.
MODEL_REGISTRY = {
    "custom_cnn":     {"path": MODELS_DIR / "custom_cnn_best.keras",     "min_valid_acc": 0.50},
    "mobilenetv2":    {"path": MODELS_DIR / "mobilenetv2_best.keras",    "min_valid_acc": 0.50},
    "mobilenetv2_ft": {"path": MODELS_DIR / "mobilenetv2_ft_best.keras", "min_valid_acc": 0.50},
}

TRAIN_CSV = {
    "custom_cnn":     LOG_DIR / "train_custom_cnn.csv",
    "mobilenetv2":    LOG_DIR / "train_mobilenetv2.csv",
    "mobilenetv2_ft": LOG_DIR / "train_mobilenetv2_ft.csv",
}

MODEL_DISPLAY = {
    "custom_cnn":     "Custom CNN",
    "mobilenetv2":    "MobileNetV2 (Phase A)",
    "mobilenetv2_ft": "MobileNetV2 (Fine-tuned)",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "evaluate.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_test_set():
    X = np.load(CACHE_DIR / "test_images.npy")
    y = np.load(CACHE_DIR / "test_labels.npy")
    log.info("Test set: X=%s  y=%s", X.shape, y.shape)
    return X, y


def get_best_val_acc(model_key: str) -> float:
    """Read best val_accuracy from the CSVLogger file."""
    csv_path = TRAIN_CSV.get(model_key)
    if not csv_path or not csv_path.exists():
        return 0.0
    best = 0.0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = float(row.get("val_accuracy", 0))
            if v > best:
                best = v
    return best


def evaluate_model(model_key: str, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, classification_report,
                                  confusion_matrix)

    info     = MODEL_REGISTRY[model_key]
    ckpt     = info["path"]
    min_acc  = info["min_valid_acc"]

    # 1. Check checkpoint exists
    if not ckpt.exists():
        log.warning("  [SKIP] %s — checkpoint not found: %s", model_key, ckpt)
        return {"model_key": model_key, "status": "MISSING_CHECKPOINT",
                "accuracy": None, "f1_macro": None}

    # 2. Check training succeeded (best val_acc in CSV)
    best_val = get_best_val_acc(model_key)
    if best_val < min_acc:
        log.warning("  [SKIP] %s — best_val_acc=%.4f below threshold %.2f "
                    "(training did not converge)", model_key, best_val, min_acc)
        return {"model_key": model_key, "status": "TRAINING_FAILED",
                "best_val_acc_training": round(best_val, 4),
                "accuracy": None, "f1_macro": None,
                "note": f"Model did not converge (val_acc={best_val:.4f}). "
                        "Results would be meaningless."}

    # 3. Load model
    log.info("\n-- Evaluating: %s --", MODEL_DISPLAY[model_key])
    log.info("  Checkpoint: %s", ckpt)
    model = tf.keras.models.load_model(str(ckpt))

    # 4. Verify model by checking a single forward pass
    h, w = model.input_shape[1], model.input_shape[2]
    log.info("  Model input size: %dx%d", h, w)

    # 5. Build test dataset (resize to model's input)
    ds = (tf.data.Dataset.from_tensor_slices(X_test)
          .map(lambda x: tf.image.resize(x, [h, w]),
               num_parallel_calls=tf.data.AUTOTUNE)
          .batch(32).prefetch(2))

    # 6. Inference time
    dummy = X_test[:10]
    ds_d  = (tf.data.Dataset.from_tensor_slices(dummy)
             .map(lambda x: tf.image.resize(x, [h, w]))
             .batch(1))
    for b in ds_d.take(3): model(b, training=False)   # warmup
    t0 = time.perf_counter()
    for b in ds_d: model(b, training=False)
    ms_per_img = (time.perf_counter() - t0) * 1000 / 10

    # 7. Predict full test set
    y_prob = model.predict(ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    # 8. Metrics
    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec   = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1    = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    log.info("  Test Accuracy : %.4f  (%d%%)", acc, round(acc * 100))
    log.info("  Macro Precision: %.4f", prec)
    log.info("  Macro Recall   : %.4f", rec)
    log.info("  Macro F1       : %.4f", f1)
    log.info("  Inference      : %.2f ms/image", ms_per_img)

    # 9. Classification report
    report = classification_report(y_test, y_pred,
                                   target_names=DISPLAY_NAMES, digits=4)
    log.info("\nClassification Report:\n%s", report)
    rpt_path = OUTPUTS_DIR / f"classification_report_{model_key}.txt"
    rpt_path.write_text(
        f"Model: {MODEL_DISPLAY[model_key]}\n"
        f"Checkpoint: {ckpt}\n\n{report}", encoding="utf-8")
    log.info("  Saved: %s", rpt_path)

    # 10. Confusion matrix plot
    _plot_confusion_matrix(y_test, y_pred, model_key)

    # 11. Training curves
    _plot_training_curves(model_key)

    return {
        "model_key":       model_key,
        "status":          "OK",
        "accuracy":        round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro":    round(rec, 4),
        "f1_macro":        round(f1, 4),
        "inference_ms":    round(ms_per_img, 2),
        "n_params":        model.count_params(),
        "best_val_acc_training": round(best_val, 4),
    }


def _plot_confusion_matrix(y_true, y_pred, model_key: str) -> None:
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title = MODEL_DISPLAY[model_key]
    for ax, norm, sub in zip(axes, [False, True],
                             ["Raw counts", "Normalised (row %)"]):
        if norm:
            rs   = cm.astype(float).sum(axis=1, keepdims=True)
            data = np.where(rs > 0, cm / rs, 0.0)
            fmt  = ".2f"
        else:
            data = cm
            fmt  = "d"
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES,
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{title}\n{sub}")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    p = PLOTS_DIR / f"confusion_matrix_{model_key}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("  Saved: %s", p)


def _plot_training_curves(model_key: str) -> None:
    csv_path = TRAIN_CSV.get(model_key)
    if not csv_path or not csv_path.exists():
        return
    epochs, acc, val_acc, loss, val_loss = [], [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]) + 1)
            acc.append(float(row["accuracy"]))
            val_acc.append(float(row["val_accuracy"]))
            loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    title = MODEL_DISPLAY[model_key]
    ax1.plot(epochs, acc,     "b-o", ms=3, label="Train")
    ax1.plot(epochs, val_acc, "r--o", ms=3, label="Val")
    ax1.set_title(f"{title} — Accuracy"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, loss,     "b-o", ms=3, label="Train")
    ax2.plot(epochs, val_loss, "r--o", ms=3, label="Val")
    ax2.set_title(f"{title} — Loss"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    p = PLOTS_DIR / f"training_curves_{model_key}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("  Saved: %s", p)


def save_comparison_csv(results: list) -> Path:
    fields = ["model_key", "status", "accuracy", "precision_macro",
              "recall_macro", "f1_macro", "inference_ms", "n_params",
              "best_val_acc_training", "note"]
    path = OUTPUTS_DIR / "evaluation_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    log.info("Evaluation CSV: %s", path)
    return path


def plot_comparison(results: list) -> None:
    valid = [r for r in results if r.get("status") == "OK"]
    if not valid:
        log.warning("No valid results to plot comparison.")
        return
    names = [MODEL_DISPLAY[r["model_key"]] for r in valid]
    accs  = [r["accuracy"]  * 100 for r in valid]
    f1s   = [r["f1_macro"]  * 100 for r in valid]
    precs = [r["precision_macro"] * 100 for r in valid]
    recs  = [r["recall_macro"]    * 100 for r in valid]

    x  = np.arange(len(names)); bw = 0.20
    fig, ax = plt.subplots(figsize=(max(9, len(names)*3.5), 5))
    cols = ["#3498db","#2ecc71","#e67e22","#9b59b6"]
    for i, (vals, lbl) in enumerate(zip([accs,f1s,precs,recs],
                                        ["Accuracy","Macro F1",
                                         "Macro Precision","Macro Recall"])):
        bars = ax.bar(x + i*bw, vals, bw, label=lbl,
                      color=cols[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + bw*1.5); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 110)
    ax.set_title("LeafIQ — Model Comparison on Test Set")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)
    plt.tight_layout()
    p = PLOTS_DIR / "model_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Comparison plot: %s", p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   choices=list(MODEL_REGISTRY.keys()) + ["all"],
                   default="all")
    return p.parse_args()


def main():
    args = parse_args()
    log.info("=" * 60)
    log.info("LeafIQ — Phase 3 Evaluation on Test Set")
    log.info("=" * 60)

    X_test, y_test = load_test_set()
    keys = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    results = []
    for key in keys:
        r = evaluate_model(key, X_test, y_test)
        results.append(r)

    save_comparison_csv(results)
    plot_comparison(results)

    # Summary
    log.info("\n-- Summary --")
    log.info("  %-28s  %-20s  %8s  %8s",
             "Model", "Status", "Accuracy", "F1-Macro")
    log.info("  " + "-" * 68)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
        f1  = f"{r['f1_macro']:.4f}"  if r.get("f1_macro")  is not None else "N/A"
        log.info("  %-28s  %-20s  %8s  %8s",
                 MODEL_DISPLAY[r["model_key"]], r.get("status","?"), acc, f1)

    log.info("\nEvaluation complete.")
    log.info("  CSV:   %s", OUTPUTS_DIR / "evaluation_results.csv")
    log.info("  Plots: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
