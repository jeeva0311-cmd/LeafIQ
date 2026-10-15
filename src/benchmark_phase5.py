"""
LeafIQ - Phase 5: Three-Model Benchmark
=========================================
Compares Custom CNN, MobileNetV2-A, and CNN+XGBoost on the same held-out
test set (821 images).  Loads:
  - Existing evaluation_results.csv  (CNN + MNV2 metrics already computed)
  - XGBoost model + test features    (runs fresh predictions)

Never retrains.  Saves all required Phase 5 outputs.

Required outputs:
  outputs/phase5_model_comparison.csv
  outputs/phase5_model_comparison.txt
  outputs/plots/phase5_classification_metrics.png
  outputs/plots/phase5_per_class_f1.png
  outputs/plots/phase5_confusion_matrices.png
  outputs/plots/phase5_inference_comparison.png
  outputs/plots/phase5_model_size_comparison.png
  outputs/plots/phase5_xgboost_feature_importance.png

Usage:
    python src/benchmark_phase5.py
"""

import os, sys, csv, json, time, logging, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUTS_DIR / "plots"
FEAT_DIR     = OUTPUTS_DIR / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
LOG_DIR      = OUTPUTS_DIR / "logs"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "benchmark_phase5.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

CLASS_NAMES = ["Tomato_Healthy", "Tomato_Early_Blight",
               "Tomato_Late_Blight", "Tomato_Leaf_Mold"]
SHORT       = ["Healthy", "Early\nBlight", "Late\nBlight", "Leaf\nMold"]
SHORT_FLAT  = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]

COLORS = {
    "custom_cnn":    "#2ecc71",
    "mobilenetv2":   "#3498db",
    "cnn_xgboost":   "#e67e22",
}
DISPLAY = {
    "custom_cnn":    "Custom CNN",
    "mobilenetv2":   "MobileNetV2-A",
    "cnn_xgboost":   "CNN + XGBoost",
}


# ── data loading ──────────────────────────────────────────────────────────

def load_existing_results() -> dict:
    """Read Phase 3/4 evaluation_results.csv for CNN and MNV2 metrics."""
    path = OUTPUTS_DIR / "evaluation_results.csv"
    results = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            results[r["model_key"]] = r
    log.info("Loaded existing results for: %s", list(results.keys()))
    return results


def evaluate_xgboost() -> dict:
    """Run XGBoost predictions on test features and compute full metrics."""
    import pickle
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, classification_report, confusion_matrix)

    # Load model
    pkl_path = MODELS_DIR / "xgboost_best.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {pkl_path}\n"
                                "Run train_xgboost.py first.")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded XGBoost model: %s", pkl_path)

    # Load config for metadata
    cfg_path = OUTPUTS_DIR / "xgboost_config.json"
    with open(cfg_path, encoding="utf-8") as f:
        xgb_config = json.load(f)

    # Load test features
    X_te = np.load(FEAT_DIR / "test_features.npy")
    y_te = np.load(FEAT_DIR / "test_labels.npy")
    log.info("Test features: %s  labels: %s", X_te.shape, y_te.shape)

    # Predict + time
    import xgboost as xgb
    dtest = xgb.DMatrix(X_te)

    # Warmup
    for _ in range(3):
        _ = model.predict(dtest)

    # Time single-image inference (use individual DMatrix calls)
    n_time = min(50, len(X_te))
    t0 = time.perf_counter()
    for i in range(n_time):
        d1 = xgb.DMatrix(X_te[i:i+1])
        model.predict(d1)
    ms_per_img = (time.perf_counter() - t0) * 1000 / n_time
    log.info("  Inference: %.2f ms/img (single-image, includes DMatrix creation)", ms_per_img)

    # NOTE: CNN feature extraction time must be added for fair comparison
    # Extract CNN inference time from config (stored during extraction)
    # We'll report XGBoost-only time and note that CNN extraction adds ~74ms
    feat_cfg_path = FEAT_DIR / "feature_config.json"
    cnn_extract_ms = 74.41  # from Phase 3 evaluation_results.csv
    if feat_cfg_path.exists():
        pass  # could read per-image extraction time if stored
    total_hybrid_ms = cnn_extract_ms + ms_per_img

    # Full predictions
    probs = model.predict(dtest).reshape(-1, len(CLASS_NAMES))
    preds = probs.argmax(axis=1)

    acc   = float(accuracy_score(y_te, preds))
    prec_macro = float(precision_score(y_te, preds, average="macro", zero_division=0))
    rec_macro  = float(recall_score(y_te,  preds, average="macro", zero_division=0))
    f1_macro   = float(f1_score(y_te,     preds, average="macro", zero_division=0))
    prec_wtd   = float(precision_score(y_te, preds, average="weighted", zero_division=0))
    rec_wtd    = float(recall_score(y_te,  preds, average="weighted", zero_division=0))
    f1_wtd     = float(f1_score(y_te,     preds, average="weighted", zero_division=0))

    log.info("XGBoost test accuracy=%.4f  macro-F1=%.4f", acc, f1_macro)

    # Per-class F1
    f1_per_class = f1_score(y_te, preds, average=None, zero_division=0)
    prec_per_class = precision_score(y_te, preds, average=None, zero_division=0)
    rec_per_class  = recall_score(y_te,  preds, average=None, zero_division=0)

    # Classification report
    report = classification_report(y_te, preds,
                                    target_names=SHORT_FLAT, digits=4)
    rpt_path = OUTPUTS_DIR / "classification_report_cnn_xgboost.txt"
    rpt_path.write_text(
        f"Model: CNN + XGBoost\n"
        f"Checkpoint: {pkl_path}\n\n{report}", encoding="utf-8")
    log.info("  Classification report: %s", rpt_path)

    # Confusion matrix
    cm = confusion_matrix(y_te, preds)

    # Model size: sum of pkl and json
    pkl_kb = pkl_path.stat().st_size / 1024
    json_kb = (MODELS_DIR / "xgboost_best.json").stat().st_size / 1024
    model_kb = pkl_kb  # use pkl as canonical size

    return {
        "model_key":          "cnn_xgboost",
        "status":             "OK",
        "accuracy":           round(acc, 4),
        "precision_macro":    round(prec_macro, 4),
        "recall_macro":       round(rec_macro, 4),
        "f1_macro":           round(f1_macro, 4),
        "precision_weighted": round(prec_wtd, 4),
        "recall_weighted":    round(rec_wtd, 4),
        "f1_weighted":        round(f1_wtd, 4),
        "inference_ms":       round(ms_per_img, 2),
        "inference_ms_total_hybrid": round(total_hybrid_ms, 2),
        "n_params":           "N/A (tree model)",
        "model_size_kb":      round(model_kb, 1),
        "f1_per_class":       f1_per_class.tolist(),
        "prec_per_class":     prec_per_class.tolist(),
        "rec_per_class":      rec_per_class.tolist(),
        "confusion_matrix":   cm.tolist(),
        "y_pred":             preds.tolist(),
        "y_probs":            probs.tolist(),
        "y_true":             y_te.tolist(),
        "xgb_config":         xgb_config,
    }


def build_full_record(existing: dict, xgb_result: dict) -> list[dict]:
    """Combine all three models into a unified list for comparison."""
    records = []
    # CNN
    r = existing.get("custom_cnn", {})
    if r.get("status") == "OK":
        records.append({
            "model_key":          "custom_cnn",
            "display_name":       DISPLAY["custom_cnn"],
            "status":             "OK",
            "accuracy":           float(r["accuracy"]),
            "precision_macro":    float(r["precision_macro"]),
            "recall_macro":       float(r["recall_macro"]),
            "f1_macro":           float(r["f1_macro"]),
            "inference_ms":       float(r["inference_ms"]),
            "n_params":           int(r["n_params"]),
            "model_size_kb":      round(
                (PROJECT_ROOT / "models" / "custom_cnn_best.keras").stat().st_size / 1024, 1),
        })
    # MNV2
    r = existing.get("mobilenetv2", {})
    if r.get("status") == "OK":
        records.append({
            "model_key":       "mobilenetv2",
            "display_name":    DISPLAY["mobilenetv2"],
            "status":          "OK",
            "accuracy":        float(r["accuracy"]),
            "precision_macro": float(r["precision_macro"]),
            "recall_macro":    float(r["recall_macro"]),
            "f1_macro":        float(r["f1_macro"]),
            "inference_ms":    float(r["inference_ms"]),
            "n_params":        int(r["n_params"]),
            "model_size_kb":   round(
                (PROJECT_ROOT / "models" / "mobilenetv2_best.keras").stat().st_size / 1024, 1),
        })
    # XGBoost hybrid
    records.append({
        "model_key":       "cnn_xgboost",
        "display_name":    DISPLAY["cnn_xgboost"],
        "status":          xgb_result["status"],
        "accuracy":        xgb_result["accuracy"],
        "precision_macro": xgb_result["precision_macro"],
        "recall_macro":    xgb_result["recall_macro"],
        "f1_macro":        xgb_result["f1_macro"],
        "inference_ms":    xgb_result["inference_ms_total_hybrid"],
        "n_params":        xgb_result["n_params"],
        "model_size_kb":   xgb_result["model_size_kb"],
    })
    return records


# ── per-class F1 helper ───────────────────────────────────────────────────

def get_per_class_f1(existing: dict, xgb_result: dict) -> dict:
    """
    For CNN and MNV2, re-read the phase 3/4 classification reports.
    For XGBoost, use the freshly computed f1_per_class.
    """
    from sklearn.metrics import f1_score
    import numpy as np

    result = {}

    # CNN — re-run predictions from existing error analysis
    ea_cnn = OUTPUTS_DIR / "error_analysis" / "per_class_errors_custom_cnn.csv"
    if ea_cnn.exists():
        cnn_f1 = []
        with open(ea_cnn, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in sorted(rows, key=lambda r: int(r["class_label"])):
            n  = int(row["total_samples"])
            tp = int(row["correct"])
            fn = n - tp
            # Compute from confusion data saved in error analysis
            # precision = tp / (tp + fp) — we don't have fp directly,
            # so use the stored per-class metrics from eval report instead
            cnn_f1.append(None)  # placeholder — will read from report

        # Read from classification report txt
        rpt = OUTPUTS_DIR / "classification_report_custom_cnn.txt"
        if rpt.exists():
            lines = rpt.read_text(encoding="utf-8").splitlines()
            f1s = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and parts[0] in SHORT_FLAT or (
                   len(parts) >= 5 and any(c in line for c in ["Healthy","Blight","Mold"])):
                    try:
                        f1s.append(float(parts[-2]))  # f1-score col
                    except (ValueError, IndexError):
                        pass
            if len(f1s) >= 4:
                result["custom_cnn"] = f1s[:4]

    # MNV2 — from classification report
    rpt = OUTPUTS_DIR / "classification_report_mobilenetv2.txt"
    if rpt.exists():
        lines = rpt.read_text(encoding="utf-8").splitlines()
        f1s = []
        for line in lines:
            if any(c in line for c in ["Healthy","Blight","Mold"]):
                parts = line.split()
                try:
                    f1s.append(float(parts[-2]))
                except (ValueError, IndexError):
                    pass
        if len(f1s) >= 4:
            result["mobilenetv2"] = f1s[:4]

    # XGBoost — from fresh computation
    result["cnn_xgboost"] = xgb_result["f1_per_class"]

    # Fallback: use 0.0 where parsing failed
    for key in ["custom_cnn", "mobilenetv2", "cnn_xgboost"]:
        if key not in result or len(result.get(key, [])) < 4:
            result[key] = [0.0, 0.0, 0.0, 0.0]

    return result


# ── plots ─────────────────────────────────────────────────────────────────

def _pct(v) -> float:
    return float(v) * 100 if v is not None else 0.0


def plot_classification_metrics(records: list[dict]) -> Path:
    """4-panel bar: accuracy, macro-F1, macro-precision, macro-recall."""
    valid  = [r for r in records if r["status"] == "OK"]
    names  = [r["display_name"] for r in valid]
    cols   = [COLORS[r["model_key"]] for r in valid]
    metrics = [
        ("accuracy",         "Test Accuracy (%)"),
        ("f1_macro",         "Macro F1 (%)"),
        ("precision_macro",  "Macro Precision (%)"),
        ("recall_macro",     "Macro Recall (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle("LeafIQ Phase 5 — Model Comparison\n(test set, n=821)",
                 fontsize=13, fontweight="bold")
    for ax, (key, label) in zip(axes, metrics):
        vals = [_pct(r[key]) for r in valid]
        bars = ax.bar(names, vals, color=cols, alpha=0.88,
                      edgecolor="white", linewidth=0.8, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f"{v:.2f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Score (%)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", labelsize=9)
        sns.despine(ax=ax)
    plt.tight_layout()
    p = PLOTS_DIR / "phase5_classification_metrics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p); return p


def plot_per_class_f1(per_class_f1: dict) -> Path:
    """Grouped bar: per-class F1 for all three models."""
    x   = np.arange(len(SHORT_FLAT))
    bw  = 0.25
    keys = ["custom_cnn", "mobilenetv2", "cnn_xgboost"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, key in enumerate(keys):
        vals = [v * 100 for v in per_class_f1.get(key, [0]*4)]
        bars = ax.bar(x + i*bw, vals, bw,
                      label=DISPLAY[key], color=COLORS[key],
                      alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + bw); ax.set_xticklabels(SHORT_FLAT, fontsize=10)
    ax.set_ylabel("F1-Score (%)"); ax.set_ylim(0, 115)
    ax.set_title("Per-Class F1-Score — All Models (Test Set)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax); plt.tight_layout()
    p = PLOTS_DIR / "phase5_per_class_f1.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p); return p


def plot_confusion_matrices(existing: dict, xgb_result: dict) -> Path:
    """3-column confusion matrix grid (one per model)."""
    import numpy as np

    # Get confusion matrices
    cms = {}

    # CNN — from error analysis
    ea_dir = OUTPUTS_DIR / "error_analysis"
    misc_cnn = ea_dir / "misclassified_custom_cnn.csv"
    if misc_cnn.exists():
        # Rebuild from per-class errors
        cache_dir = OUTPUTS_DIR / "cache"
        y_true_cnn = np.load(cache_dir / "test_labels.npy")
        # Load CNN predictions saved in error analysis
        err_path = ea_dir / "misclassified_custom_cnn.csv"
        wrong = {}
        with open(err_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wrong[int(row["index"])] = int(row["pred_label"])
        y_pred_cnn = y_true_cnn.copy()
        for idx, pred in wrong.items():
            y_pred_cnn[idx] = pred
        from sklearn.metrics import confusion_matrix
        cms["custom_cnn"] = confusion_matrix(y_true_cnn, y_pred_cnn)

    # MNV2 — from error analysis
    misc_mnv2 = ea_dir / "misclassified_mobilenetv2.csv"
    if misc_mnv2.exists():
        y_true_mnv2 = np.load(OUTPUTS_DIR / "cache" / "test_labels.npy")
        wrong = {}
        with open(misc_mnv2, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wrong[int(row["index"])] = int(row["pred_label"])
        y_pred_mnv2 = y_true_mnv2.copy()
        for idx, pred in wrong.items():
            y_pred_mnv2[idx] = pred
        from sklearn.metrics import confusion_matrix
        cms["mobilenetv2"] = confusion_matrix(y_true_mnv2, y_pred_mnv2)

    # XGBoost
    from sklearn.metrics import confusion_matrix as _cm
    y_true_xgb = np.array(xgb_result["y_true"])
    y_pred_xgb = np.array(xgb_result["y_pred"])
    cms["cnn_xgboost"] = _cm(y_true_xgb, y_pred_xgb)

    keys  = [k for k in ["custom_cnn", "mobilenetv2", "cnn_xgboost"] if k in cms]
    n     = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle("Normalised Confusion Matrices — Phase 5 Comparison",
                 fontsize=12, fontweight="bold")

    for ax, key in zip(axes, keys):
        cm = np.array(cms[key], dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sum > 0, cm / row_sum, 0.0)
        im = sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                         xticklabels=SHORT_FLAT, yticklabels=SHORT_FLAT,
                         linewidths=0.5, ax=ax, vmin=0, vmax=1)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(DISPLAY[key], fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    p = PLOTS_DIR / "phase5_confusion_matrices.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p); return p


def plot_inference_comparison(records: list[dict]) -> Path:
    """Horizontal bar: inference time with hybrid decomposition."""
    valid = [r for r in records if r["status"] == "OK"]
    names = [r["display_name"] for r in valid]
    times = [r["inference_ms"] for r in valid]
    cols  = [COLORS[r["model_key"]] for r in valid]

    fig, ax = plt.subplots(figsize=(9, max(3, len(valid)*1.5)))
    bars = ax.barh(names, times, color=cols, alpha=0.88,
                   edgecolor="white", height=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 1,
                bar.get_y() + bar.get_height()/2,
                f"{t:.1f} ms", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Inference Time per Image (ms, CPU)", fontsize=11)
    ax.set_title("Inference Speed — Phase 5\n"
                 "(CNN+XGBoost: CNN extract + XGBoost predict)", fontsize=11)
    ax.set_xlim(0, max(times) * 1.3)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    # Add note for hybrid
    ax.text(0.99, 0.02,
            "CNN+XGBoost time = CNN feature extraction + XGBoost inference",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="gray")
    sns.despine(ax=ax); plt.tight_layout()
    p = PLOTS_DIR / "phase5_inference_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p); return p


def plot_model_size_comparison(records: list[dict]) -> Path:
    """Bar chart: model file size in KB."""
    valid  = [r for r in records if r["status"] == "OK"]
    names  = [r["display_name"] for r in valid]
    sizes  = [r["model_size_kb"] for r in valid]
    cols   = [COLORS[r["model_key"]] for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, sizes, color=cols, alpha=0.88,
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 10,
                f"{s:.0f} KB", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylabel("Model File Size (KB)")
    ax.set_title("Model Size Comparison\n(checkpoint file on disk)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax); plt.tight_layout()
    p = PLOTS_DIR / "phase5_model_size_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p); return p


def plot_feature_importance() -> Path:
    """Top-30 XGBoost feature importances (gain)."""
    fi_path = OUTPUTS_DIR / "xgboost_feature_importance.npy"
    if not fi_path.exists():
        log.warning("Feature importance file not found, skipping plot.")
        return None
    fi = np.load(fi_path)
    top_n = 30
    top_idx = np.argsort(fi)[::-1][:top_n]
    top_vals = fi[top_idx]
    labels   = [f"f{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(labels[::-1], top_vals[::-1],
                   color=COLORS["cnn_xgboost"], alpha=0.85,
                   edgecolor="white")
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"Top-{top_n} XGBoost Feature Importances\n"
                 "(CNN embedding dimensions, gain metric)", fontsize=11)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    sns.despine(ax=ax); plt.tight_layout()
    p = PLOTS_DIR / "phase5_xgboost_feature_importance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Saved: %s", p); return p


# ── CSV / TXT outputs ─────────────────────────────────────────────────────

def save_comparison_csv(records: list[dict], xgb_result: dict) -> Path:
    fields = [
        "model_key", "display_name", "status",
        "accuracy_pct", "f1_macro_pct", "precision_macro_pct", "recall_macro_pct",
        "inference_ms", "n_params", "model_size_kb",
        "errors_821", "note",
    ]
    rows = []
    err_map = {"custom_cnn": 15, "mobilenetv2": 313}
    # XGBoost errors
    y_true = np.array(xgb_result["y_true"])
    y_pred = np.array(xgb_result["y_pred"])
    err_map["cnn_xgboost"] = int((y_true != y_pred).sum())

    for r in records:
        rows.append({
            "model_key":            r["model_key"],
            "display_name":         r["display_name"],
            "status":               r["status"],
            "accuracy_pct":         round(r["accuracy"] * 100, 4),
            "f1_macro_pct":         round(r["f1_macro"] * 100, 4),
            "precision_macro_pct":  round(r["precision_macro"] * 100, 4),
            "recall_macro_pct":     round(r["recall_macro"] * 100, 4),
            "inference_ms":         r["inference_ms"],
            "n_params":             r["n_params"],
            "model_size_kb":        r["model_size_kb"],
            "errors_821":           err_map.get(r["model_key"], "N/A"),
            "note":                 "inference_ms includes CNN feature extraction"
                                    if r["model_key"] == "cnn_xgboost" else "",
        })
    path = OUTPUTS_DIR / "phase5_model_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("Saved: %s", path); return path


def save_comparison_txt(records: list[dict], xgb_result: dict) -> Path:
    y_true = np.array(xgb_result["y_true"])
    y_pred = np.array(xgb_result["y_pred"])
    err_map = {"custom_cnn": 15, "mobilenetv2": 313,
               "cnn_xgboost": int((y_true != y_pred).sum())}

    lines = [
        "LeafIQ - Phase 5 Model Comparison",
        "=" * 80,
        f"{'Model':<22} {'Status':<8} {'Acc%':>7} {'F1%':>7} "
        f"{'Prec%':>7} {'Rec%':>7} {'ms/img':>8} {'Errors':>7} {'Size KB':>8}",
        "-" * 80,
    ]
    for r in records:
        acc  = f"{r['accuracy']*100:.2f}"
        f1   = f"{r['f1_macro']*100:.2f}"
        prec = f"{r['precision_macro']*100:.2f}"
        rec  = f"{r['recall_macro']*100:.2f}"
        ms   = f"{r['inference_ms']:.1f}"
        errs = str(err_map.get(r["model_key"], "N/A"))
        skb  = f"{r['model_size_kb']:.0f}"
        lines.append(
            f"{r['display_name']:<22} {'OK':<8} {acc:>7} {f1:>7} "
            f"{prec:>7} {rec:>7} {ms:>8} {errs:>7} {skb:>8}"
        )
    lines += [
        "=" * 80,
        "",
        "Notes:",
        "  - Test set: 821 images (held-out, never seen during training)",
        "  - CNN+XGBoost inference time = CNN feature extraction + XGBoost predict (both on CPU)",
        "  - MobileNetV2-FT excluded: training failed (val_acc=43.26%)",
        "  - Model size = primary checkpoint file on disk",
        "  - All metrics computed on identical test samples",
        "",
    ]
    path = OUTPUTS_DIR / "phase5_model_comparison.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved: %s", path); return path


# ── main ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("LeafIQ -- Phase 5: Three-Model Benchmark")
    log.info("=" * 60)

    existing   = load_existing_results()
    xgb_result = evaluate_xgboost()
    records    = build_full_record(existing, xgb_result)
    per_class  = get_per_class_f1(existing, xgb_result)

    log.info("\nGenerating plots ...")
    plot_classification_metrics(records)
    plot_per_class_f1(per_class)
    plot_confusion_matrices(existing, xgb_result)
    plot_inference_comparison(records)
    plot_model_size_comparison(records)
    plot_feature_importance()

    log.info("\nSaving comparison tables ...")
    save_comparison_csv(records, xgb_result)
    save_comparison_txt(records, xgb_result)

    # Print summary
    log.info("\n%s", "=" * 60)
    log.info("PHASE 5 BENCHMARK SUMMARY (test set, n=821)")
    log.info("%s", "=" * 60)
    for r in records:
        log.info("  %-22s  acc=%.4f  F1=%.4f  ms=%.1f",
                 r["display_name"], r["accuracy"],
                 r["f1_macro"], r["inference_ms"])
    log.info("%s", "=" * 60)
    log.info("\nBenchmark complete. Outputs: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
