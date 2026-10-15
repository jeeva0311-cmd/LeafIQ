"""
LeafIQ - Phase 4: Error Analysis
==================================
Loads the best-checkpoint models and runs detailed error analysis on the
held-out test set.  Never retrains — loads checkpoints and cached test data.

Outputs  (outputs/error_analysis/):
  misclassified_<model>.csv          — every wrong prediction (image, true, pred, conf)
  per_class_errors_<model>.csv       — per-class error counts + error rate
  confidence_distribution_<model>.csv — confidence bucket summary
  error_analysis_<model>.png         — 4-panel error-analysis figure
  misclassified_grid_<model>.png     — 5x4 mosaic of worst-confidence wrong images
  top_errors_<model>.png             — top-10 most confident wrong predictions

Outputs  (outputs/):
  error_analysis_summary.csv         — one-row-per-model summary used by report

Usage:
    python src/error_analysis.py                  # analyse both models
    python src/error_analysis.py --model cnn      # custom_cnn only
    python src/error_analysis.py --model mnv2     # mobilenetv2 only
"""

import os, sys, csv, json, time, argparse, logging, warnings
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
OUTPUTS      = PROJECT_ROOT / "outputs"
PLOTS        = OUTPUTS / "plots"
EA_DIR       = OUTPUTS / "error_analysis"
LOG_DIR      = OUTPUTS / "logs"

for d in (PLOTS, EA_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "error_analysis.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
]
SHORT = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]

MODEL_CONFIGS = {
    "custom_cnn": {
        "path":      PROJECT_ROOT / "models" / "custom_cnn_best.keras",
        "input_hw":  64,
        "label":     "Custom CNN",
        "color":     "#2ecc71",
    },
    "mobilenetv2": {
        "path":      PROJECT_ROOT / "models" / "mobilenetv2_best.keras",
        "input_hw":  96,
        "label":     "MobileNetV2-A",
        "color":     "#3498db",
    },
}

# ── helpers ────────────────────────────────────────────────────────────────

def _resize_batch(images: np.ndarray, hw: int) -> np.ndarray:
    """Bilinear resize a (N,H,W,3) float32 batch to (N,hw,hw,3)."""
    # Use tf only inside the function to avoid top-level import noise
    import tensorflow as tf
    return tf.image.resize(images, [hw, hw], method="bilinear").numpy()


def load_test_data():
    imgs  = np.load(OUTPUTS / "cache" / "test_images.npy",  mmap_mode="r")
    labs  = np.load(OUTPUTS / "cache" / "test_labels.npy")
    paths = np.load(OUTPUTS / "cache" / "test_paths.npy",   allow_pickle=True)
    log.info("Test cache: %s  labels: %s  paths: %s",
             imgs.shape, labs.shape, paths.shape)
    return imgs, labs, paths


def load_model(cfg: dict):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(cfg["path"])
    log.info("Loaded %s  input=%s  params=%d",
             cfg["label"], model.input_shape, model.count_params())
    return model


# ── prediction ─────────────────────────────────────────────────────────────

def run_predictions(model, images: np.ndarray, hw: int,
                    batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_classes, pred_probs) — pred_probs shape (N, n_classes)."""
    log.info("  Resizing %d images to %dx%d ...", len(images), hw, hw)
    resized = _resize_batch(images, hw)          # (N, hw, hw, 3)

    all_probs = []
    n = len(resized)
    for start in range(0, n, batch_size):
        batch = resized[start : start + batch_size]
        probs = model.predict(batch, verbose=0)  # (batch, n_classes)
        all_probs.append(probs)
    probs_all  = np.concatenate(all_probs, axis=0)   # (N, n_classes)
    preds      = np.argmax(probs_all, axis=1)
    return preds, probs_all


# ── CSV exports ─────────────────────────────────────────────────────────────

def save_misclassified_csv(
    key: str, labels: np.ndarray, preds: np.ndarray,
    probs: np.ndarray, paths: np.ndarray
) -> tuple[Path, list[dict]]:
    wrong_idx = np.where(preds != labels)[0]
    rows = []
    for i in wrong_idx:
        conf = float(probs[i, preds[i]])
        rows.append({
            "index":           int(i),
            "image_path":      str(paths[i]),
            "true_label":      int(labels[i]),
            "true_class":      CLASS_NAMES[labels[i]],
            "pred_label":      int(preds[i]),
            "pred_class":      CLASS_NAMES[preds[i]],
            "confidence":      round(conf, 6),
            "true_class_prob": round(float(probs[i, labels[i]]), 6),
        })
    rows.sort(key=lambda r: -r["confidence"])       # highest-conf errors first
    path = EA_DIR / f"misclassified_{key}.csv"
    fields = ["index","image_path","true_label","true_class",
              "pred_label","pred_class","confidence","true_class_prob"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("  Misclassified CSV: %d errors saved to %s", len(rows), path)
    return path, rows


def save_per_class_errors_csv(
    key: str, labels: np.ndarray, preds: np.ndarray
) -> Path:
    n_classes = len(CLASS_NAMES)
    rows = []
    for c in range(n_classes):
        mask       = labels == c
        total      = int(mask.sum())
        correct    = int((preds[mask] == c).sum())
        errors     = total - correct
        error_rate = errors / total if total > 0 else 0.0
        # confusion: where did this class get confused to?
        confused_to = {}
        for p in preds[mask]:
            if p != c:
                confused_to[CLASS_NAMES[p]] = confused_to.get(CLASS_NAMES[p], 0) + 1
        rows.append({
            "class_label":      c,
            "class_name":       CLASS_NAMES[c],
            "total_samples":    total,
            "correct":          correct,
            "errors":           errors,
            "error_rate_pct":   round(error_rate * 100, 4),
            "confused_to":      "; ".join(f"{k}:{v}" for k, v in
                                          sorted(confused_to.items(), key=lambda x: -x[1])),
        })
    path = EA_DIR / f"per_class_errors_{key}.csv"
    fields = ["class_label","class_name","total_samples","correct",
              "errors","error_rate_pct","confused_to"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("  Per-class CSV saved to %s", path)
    return path


def save_confidence_csv(
    key: str, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> Path:
    max_conf = np.max(probs, axis=1)
    correct  = (preds == labels).astype(int)
    bins     = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001]
    bin_labels = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-95%", ">=95%"]
    rows = []
    for i in range(len(bins) - 1):
        mask  = (max_conf >= bins[i]) & (max_conf < bins[i+1])
        total = int(mask.sum())
        corr  = int(correct[mask].sum())
        rows.append({
            "confidence_bin":    bin_labels[i],
            "total_predictions": total,
            "correct":           corr,
            "errors":            total - corr,
            "accuracy_in_bin":   round(corr / total * 100, 2) if total > 0 else "N/A",
        })
    path = EA_DIR / f"confidence_distribution_{key}.csv"
    fields = ["confidence_bin","total_predictions","correct","errors","accuracy_in_bin"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("  Confidence CSV saved to %s", path)
    return path


# ── plots ──────────────────────────────────────────────────────────────────

def _cm_matrix(labels, preds, n=4) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_error_analysis(
    key: str, cfg: dict,
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray,
    wrong_rows: list[dict]
) -> Path:
    """4-panel figure: confusion matrix | error rate bar | confidence hist | top-5 confusion pairs."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Error Analysis — {cfg['label']}\n"
                 f"(test set n={len(labels)}, "
                 f"errors={len(wrong_rows)}, "
                 f"accuracy={100*(1-len(wrong_rows)/len(labels)):.2f}%)",
                 fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_cm   = fig.add_subplot(gs[0, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[1, 0])
    ax_pair = fig.add_subplot(gs[1, 1])

    # ── panel 1: normalised confusion matrix ──
    cm = _cm_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(4):
        for j in range(4):
            val   = cm_norm[i, j]
            raw   = cm[i, j]
            color = "white" if val > 0.6 else "black"
            ax_cm.text(j, i, f"{val:.2f}\n({raw})",
                       ha="center", va="center", fontsize=7.5, color=color)
    ax_cm.set_xticks(range(4)); ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(SHORT, rotation=30, ha="right", fontsize=8)
    ax_cm.set_yticklabels(SHORT, fontsize=8)
    ax_cm.set_xlabel("Predicted", fontsize=9); ax_cm.set_ylabel("True", fontsize=9)
    ax_cm.set_title("Normalised Confusion Matrix", fontsize=10)

    # ── panel 2: per-class error rate bar ──
    error_rates = []
    for c in range(4):
        mask = labels == c
        err  = (preds[mask] != c).sum()
        rate = err / mask.sum() * 100 if mask.sum() > 0 else 0
        error_rates.append(rate)
    bars = ax_err.bar(SHORT, error_rates, color=cfg["color"],
                      alpha=0.85, edgecolor="white", linewidth=0.8)
    for bar, rate in zip(bars, error_rates):
        if rate > 0:
            ax_err.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f"{rate:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    ax_err.set_title("Per-Class Error Rate", fontsize=10)
    ax_err.set_ylabel("Error Rate (%)")
    ax_err.set_ylim(0, max(error_rates) * 1.35 + 1)
    ax_err.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax_err)

    # ── panel 3: confidence histogram (correct vs wrong) ──
    max_conf = np.max(probs, axis=1)
    correct_mask = preds == labels
    ax_conf.hist(max_conf[correct_mask]  * 100, bins=20, alpha=0.7,
                 color="#27ae60", label=f"Correct ({correct_mask.sum()})")
    ax_conf.hist(max_conf[~correct_mask] * 100, bins=20, alpha=0.7,
                 color="#e74c3c", label=f"Wrong ({(~correct_mask).sum()})")
    ax_conf.set_xlabel("Max Predicted Probability (%)")
    ax_conf.set_ylabel("Count")
    ax_conf.set_title("Confidence Distribution\n(correct vs misclassified)", fontsize=10)
    ax_conf.legend(fontsize=8)
    sns.despine(ax=ax_conf)

    # ── panel 4: top confusion pairs ──
    pair_counts: dict[tuple, int] = {}
    for t, p in zip(labels[~correct_mask], preds[~correct_mask]):
        pair_counts[(t, p)] = pair_counts.get((t, p), 0) + 1
    if pair_counts:
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:8]
        pair_labels = [f"{SHORT[t]} →\n{SHORT[p]}" for (t, p), _ in top_pairs]
        pair_vals   = [v for _, v in top_pairs]
        y_pos = range(len(pair_labels))
        ax_pair.barh(list(y_pos), pair_vals, color=cfg["color"],
                     alpha=0.85, edgecolor="white", height=0.6)
        for i, v in enumerate(pair_vals):
            ax_pair.text(v + 0.05, i, str(v), va="center", fontsize=9,
                         fontweight="bold")
        ax_pair.set_yticks(list(y_pos))
        ax_pair.set_yticklabels(pair_labels, fontsize=8)
        ax_pair.set_xlabel("Number of Errors")
        ax_pair.set_title("Top Confusion Pairs\n(true → predicted)", fontsize=10)
        ax_pair.set_xlim(0, max(pair_vals) * 1.3)
        sns.despine(ax=ax_pair)
    else:
        ax_pair.text(0.5, 0.5, "No errors to display",
                     ha="center", va="center", transform=ax_pair.transAxes,
                     fontsize=12, color="gray")
        ax_pair.set_title("Top Confusion Pairs", fontsize=10)
        ax_pair.axis("off")

    path = EA_DIR / f"error_analysis_{key}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Error analysis figure saved: %s", path)
    return path


def plot_misclassified_grid(
    key: str, cfg: dict,
    images: np.ndarray, labels: np.ndarray, preds: np.ndarray,
    probs: np.ndarray, paths: np.ndarray,
    max_show: int = 20
) -> Path:
    """Grid of up to 20 misclassified images (highest-confidence errors first)."""
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) == 0:
        log.info("  No errors for %s — skipping grid.", key)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications!", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#27ae60")
        ax.axis("off")
        path = EA_DIR / f"misclassified_grid_{key}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig); return path

    # Sort by confidence (descending)
    confs     = np.array([probs[i, preds[i]] for i in wrong_idx])
    order     = np.argsort(-confs)
    wrong_idx = wrong_idx[order][:max_show]

    n_show = len(wrong_idx)
    ncols  = 5
    nrows  = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.8, nrows * 3.0))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Misclassified Images — {cfg['label']}\n"
        f"(showing {n_show} of {len(wrong_idx)} errors, sorted by confidence↓)",
        fontsize=11, fontweight="bold"
    )

    for plot_i, idx in enumerate(wrong_idx):
        ax  = axes[plot_i]
        img = np.clip(images[idx], 0.0, 1.0)
        ax.imshow(img)
        true_cls = SHORT[labels[idx]]
        pred_cls = SHORT[preds[idx]]
        conf     = probs[idx, preds[idx]] * 100
        ax.set_title(
            f"True: {true_cls}\nPred: {pred_cls}\n({conf:.1f}%)",
            fontsize=6.5, color="#c0392b"
        )
        ax.axis("off")

    for ax in axes[n_show:]:
        ax.axis("off")

    plt.tight_layout()
    path = EA_DIR / f"misclassified_grid_{key}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("  Misclassified grid saved: %s", path)
    return path


def plot_top_confident_errors(
    key: str, cfg: dict,
    images: np.ndarray, labels: np.ndarray, preds: np.ndarray,
    probs: np.ndarray, n_show: int = 10
) -> Path:
    """Show top-N most confidently wrong predictions."""
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) == 0:
        log.info("  No errors for %s — skipping top errors plot.", key)
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications!", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#27ae60")
        ax.axis("off")
        path = EA_DIR / f"top_errors_{key}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
        return path

    confs = np.array([probs[i, preds[i]] for i in wrong_idx])
    order = np.argsort(-confs)
    top   = wrong_idx[order[:n_show]]

    ncols = min(5, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.0, nrows * 3.2))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Top-{n_show} Most Confident Wrong Predictions — {cfg['label']}",
        fontsize=11, fontweight="bold"
    )

    for plot_i, idx in enumerate(top):
        ax   = axes[plot_i]
        img  = np.clip(images[idx], 0.0, 1.0)
        conf = probs[idx, preds[idx]] * 100

        # Show all 4 class probs as a mini bar chart overlay via title
        bar_str = "  ".join(
            f"{SHORT[c][0]}:{probs[idx,c]*100:.0f}%"
            for c in range(4)
        )
        ax.imshow(img)
        ax.set_title(
            f"True: {SHORT[labels[idx]]}\n"
            f"Pred: {SHORT[preds[idx]]} ({conf:.1f}%)\n"
            f"{bar_str}",
            fontsize=6.0, color="#c0392b"
        )
        ax.axis("off")

    for ax in axes[len(top):]:
        ax.axis("off")

    plt.tight_layout()
    path = EA_DIR / f"top_errors_{key}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("  Top-errors plot saved: %s", path)
    return path


# ── summary CSV ────────────────────────────────────────────────────────────

def update_summary_csv(summary_rows: list[dict]) -> Path:
    path   = OUTPUTS / "error_analysis_summary.csv"
    fields = [
        "model_key", "display_name", "total_test", "total_errors",
        "error_rate_pct", "accuracy_pct",
        "most_confused_true", "most_confused_pred", "most_confused_count",
        "highest_error_class", "highest_error_rate_pct",
        "lowest_error_class",  "lowest_error_rate_pct",
        "pct_errors_above_90pct_conf",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)
    log.info("Error analysis summary saved: %s", path)
    return path


# ── per-model analysis ─────────────────────────────────────────────────────

def analyse_model(key: str, images: np.ndarray, labels: np.ndarray,
                  paths: np.ndarray) -> dict:
    cfg = MODEL_CONFIGS[key]
    log.info("\n%s", "=" * 60)
    log.info("Analysing: %s", cfg["label"])
    log.info("%s", "=" * 60)

    model = load_model(cfg)
    preds, probs = run_predictions(model, images, cfg["input_hw"])
    del model   # free memory

    n_total  = len(labels)
    n_errors = int((preds != labels).sum())
    acc      = (n_total - n_errors) / n_total * 100
    log.info("  Total: %d  Errors: %d  Accuracy: %.2f%%", n_total, n_errors, acc)

    # CSVs
    _, wrong_rows = save_misclassified_csv(key, labels, preds, probs, paths)
    save_per_class_errors_csv(key, labels, preds)
    save_confidence_csv(key, labels, preds, probs)

    # Plots
    plot_error_analysis(key, cfg, labels, preds, probs, wrong_rows)
    plot_misclassified_grid(key, cfg, images, labels, preds, probs, paths)
    plot_top_confident_errors(key, cfg, images, labels, preds, probs)

    # ── build summary row ──
    per_class_err = []
    for c in range(4):
        mask = labels == c
        if mask.sum() == 0: continue
        rate = (preds[mask] != c).sum() / mask.sum() * 100
        per_class_err.append((rate, CLASS_NAMES[c]))

    highest_err = max(per_class_err, key=lambda x: x[0])
    lowest_err  = min(per_class_err, key=lambda x: x[0])

    # most confused pair
    pair_counts: dict[tuple, int] = {}
    wrong_mask = preds != labels
    for t, p in zip(labels[wrong_mask], preds[wrong_mask]):
        pair_counts[(int(t), int(p))] = pair_counts.get((int(t), int(p)), 0) + 1
    if pair_counts:
        top_pair = max(pair_counts, key=lambda k: pair_counts[k])
        top_pair_count = pair_counts[top_pair]
    else:
        top_pair = (0, 0); top_pair_count = 0

    # high-conf errors (conf >= 0.90)
    max_conf    = np.max(probs, axis=1)
    hi_conf_err = int(((preds != labels) & (max_conf >= 0.90)).sum())
    pct_hi_conf = round(hi_conf_err / n_errors * 100, 2) if n_errors > 0 else 0.0

    summary = {
        "model_key":                   key,
        "display_name":                cfg["label"],
        "total_test":                  n_total,
        "total_errors":                n_errors,
        "error_rate_pct":              round(n_errors / n_total * 100, 4),
        "accuracy_pct":                round(acc, 4),
        "most_confused_true":          CLASS_NAMES[top_pair[0]],
        "most_confused_pred":          CLASS_NAMES[top_pair[1]],
        "most_confused_count":         top_pair_count,
        "highest_error_class":         highest_err[1],
        "highest_error_rate_pct":      round(highest_err[0], 4),
        "lowest_error_class":          lowest_err[1],
        "lowest_error_rate_pct":       round(lowest_err[0], 4),
        "pct_errors_above_90pct_conf": pct_hi_conf,
    }
    log.info("  Summary: errors=%d  highest_err_class=%s (%.1f%%)",
             n_errors, highest_err[1], highest_err[0])
    return summary


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeafIQ Phase 4 Error Analysis")
    parser.add_argument("--model", choices=["cnn", "mnv2", "all"],
                        default="all", help="Which model to analyse")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LeafIQ -- Phase 4: Error Analysis")
    log.info("=" * 60)

    images, labels, paths = load_test_data()

    keys_to_run = {
        "all":  list(MODEL_CONFIGS.keys()),
        "cnn":  ["custom_cnn"],
        "mnv2": ["mobilenetv2"],
    }[args.model]

    summaries = []
    for key in keys_to_run:
        s = analyse_model(key, images, labels, paths)
        summaries.append(s)

    update_summary_csv(summaries)

    log.info("\nError analysis complete.")
    log.info("  Outputs: %s", EA_DIR)
    log.info("  Summary: %s", OUTPUTS / "error_analysis_summary.csv")


if __name__ == "__main__":
    main()
