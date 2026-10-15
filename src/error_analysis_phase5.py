"""
LeafIQ - Phase 5: Hybrid Model Error Analysis
===============================================
Runs detailed error analysis on the CNN+XGBoost hybrid model on the
held-out test set.  Mirrors the structure of error_analysis.py (Phase 4)
so all outputs are consistent and comparable.

Outputs (outputs/error_analysis/):
  misclassified_cnn_xgboost.csv
  per_class_errors_cnn_xgboost.csv
  confidence_distribution_cnn_xgboost.csv
  error_analysis_cnn_xgboost.png        (4-panel figure)
  misclassified_grid_cnn_xgboost.png    (grid of misclassified images)
  top_errors_cnn_xgboost.png            (top-10 confident errors)

Usage:
    python src/error_analysis_phase5.py
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
FEAT_DIR     = OUTPUTS_DIR / "features"
CACHE_DIR    = OUTPUTS_DIR / "cache"
EA_DIR       = OUTPUTS_DIR / "error_analysis"
MODELS_DIR   = PROJECT_ROOT / "models"
LOG_DIR      = OUTPUTS_DIR / "logs"

EA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "error_analysis_phase5.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

CLASS_NAMES = ["Tomato_Healthy", "Tomato_Early_Blight",
               "Tomato_Late_Blight", "Tomato_Leaf_Mold"]
SHORT       = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]
COLOR       = "#e67e22"
KEY         = "cnn_xgboost"
DISPLAY     = "CNN + XGBoost"


# ── load model + predict ──────────────────────────────────────────────────

def load_and_predict():
    """Load XGBoost, predict on test features, return (probs, preds, labels, paths)."""
    import pickle, xgboost as xgb

    pkl_path = MODELS_DIR / "xgboost_best.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    log.info("Loaded XGBoost model: %s", pkl_path)

    X_te = np.load(FEAT_DIR / "test_features.npy")
    y_te = np.load(FEAT_DIR / "test_labels.npy")
    paths = np.load(CACHE_DIR / "test_paths.npy", allow_pickle=True)

    dtest = xgb.DMatrix(X_te)
    probs = model.predict(dtest).reshape(-1, len(CLASS_NAMES))
    preds = probs.argmax(axis=1).astype(np.int64)

    n_errors = int((preds != y_te).sum())
    acc      = float((preds == y_te).mean()) * 100
    log.info("Test: %d samples  errors=%d  accuracy=%.2f%%",
             len(y_te), n_errors, acc)
    return probs, preds, y_te, paths


# ── CSVs ──────────────────────────────────────────────────────────────────

def save_misclassified_csv(labels, preds, probs, paths) -> tuple[Path, list]:
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
    rows.sort(key=lambda r: -r["confidence"])
    path = EA_DIR / f"misclassified_{KEY}.csv"
    fields = ["index", "image_path", "true_label", "true_class",
              "pred_label", "pred_class", "confidence", "true_class_prob"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("Misclassified CSV: %d errors -> %s", len(rows), path)
    return path, rows


def save_per_class_errors_csv(labels, preds) -> Path:
    rows = []
    for c in range(len(CLASS_NAMES)):
        mask   = labels == c
        total  = int(mask.sum())
        errors = int((preds[mask] != c).sum())
        confused_to = {}
        for p in preds[mask]:
            if p != c:
                confused_to[CLASS_NAMES[p]] = confused_to.get(CLASS_NAMES[p], 0) + 1
        rows.append({
            "class_label":   c,
            "class_name":    CLASS_NAMES[c],
            "total_samples": total,
            "correct":       total - errors,
            "errors":        errors,
            "error_rate_pct": round(errors / total * 100, 4) if total else 0.0,
            "confused_to":   "; ".join(f"{k}:{v}" for k, v in
                                        sorted(confused_to.items(), key=lambda x: -x[1])),
        })
    path = EA_DIR / f"per_class_errors_{KEY}.csv"
    fields = ["class_label", "class_name", "total_samples", "correct",
              "errors", "error_rate_pct", "confused_to"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("Per-class CSV: %s", path)
    return path


def save_confidence_csv(labels, preds, probs) -> Path:
    max_conf = probs.max(axis=1)
    correct  = (preds == labels).astype(int)
    bins      = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.001]
    bin_labels= ["<50%","50-60%","60-70%","70-80%","80-90%","90-95%",">=95%"]
    rows = []
    for i in range(len(bins)-1):
        mask  = (max_conf >= bins[i]) & (max_conf < bins[i+1])
        total = int(mask.sum())
        corr  = int(correct[mask].sum())
        rows.append({
            "confidence_bin":    bin_labels[i],
            "total_predictions": total,
            "correct":           corr,
            "errors":            total - corr,
            "accuracy_in_bin":   round(corr/total*100, 2) if total else "N/A",
        })
    path = EA_DIR / f"confidence_distribution_{KEY}.csv"
    fields = ["confidence_bin","total_predictions","correct","errors","accuracy_in_bin"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    log.info("Confidence CSV: %s", path)
    return path


# ── plots ─────────────────────────────────────────────────────────────────

def _cm_matrix(labels, preds, n=4):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_error_analysis(labels, preds, probs, wrong_rows) -> Path:
    n_total  = len(labels)
    n_errors = len(wrong_rows)
    acc      = (n_total - n_errors) / n_total * 100

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Error Analysis — {DISPLAY}\n"
        f"(test set n={n_total}, errors={n_errors}, accuracy={acc:.2f}%)",
        fontsize=13, fontweight="bold"
    )
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_cm   = fig.add_subplot(gs[0, 0])
    ax_err  = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[1, 0])
    ax_pair = fig.add_subplot(gs[1, 1])

    # Confusion matrix
    cm = _cm_matrix(labels, preds)
    cm_norm = cm.astype(float) / np.where(cm.sum(axis=1, keepdims=True) > 0,
                                           cm.sum(axis=1, keepdims=True), 1)
    im = ax_cm.imshow(cm_norm, cmap="Oranges", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    for i in range(4):
        for j in range(4):
            v = cm_norm[i, j]; r = cm[i, j]
            c = "white" if v > 0.6 else "black"
            ax_cm.text(j, i, f"{v:.2f}\n({r})", ha="center", va="center",
                       fontsize=7.5, color=c)
    ax_cm.set_xticks(range(4)); ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(SHORT, rotation=30, ha="right", fontsize=8)
    ax_cm.set_yticklabels(SHORT, fontsize=8)
    ax_cm.set_xlabel("Predicted", fontsize=9); ax_cm.set_ylabel("True", fontsize=9)
    ax_cm.set_title("Normalised Confusion Matrix", fontsize=10)

    # Per-class error rate
    error_rates = []
    for c in range(4):
        mask = labels == c
        err  = (preds[mask] != c).sum()
        rate = err / mask.sum() * 100 if mask.sum() > 0 else 0
        error_rates.append(float(rate))
    bars = ax_err.bar(SHORT, error_rates, color=COLOR,
                      alpha=0.85, edgecolor="white", linewidth=0.8)
    for bar, rate in zip(bars, error_rates):
        if rate > 0:
            ax_err.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.1,
                        f"{rate:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    ax_err.set_title("Per-Class Error Rate", fontsize=10)
    ax_err.set_ylabel("Error Rate (%)")
    ax_err.set_ylim(0, max(error_rates+[1]) * 1.35 + 1)
    ax_err.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax_err)

    # Confidence histogram
    max_conf = probs.max(axis=1)
    correct_mask = preds == labels
    ax_conf.hist(max_conf[correct_mask]*100, bins=20, alpha=0.7,
                 color="#27ae60", label=f"Correct ({correct_mask.sum()})")
    ax_conf.hist(max_conf[~correct_mask]*100, bins=20, alpha=0.7,
                 color="#e74c3c", label=f"Wrong ({(~correct_mask).sum()})")
    ax_conf.set_xlabel("Max Predicted Probability (%)")
    ax_conf.set_ylabel("Count")
    ax_conf.set_title("Confidence Distribution\n(correct vs misclassified)", fontsize=10)
    ax_conf.legend(fontsize=8)
    sns.despine(ax=ax_conf)

    # Top confusion pairs
    pair_counts: dict = {}
    for t, p in zip(labels[~correct_mask], preds[~correct_mask]):
        pair_counts[(int(t), int(p))] = pair_counts.get((int(t), int(p)), 0) + 1
    if pair_counts:
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:8]
        pair_labels = [f"{SHORT[t]} →\n{SHORT[p]}" for (t,p),_ in top_pairs]
        pair_vals   = [v for _, v in top_pairs]
        y_pos = range(len(pair_labels))
        ax_pair.barh(list(y_pos), pair_vals, color=COLOR,
                     alpha=0.85, edgecolor="white", height=0.6)
        for i, v in enumerate(pair_vals):
            ax_pair.text(v+0.05, i, str(v), va="center", fontsize=9, fontweight="bold")
        ax_pair.set_yticks(list(y_pos))
        ax_pair.set_yticklabels(pair_labels, fontsize=8)
        ax_pair.set_xlabel("Number of Errors")
        ax_pair.set_title("Top Confusion Pairs (true → predicted)", fontsize=10)
        ax_pair.set_xlim(0, max(pair_vals)*1.3)
        sns.despine(ax=ax_pair)
    else:
        ax_pair.text(0.5, 0.5, "No errors!", ha="center", va="center",
                     transform=ax_pair.transAxes, fontsize=14)
        ax_pair.axis("off")

    path = EA_DIR / f"error_analysis_{KEY}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info("Error analysis figure: %s", path); return path


def plot_misclassified_grid(images, labels, preds, probs, max_show=20) -> Path:
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications!", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#27ae60")
        ax.axis("off")
        path = EA_DIR / f"misclassified_grid_{KEY}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig); return path

    confs = np.array([probs[i, preds[i]] for i in wrong_idx])
    order = np.argsort(-confs)
    wrong_idx = wrong_idx[order][:max_show]
    n_show = len(wrong_idx)
    ncols  = 5
    nrows  = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.8, nrows*3.0))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Misclassified Images — {DISPLAY}\n"
        f"(showing {n_show} of {len(np.where(preds!=labels)[0])}, sorted by confidence)",
        fontsize=11, fontweight="bold"
    )
    for plot_i, idx in enumerate(wrong_idx):
        ax  = axes[plot_i]
        img = np.clip(images[idx], 0.0, 1.0)
        ax.imshow(img)
        ax.set_title(
            f"True: {SHORT[labels[idx]]}\nPred: {SHORT[preds[idx]]}\n"
            f"({probs[idx,preds[idx]]*100:.1f}%)",
            fontsize=6.5, color="#c0392b"
        )
        ax.axis("off")
    for ax in axes[n_show:]:
        ax.axis("off")

    plt.tight_layout()
    path = EA_DIR / f"misclassified_grid_{KEY}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    log.info("Misclassified grid: %s", path); return path


def plot_top_errors(images, labels, preds, probs, n_show=10) -> Path:
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassifications!", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="#27ae60")
        ax.axis("off")
        path = EA_DIR / f"top_errors_{KEY}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
        return path

    confs = np.array([probs[i, preds[i]] for i in wrong_idx])
    order = np.argsort(-confs)
    top   = wrong_idx[order[:n_show]]
    ncols = min(5, n_show)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.0, nrows*3.2))
    axes = np.array(axes).flatten()
    fig.suptitle(f"Top-{n_show} Most Confident Wrong Predictions — {DISPLAY}",
                 fontsize=11, fontweight="bold")

    for plot_i, idx in enumerate(top):
        ax   = axes[plot_i]
        img  = np.clip(images[idx], 0.0, 1.0)
        conf = probs[idx, preds[idx]] * 100
        bar_str = "  ".join(
            f"{SHORT[c][0]}:{probs[idx,c]*100:.0f}%" for c in range(4)
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
    path = EA_DIR / f"top_errors_{KEY}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    log.info("Top errors plot: %s", path); return path


# ── summary row ───────────────────────────────────────────────────────────

def append_to_error_summary(labels, preds, probs):
    n_total  = len(labels)
    n_errors = int((preds != labels).sum())
    acc      = (n_total - n_errors) / n_total * 100

    per_class_err = []
    for c in range(4):
        mask = labels == c
        if mask.sum() == 0: continue
        rate = (preds[mask] != c).sum() / mask.sum() * 100
        per_class_err.append((float(rate), CLASS_NAMES[c]))

    highest_err = max(per_class_err, key=lambda x: x[0])
    lowest_err  = min(per_class_err, key=lambda x: x[0])

    pair_counts: dict = {}
    wrong_mask = preds != labels
    for t, p in zip(labels[wrong_mask], preds[wrong_mask]):
        pair_counts[(int(t), int(p))] = pair_counts.get((int(t), int(p)), 0) + 1

    if pair_counts:
        top_pair = max(pair_counts, key=lambda k: pair_counts[k])
        top_count = pair_counts[top_pair]
    else:
        top_pair = (0, 0); top_count = 0

    max_conf = probs.max(axis=1)
    hi_err   = int(((preds != labels) & (max_conf >= 0.90)).sum())
    pct_hi   = round(hi_err / n_errors * 100, 2) if n_errors > 0 else 0.0

    new_row = {
        "model_key":                   KEY,
        "display_name":                DISPLAY,
        "total_test":                  n_total,
        "total_errors":                n_errors,
        "error_rate_pct":              round(n_errors/n_total*100, 4),
        "accuracy_pct":                round(acc, 4),
        "most_confused_true":          CLASS_NAMES[top_pair[0]],
        "most_confused_pred":          CLASS_NAMES[top_pair[1]],
        "most_confused_count":         top_count,
        "highest_error_class":         highest_err[1],
        "highest_error_rate_pct":      round(highest_err[0], 4),
        "lowest_error_class":          lowest_err[1],
        "lowest_error_rate_pct":       round(lowest_err[0], 4),
        "pct_errors_above_90pct_conf": pct_hi,
    }

    # Read existing summary
    summary_path = OUTPUTS_DIR / "error_analysis_summary.csv"
    existing_rows = []
    fields = list(new_row.keys())
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or fields
            for r in reader:
                if r["model_key"] != KEY:  # remove old entry if re-run
                    existing_rows.append(r)

    existing_rows.append(new_row)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(existing_rows)
    log.info("Error analysis summary updated: %s", summary_path)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("LeafIQ -- Phase 5: Hybrid Model Error Analysis")
    log.info("=" * 60)

    probs, preds, labels, paths = load_and_predict()

    # Load original 224x224 test images for visualisation
    log.info("Loading test images for visualisation ...")
    images = np.load(CACHE_DIR / "test_images.npy", mmap_mode="r")

    log.info("Saving CSVs ...")
    _, wrong_rows = save_misclassified_csv(labels, preds, probs, paths)
    save_per_class_errors_csv(labels, preds)
    save_confidence_csv(labels, preds, probs)

    log.info("Generating plots ...")
    plot_error_analysis(labels, preds, probs, wrong_rows)
    plot_misclassified_grid(images, labels, preds, probs)
    plot_top_errors(images, labels, preds, probs)

    append_to_error_summary(labels, preds, probs)

    n_errors = int((preds != labels).sum())
    log.info("\nError analysis complete.")
    log.info("  Total errors : %d / %d", n_errors, len(labels))
    log.info("  Accuracy     : %.2f%%", (len(labels)-n_errors)/len(labels)*100)
    log.info("  Outputs      : %s", EA_DIR)


if __name__ == "__main__":
    main()
